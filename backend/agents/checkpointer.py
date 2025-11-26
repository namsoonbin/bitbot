"""
MongoDB Checkpointer for LangGraph (v2 BaseCheckpointSaver)

Persists checkpoints and pending writes so agent state can be resumed.
"""

from __future__ import annotations

from datetime import datetime
import os
from typing import Any, Dict, Iterator, Optional, Sequence, AsyncIterator

from bson.binary import Binary
from loguru import logger
from pymongo import MongoClient

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    PendingWrite,
)
from langgraph.checkpoint.memory import (
    WRITES_IDX_MAP,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class MongoDBCheckpointSaver(BaseCheckpointSaver):
    """
    MongoDB-based checkpoint saver for LangGraph.

    Stores checkpoints and pending writes so workflows can be paused/resumed.
    """

    def __init__(
        self,
        mongo_host: str = "localhost",
        mongo_port: int = 27017,
        mongo_user: str = "hats_user",
        mongo_password: str = "hats_password",
        db_name: str = "hats_trading",
        collection_name: str = "agent_checkpoints",
        *,
        serde=None,
    ):
        super().__init__(serde=serde)

        connection_string = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/"
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.writes_collection = self.db[f"{collection_name}_writes"]

        self.collection.create_index(
            [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", 1)], unique=True
        )
        self.collection.create_index([("thread_id", 1), ("created_at", -1)])
        self.writes_collection.create_index(
            [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", 1)], unique=True
        )

        logger.info(
            f"MongoDB Checkpointer initialized: db={db_name}, collection={collection_name}"
        )

    # ---- helpers -----------------------------------------------------
    def _encode(self, value: Any) -> Dict[str, Any]:
        fmt, payload = self.serde.dumps_typed(value)
        return {"format": fmt, "data": Binary(payload)}

    def _decode(self, payload: Optional[Dict[str, Any]]) -> Any:
        if not payload:
            return None
        return self.serde.loads_typed((payload["format"], bytes(payload["data"])))

    def _load_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[PendingWrite] | None:
        doc = self.writes_collection.find_one(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        )
        if not doc or not doc.get("writes"):
            return None

        pending: list[PendingWrite] = []
        for write in doc["writes"]:
            pending.append(
                (
                    write["task_id"],
                    write["channel"],
                    self._decode(write["value"]),
                )
            )
        return pending or None

    # ---- required BaseCheckpointSaver methods -----------------------
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "") or ""
        if not thread_id:
            logger.warning("No thread_id in config; cannot retrieve checkpoint.")
            return None

        checkpoint_id = get_checkpoint_id(config) if config.get("configurable") else None
        query: Dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        if checkpoint_id:
            query["checkpoint_id"] = checkpoint_id

        doc = self.collection.find_one(query, sort=[("created_at", -1)])
        if not doc:
            logger.debug(f"No checkpoint found for thread={thread_id}")
            return None

        checkpoint: Checkpoint = self._decode(doc["checkpoint"])
        metadata: CheckpointMetadata = self._decode(doc["metadata"]) or {}
        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": doc.get("parent_checkpoint_id"),
                }
            }
            if doc.get("parent_checkpoint_id")
            else None
        )
        pending_writes = self._load_writes(
            thread_id, checkpoint_ns, doc["checkpoint_id"]
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": doc["checkpoint_id"],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        query: Dict[str, Any] = {}
        if config and config.get("configurable"):
            cfg = config["configurable"]
            if cfg.get("thread_id"):
                query["thread_id"] = cfg["thread_id"]
            if cfg.get("checkpoint_ns") is not None:
                query["checkpoint_ns"] = cfg.get("checkpoint_ns", "")
            if get_checkpoint_id(config):
                query["checkpoint_id"] = get_checkpoint_id(config)

        if before and get_checkpoint_id(before):
            before_doc = self.collection.find_one(
                {"checkpoint_id": get_checkpoint_id(before)}
            )
            if before_doc and before_doc.get("created_at"):
                query["created_at"] = {"$lt": before_doc["created_at"]}

        cursor = self.collection.find(query).sort("created_at", -1)
        if limit:
            cursor = cursor.limit(limit)

        for doc in cursor:
            metadata: CheckpointMetadata = self._decode(doc["metadata"]) or {}
            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue

            checkpoint: Checkpoint = self._decode(doc["checkpoint"])
            parent_config = (
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc.get("checkpoint_ns", ""),
                        "checkpoint_id": doc.get("parent_checkpoint_id"),
                    }
                }
                if doc.get("parent_checkpoint_id")
                else None
            )
            pending_writes = self._load_writes(
                doc["thread_id"], doc.get("checkpoint_ns", ""), doc["checkpoint_id"]
            )

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc.get("checkpoint_ns", ""),
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        cfg = config.get("configurable", {})
        thread_id = cfg.get("thread_id")
        checkpoint_ns = cfg.get("checkpoint_ns", "") or ""
        parent_checkpoint_id = cfg.get("checkpoint_id")

        if not thread_id:
            raise ValueError("thread_id is required in config.configurable")

        checkpoint_id = checkpoint["id"]
        checkpoint_payload = self._encode(checkpoint)
        metadata_payload = self._encode(get_checkpoint_metadata(config, metadata))

        doc = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "checkpoint": checkpoint_payload,
            "metadata": metadata_payload,
            "new_versions": new_versions,
            "created_at": datetime.utcnow(),
        }

        self.collection.update_one(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
            {"$set": doc},
            upsert=True,
        )

        self.writes_collection.update_one(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
            {"$setOnInsert": {"writes": []}},
            upsert=True,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        cfg = config.get("configurable", {})
        thread_id = cfg.get("thread_id")
        checkpoint_ns = cfg.get("checkpoint_ns", "") or ""
        checkpoint_id = cfg.get("checkpoint_id")

        if not thread_id or not checkpoint_id:
            logger.warning("Cannot save writes without thread_id and checkpoint_id")
            return

        filter_query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        doc = self.writes_collection.find_one(filter_query) or {"writes": []}

        existing_keys = {
            (w["task_id"], w.get("write_idx", idx))
            for idx, w in enumerate(doc.get("writes", []))
        }

        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            key = (task_id, write_idx)
            if write_idx >= 0 and key in existing_keys:
                continue

            doc.setdefault("writes", []).append(
                {
                    "task_id": task_id,
                    "channel": channel,
                    "value": self._encode(value),
                    "task_path": task_path,
                    "write_idx": write_idx,
                }
            )
            existing_keys.add(key)

        self.writes_collection.update_one(
            filter_query,
            {"$set": {"writes": doc.get("writes", [])}},
            upsert=True,
        )

    def delete_thread(self, thread_id: str) -> None:
        self.collection.delete_many({"thread_id": thread_id})
        self.writes_collection.delete_many({"thread_id": thread_id})
        logger.info(f"Deleted checkpoints for thread={thread_id}")

    # ---- async wrappers ---------------------------------------------
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)

    # ---- lifecycle --------------------------------------------------
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("MongoDB Checkpointer connection closed")


def create_checkpointer(
    mongo_host: Optional[str] = None,
    mongo_port: Optional[int] = None,
    mongo_user: Optional[str] = None,
    mongo_password: Optional[str] = None,
    db_name: Optional[str] = None,
) -> MongoDBCheckpointSaver:
    """
    Factory to create a MongoDB checkpointer using environment defaults.
    """
    return MongoDBCheckpointSaver(
        mongo_host=mongo_host or os.getenv("MONGO_HOST", "localhost"),
        mongo_port=mongo_port or int(os.getenv("MONGO_PORT", 27017)),
        mongo_user=mongo_user or os.getenv("MONGO_USER", "hats_user"),
        mongo_password=mongo_password or os.getenv("MONGO_PASSWORD", "hats_password"),
        db_name=db_name or os.getenv("MONGO_DB", "hats_trading"),
    )


if __name__ == "__main__":
    from .state import create_initial_state, MarketData
    from .graph import compile_trading_graph
    import uuid

    sample_market_data = MarketData(
        timestamp=datetime.now(),
        symbol="BTC/USDT",
        current_price=50000.0,
        open=49500.0,
        high=50500.0,
        low=49000.0,
        volume=1000000.0,
        price_change_24h=500.0,
        price_change_pct_24h=1.01,
    )

    session_id = str(uuid.uuid4())
    initial_state = create_initial_state(session_id, sample_market_data)

    checkpointer = create_checkpointer()
    app = compile_trading_graph(checkpointer=checkpointer)

    logger.info("Graph compiled with MongoDB checkpointer (manual smoke test).")
    checkpointer.close()
