import json
import os
from typing import Optional

from btree_core import BTreeManager, SensorRecord


class SingleFileLogger:
    """Append-only JSONL logger that writes records coming from the B-tree."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.base_dir, "readings.jsonl")

    def append_from_record(self, record: SensorRecord) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": record.timestamp,
                "object_id": record.object_id,
                "object_type": record.object_type,
                "position": list(record.position),
                "velocity": record.velocity,
                "distance_to_our_car": record.distance_to_our_car,
            }) + "\n")

    def path(self) -> str:
        return self.jsonl_path


