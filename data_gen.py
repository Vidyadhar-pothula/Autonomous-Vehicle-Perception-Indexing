from dataclasses import asdict
from typing import Callable

from btree_core import SensorRecord, BTreeManager


class DataGenerator:
    def __init__(self, btree: BTreeManager) -> None:
        self._btree = btree
        self._t = 0
        self._our_start_x = 0.0
        self._our_velocity = 30.0
        self._other_car_start_x = 500.0
        self._other_car_velocity = 0.0
        self._safety_threshold = 5.0

    def next(self) -> SensorRecord:
        our_x = self._our_start_x + self._our_velocity * self._t
        other_x = self._other_car_start_x + self._other_car_velocity * self._t
        distance = other_x - our_x
        record = SensorRecord(
            timestamp=self._t,
            object_id=101,
            object_type="car",
            position=(other_x, 0.0, 0.0),
            velocity=self._other_car_velocity,
            distance_to_our_car=distance,
        )
        # Insert to B-tree first (authoritative store for UI queries)
        self._btree.insert_record(record)
        self._t += 1
        return record

    def should_brake(self, record: SensorRecord) -> bool:
        return record.distance_to_our_car <= self._safety_threshold


