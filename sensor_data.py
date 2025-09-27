import json
from dataclasses import dataclass, asdict
from typing import List, Tuple


OUTPUT_PATH = "/Users/vidyadharpothula/dsa_project/sensor_readings.json"


@dataclass
class SensorRecord:
    timestamp: int  # seconds
    object_id: int
    object_type: str
    position: Tuple[float, float, float]
    velocity: float  # m/s (object's velocity)
    distance_to_our_car: float  # meters


def generate_sensor_readings() -> List[SensorRecord]:
    our_start_x = 0.0
    our_velocity = 30.0  # m/s
    other_car_start_x = 500.0
    other_car_velocity = 0.0
    safety_threshold = 5.0

    readings: List[SensorRecord] = []
    t = 0
    while True:
        our_x = our_start_x + our_velocity * t
        other_x = other_car_start_x + other_car_velocity * t
        distance = other_x - our_x

        record = SensorRecord(
            timestamp=t,
            object_id=101,
            object_type="car",
            position=(other_x, 0.0, 0.0),
            velocity=other_car_velocity,
            distance_to_our_car=distance,
        )
        readings.append(record)

        if distance <= safety_threshold:
            break
        t += 1

    return readings


def save_readings_to_json(readings: List[SensorRecord], path: str) -> None:
    payload = [asdict(r) for r in readings]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    readings = generate_sensor_readings()
    save_readings_to_json(readings, OUTPUT_PATH)
    print(f"Saved {len(readings)} readings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


