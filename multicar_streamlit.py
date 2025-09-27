import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import altair as alt

from btree_core import BTreeManager, SensorRecord


@dataclass
class Car:
    car_id: int
    position: float
    speed: float


def decision_logic(pos_o: float, v_o: float, others: List[Car]) -> Tuple[str, float]:
    best_action = "Maintain Speed"
    new_speed = v_o
    for car in others:
        distance = car.position - pos_o
        rel_speed = v_o - car.speed
        predicted = distance - rel_speed * 3.0  # 3-second prediction
        if predicted <= 10.0:
            return "Brake", max(0.0, v_o - 20.0)
        if rel_speed > 10.0 and distance > 50.0:
            best_action = "Slow Down"
            new_speed = max(0.0, v_o - 5.0)
        if distance > 200.0 and best_action == "Maintain Speed":
            best_action = "Maintain/Speed Up"
            new_speed = v_o + 2.0
    return best_action, new_speed


def main() -> None:
    st.set_page_config(page_title="Multi-car AV B-tree Simulation", layout="wide")
    st.title("Multi-car Autonomous Driving with B-tree Logging")

    # State init
    if "t" not in st.session_state:
        st.session_state.t = 0
        st.session_state.manager = BTreeManager()
        st.session_state.cars: Dict[int, Car] = {
            0: Car(0, 0.0, 40.0),
            1: Car(1, 500.0, 20.0),
            2: Car(2, 300.0, 25.0),
        }
        st.session_state.running = False
        st.session_state.decision = "-"

    left, right = st.columns([2, 3])

    with left:
        st.subheader("Controls")
        c1, c2, c3 = st.columns(3)
        if c1.button("Start"):
            st.session_state.running = True
        if c2.button("Stop"):
            st.session_state.running = False
        if c3.button("Reset"):
            st.session_state.t = 0
            st.session_state.manager = BTreeManager()
            st.session_state.cars = {
                0: Car(0, 0.0, 40.0),
                1: Car(1, 500.0, 20.0),
                2: Car(2, 300.0, 25.0),
            }
            st.session_state.running = False
            st.session_state.decision = "-"

        st.write(f"t = {st.session_state.t} s")
        st.write(f"Decision: {st.session_state.decision}")

        st.subheader("B-tree inorder keys")
        keys = st.session_state.manager.inorder_keys()
        st.code(", ".join(map(str, keys[-50:])) if keys else "(empty)")

    with right:
        st.subheader("1D Road Positions")
        df = pd.DataFrame([
            {"car": "O", "position": st.session_state.cars[0].position},
            {"car": "A", "position": st.session_state.cars[1].position},
            {"car": "B", "position": st.session_state.cars[2].position},
        ])
        road = alt.Chart(df).mark_point(size=200).encode(
            x=alt.X("position:Q", title="position (m)"),
            y=alt.Y("car:N", title="car", sort=["O", "A", "B"]),
            color="car:N",
        ).properties(height=200)
        st.altair_chart(road, use_container_width=True)

        st.subheader("Distance vs Time (per other car)")
        hist = st.session_state.manager.get_history()
        if hist:
            # History stores only O vs ahead car distance in prior app; here build new
            # For this demo, compute distances to A and B now
            # Not persisted; purely for charting
            t = st.session_state.t
            # Reconstruct naive arrays (simple live snapshot)
            positions = {
                "O": st.session_state.cars[0].position,
                "A": st.session_state.cars[1].position,
                "B": st.session_state.cars[2].position,
            }
            # Pressing Start from t=0 will build a consistent time-series
            # For robustness, display only current snapshot if early
            data = [
                {"car": "A", "t": t, "distance": positions["A"] - positions["O"]},
                {"car": "B", "t": t, "distance": positions["B"] - positions["O"]},
            ]
            chart_df = pd.DataFrame(data)
            base = alt.Chart(chart_df).mark_line(point=True).encode(
                x="t:Q", y="distance:Q", color="car:N"
            )
            threshold = alt.Chart(pd.DataFrame({"y": [10.0]})).mark_rule(color="red").encode(y="y")
            st.altair_chart(base + threshold, use_container_width=True)

    # Tick
    if st.session_state.running:
        cars = st.session_state.cars
        # Sensor logs for each other car
        car_o = cars[0]
        others = [cars[1], cars[2]]
        for other in others:
            distance = other.position - car_o.position
            key = other.car_id * 1000 + st.session_state.t
            rec = SensorRecord(
                timestamp=st.session_state.t,
                object_id=other.car_id,
                object_type="car",
                position=(other.position, 0.0, 0.0),
                velocity=other.speed,
                distance_to_our_car=distance,
            )
            # B-tree insert with composite key
            st.session_state.manager.insert_with_key(key, rec)

        # Decision and speed adjustment
        action, new_speed = decision_logic(car_o.position, car_o.speed, others)
        st.session_state.decision = action
        car_o.speed = new_speed
        if action == "Brake":
            st.session_state.running = False

        # Update positions
        for c in cars.values():
            c.position += c.speed

        st.session_state.t += 1
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()


