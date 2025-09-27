import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd
import altair as alt

from btree_core import BTreeManager, SensorRecord
from data_gen import DataGenerator
from file_logger import SingleFileLogger


@dataclass
class SensorRecord:
    timestamp: int
    object_id: int
    object_type: str
    position: Tuple[float, float, float]
    velocity: float
    distance_to_our_car: float


class BTreeNode:
    def __init__(self, min_degree: int, leaf: bool) -> None:
        self.min_degree = min_degree
        self.leaf = leaf
        self.keys: List[int] = []
        self.values: List[SensorRecord] = []
        self.children: List["BTreeNode"] = []


class BTree:
    def __init__(self, min_degree: int = 2) -> None:
        if min_degree < 2:
            raise ValueError("min_degree must be >= 2")
        self.t = min_degree
        self.root = BTreeNode(min_degree=self.t, leaf=True)

    def split_child(self, parent: BTreeNode, index: int) -> None:
        t = self.t
        node_to_split = parent.children[index]
        new_node = BTreeNode(min_degree=t, leaf=node_to_split.leaf)
        mid_key = node_to_split.keys[t - 1]
        mid_val = node_to_split.values[t - 1]
        new_node.keys = node_to_split.keys[t:]
        new_node.values = node_to_split.values[t:]
        if not node_to_split.leaf:
            new_node.children = node_to_split.children[t:]
        node_to_split.keys = node_to_split.keys[: t - 1]
        node_to_split.values = node_to_split.values[: t - 1]
        if not node_to_split.leaf:
            node_to_split.children = node_to_split.children[:t]
        parent.keys.insert(index, mid_key)
        parent.values.insert(index, mid_val)
        parent.children.insert(index + 1, new_node)

    def insert_non_full(self, node: BTreeNode, key: int, value: SensorRecord) -> None:
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(0)
            node.values.append(value)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.t - 1):
                self.split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self.insert_non_full(node.children[i], key, value)

    def insert(self, key: int, value: SensorRecord) -> None:
        root = self.root
        if len(root.keys) == (2 * self.t - 1):
            new_root = BTreeNode(min_degree=self.t, leaf=False)
            new_root.children.append(root)
            self.split_child(new_root, 0)
            self.root = new_root
            self.insert_non_full(new_root, key, value)
        else:
            self.insert_non_full(root, key, value)

    def get_latest(self) -> Optional[Tuple[int, SensorRecord]]:
        node = self.root
        if not node.keys:
            return None
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1], node.values[-1]


class RealtimeLogger:
    """Single-file logger (append-only JSONL)."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.base_dir, "readings.jsonl")

    def log(self, record: SensorRecord) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")


def generate_record(t: int) -> SensorRecord:
    our_start_x = 0.0
    our_velocity = 30.0
    other_car_start_x = 500.0
    other_car_velocity = 0.0
    our_x = our_start_x + our_velocity * t
    other_x = other_car_start_x + other_car_velocity * t
    distance = other_x - our_x
    return SensorRecord(
        timestamp=t,
        object_id=101,
        object_type="car",
        position=(other_x, 0.0, 0.0),
        velocity=other_car_velocity,
        distance_to_our_car=distance,
    )


def main() -> None:
    st.set_page_config(page_title="AV Perception Indexing - B-tree", layout="wide")
    st.title("Autonomous Vehicle Perception Indexing (B-tree)")

    if "running" not in st.session_state:
        st.session_state.running = False
        st.session_state.t = 0
        st.session_state.manager = BTreeManager()
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.log_dir = os.path.join(
            "/Users/vidyadharpothula/dsa_project", f"logs_streamlit_{run_stamp}"
        )
        st.session_state.logger = SingleFileLogger(st.session_state.log_dir)
        st.session_state.generator = DataGenerator(st.session_state.manager)

    live_tab, review_tab = st.tabs(["Live", "Review"])

    with live_tab:
        left, right = st.columns([2, 3])

        with left:
            st.subheader("Controls")
            c1, c2, c3 = st.columns(3)
            if c1.button("Start"):
                st.session_state.running = True
            if c2.button("Stop"):
                st.session_state.running = False
            if c3.button("Reset"):
                st.session_state.running = False
                st.session_state.t = 0
                st.session_state.manager = BTreeManager()
                run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.log_dir = os.path.join(
                    "/Users/vidyadharpothula/dsa_project", f"logs_streamlit_{run_stamp}"
                )
                st.session_state.logger = SingleFileLogger(st.session_state.log_dir)
                st.session_state.generator = DataGenerator(st.session_state.manager)

            st.write(f"t = {st.session_state.t} s")

            history = st.session_state.manager.get_history()
            latest_distance = history[-1][1] if history else None
            if latest_distance is None:
                st.info("Latest distance: -")
            else:
                if latest_distance <= 5.0:
                    st.error(f"Latest distance: {latest_distance:.2f} m (BRAKE APPLIED)")
                else:
                    st.success(f"Latest distance: {latest_distance:.2f} m")

            st.subheader("Log file (append-only)")
            jsonl_path = st.session_state.logger.path()
            st.code(jsonl_path)

        with right:
            st.subheader("Distance vs Time")
            history = st.session_state.manager.get_history()
            if history:
                df = pd.DataFrame(history, columns=["t", "distance"]).astype({"t": float, "distance": float})
                max_d = max(10.0, df["distance"].max())
                min_d = min(0.0, df["distance"].min())
                base = alt.Chart(df).mark_line(color="#1f77b4", point=True).encode(
                    x=alt.X("t:Q", title="time (s)"),
                    y=alt.Y("distance:Q", title="distance (m)", scale=alt.Scale(domain=[min_d, max_d])),
                    tooltip=["t", "distance"],
                ).properties(width="container", height=300)
                threshold = alt.Chart(pd.DataFrame({"y": [5.0]})).mark_rule(color="red").encode(y="y")
                chart = (base + threshold).interactive()
                st.altair_chart(chart, use_container_width=True, theme=None)
            else:
                st.write("(no data yet)")

    with review_tab:
        st.subheader("Run Review")
        history = st.session_state.manager.get_history()
        if history and not st.session_state.running:
            df = pd.DataFrame(history, columns=["t", "distance"]).copy()
            brake_rows = df[df["distance"] <= 5.0]
            brake_time = int(brake_rows.iloc[0]["t"]) if not brake_rows.empty else None

            st.markdown("**Workflow summary**")
            st.write("- Sensor readings generated at 1 Hz and logged to files.")
            st.write("- Each reading inserted into a B-tree keyed by timestamp.")
            st.write("- Latest distance retrieved after each insert; braking when distance ≤ 5 m.")
            if brake_time is not None:
                st.success(f"Brake Applied at t = {brake_time}s (distance ≤ 5 m)")
            else:
                st.info("Brake was not applied in this run.")

            st.markdown("**Table of readings**")
            st.dataframe(df, use_container_width=True)

            st.markdown("**Log file**")
            jsonl_path = st.session_state.logger.path()
            if os.path.exists(jsonl_path):
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()[-5:]
                    st.markdown("**readings.jsonl (last 5 lines)**")
                    st.code("".join(lines) if lines else "(empty)")
                except Exception as ex:
                    st.warning(f"Could not read JSONL log: {ex}")
        else:
            st.info("Review becomes available after a run completes (when the simulation stops).")

    # Simulation tick
    if st.session_state.running:
        rec = st.session_state.generator.next()
        st.session_state.logger.append_from_record(rec)
        latest = st.session_state.manager.get_latest()
        if latest is not None and latest[1].distance_to_our_car <= 5.0:
            st.session_state.running = False
        st.session_state.t += 1
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()


