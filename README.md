Autonomous Vehicle Perception Indexing Prototype (B-tree)

Run

1. Ensure Python 3.9+ is installed.
2. Generate sensor data file:

```
python3 sensor_data.py
```

3. Run the B-tree indexing and decision logic:

```
python3 btree_main.py
```

4. GUI (browser-based via Streamlit):

```
pip3 install streamlit pandas
streamlit run app_streamlit.py
```

Auto-launch (opens your browser)

```
python3 run_app.py
```

What it does

- `sensor_data.py` simulates sensor readings (one per second) and saves them to `sensor_readings.json`.
- `btree_main.py` loads the saved readings, inserts into a minimal B-tree keyed by `timestamp`, and after each insert queries the latest reading to decide whether to apply brakes.
- Prints "Brake Applied!" once distance <= 5 m, then stops.
- Console output includes a per-second log of timestamp and distance.

Data fields per record

- `timestamp` (seconds)
- `object_id` (e.g., 101)
- `object_type` ("car")
- `position` (x, y, z) — 1D along x; y=z=0
- `velocity` (m/s) — the ahead car's velocity (0 in this scenario)
- `distance_to_our_car` (m)

Notes

- This is a didactic, in-memory prototype of a B-tree for time-indexed sensor data.
- The B-tree here supports `insert`, `search(key)`, and `get_latest()`.

GUI

- Real-time generation of sensor readings (1 Hz) with logging to a single append-only JSONL file.
- Visual display of latest distance and brake status; simple live plot of distance vs time.
- Start/Stop/Reset controls.

Multi-car simulation (decisions + B-tree traversal)

```
pip3 install streamlit pandas altair
streamlit run multicar_streamlit.py
```
What it shows:
- Three cars moving in 1D (positions update each second)
- Continuous sensor logging into the B-tree (in-order keys view)
- Decisions per tick: Brake / Slow Down / Maintain Speed

Legacy single-file demo

If you prefer a one-file demo that both generates data and runs the B-tree simulation in one go, you can also run:

```
python3 prototype_btree_av.py
```

Multilane simulation (2 lanes, 5 cars)

```
pip3 install streamlit pandas altair
streamlit run multilane_streamlit.py
```
What it shows:
- Two lanes with five cars (O, A, B in lane 1; C, D in lane 2)
- Random speed jitter, continuous movement
- Single-file JSONL logging via B-tree (composite keys)
- Live decision display (Brake / Slow Down / Maintain / Change Lane)
- B-tree in-order traversal keys live



