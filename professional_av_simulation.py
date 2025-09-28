import random
import time
import math
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt

from btree_core import BTreeManager, SensorRecord
from file_logger import SingleFileLogger


class SignalState(Enum):
    GREEN = auto()
    YELLOW = auto()
    RED = auto()


@dataclass
class TrafficSignal:
    position: float = 500.0
    state: SignalState = SignalState.GREEN
    state_timer: int = 0
    cycle_times = {
        SignalState.GREEN: 80,
        SignalState.YELLOW: 20,
        SignalState.RED: 60
    }
    
    def update(self) -> None:
        self.state_timer += 1
        if self.state_timer >= self.cycle_times[self.state]:
            if self.state == SignalState.GREEN:
                self.state = SignalState.YELLOW
            elif self.state == SignalState.YELLOW:
                self.state = SignalState.RED
            else:
                self.state = SignalState.GREEN
            self.state_timer = 0
    
    def get_state_name(self) -> str:
        return self.state.name.title()


@dataclass
class Car:
    car_id: int
    lane_id: int
    position: float
    speed: float  # km/h
    max_speed: float = 80.0
    acceleration: float = 3.0  # m/s²
    deceleration: float = 5.0  # m/s²
    length: float = 4.5
    target_speed: float = 65.0
    original_lane: int = 2  # Track original lane for overtaking
    
    def update_position(self, dt: float = 0.05) -> None:
        """Update position based on current speed"""
        self.position += self.speed * (dt / 3.6)  # Convert km/h to m/s
        
    def accelerate(self, dt: float = 0.05) -> None:
        """Gradually increase speed"""
        self.speed = min(self.speed + self.acceleration * dt * 3.6, self.max_speed)
        
    def decelerate(self, dt: float = 0.05) -> None:
        """Gradually decrease speed"""
        self.speed = max(0, self.speed - self.deceleration * dt * 3.6)
        
    def maintain_speed(self, dt: float = 0.05) -> None:
        """Maintain current speed with small variations"""
        self.speed += random.uniform(-0.3, 0.3)
        self.speed = max(0, min(self.speed, self.max_speed))
        
    def speed_up(self, dt: float = 0.05) -> None:
        """Speed up for overtaking"""
        self.speed = min(self.speed + self.acceleration * dt * 3.6 * 2.0, self.max_speed)
        
    def emergency_brake(self, dt: float = 0.05) -> None:
        """Emergency braking"""
        self.speed = max(0, self.speed - self.deceleration * dt * 3.6 * 3.0)


class DetailedLogger:
    """Enhanced logging system for detailed B-tree data tracking"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.detailed_log_path = os.path.join(self.base_dir, "detailed_log.jsonl")
        self.decision_log_path = os.path.join(self.base_dir, "decisions.jsonl")
        
    def log_detailed_data(self, timestamp: int, our_car: Car, all_cars: List[Car], 
                         btree_keys: List[int], decision: str, surrounding_data: Dict):
        """Log detailed simulation data"""
        log_entry = {
            "timestamp": timestamp,
            "our_car": {
                "id": our_car.car_id,
                "lane": our_car.lane_id,
                "position": our_car.position,
                "speed": our_car.speed,
                "original_lane": our_car.original_lane
            },
            "other_cars": [
                {
                    "id": car.car_id,
                    "lane": car.lane_id,
                    "position": car.position,
                    "speed": car.speed,
                    "distance_from_our_car": car.position - our_car.position
                }
                for car in all_cars if car.car_id != our_car.car_id
            ],
            "btree_data": {
                "total_keys": len(btree_keys),
                "recent_keys": btree_keys[-10:],
                "surrounding_cars_count": {
                    "ahead_same_lane": len(surrounding_data.get('ahead_same_lane', [])),
                    "behind_same_lane": len(surrounding_data.get('behind_same_lane', [])),
                    "left_lane": len(surrounding_data.get('left_lane', [])),
                    "right_lane": len(surrounding_data.get('right_lane', []))
                }
            },
            "decision": decision,
            "surrounding_details": surrounding_data
        }
        
        with open(self.detailed_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def log_decision(self, timestamp: int, decision: str, context: Dict):
        """Log driving decisions with context"""
        decision_entry = {
            "timestamp": timestamp,
            "decision": decision,
            "context": context,
            "datetime": datetime.now().isoformat()
        }
        
        with open(self.decision_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision_entry) + "\n")


class SensorDataProcessor:
    """Enhanced sensor data processing with detailed logging"""
    
    def __init__(self, btree_manager: BTreeManager):
        self.btree_manager = btree_manager
        
    def get_surrounding_cars(self, our_car: Car, all_cars: List[Car]) -> Dict:
        """Get detailed surrounding cars data using B-tree sensor data"""
        surrounding = {
            'ahead_same_lane': [],
            'behind_same_lane': [],
            'left_lane': [],
            'right_lane': [],
            'all_nearby': []
        }
        
        # Get recent sensor data from B-tree (last 100 readings)
        recent_keys = self.btree_manager.inorder_keys()[-100:]
        
        for key in recent_keys:
            record = self.btree_manager.search(key)
            if not record or record.object_id == our_car.car_id:
                continue
                
            other_pos = record.position[0]
            other_lane = getattr(record, 'lane_id', 1)
            distance = other_pos - our_car.position
            
            # Only consider cars within 300m for better decision making
            if abs(distance) < 300:
                car_data = {
                    'car_id': record.object_id,
                    'position': other_pos,
                    'distance': abs(distance),
                    'speed': record.velocity,
                    'lane': other_lane,
                    'relative_speed': our_car.speed - record.velocity,
                    'timestamp': record.timestamp
                }
                
                surrounding['all_nearby'].append(car_data)
                
                if other_lane == our_car.lane_id:
                    if distance > 0:
                        surrounding['ahead_same_lane'].append(car_data)
                    else:
                        surrounding['behind_same_lane'].append(car_data)
                elif other_lane == our_car.lane_id - 1 and our_car.lane_id > 1:
                    surrounding['left_lane'].append(car_data)
                elif other_lane == our_car.lane_id + 1 and our_car.lane_id < 3:
                    surrounding['right_lane'].append(car_data)
        
        # Sort by distance for better decision making
        for key in surrounding:
            surrounding[key].sort(key=lambda x: x['distance'])
            
        return surrounding
    
    def is_lane_clear(self, target_lane: int, our_car: Car, gap_m: float = 25.0) -> bool:
        """Check if a lane is clear for lane change with detailed analysis"""
        surrounding = self.get_surrounding_cars(our_car, [])
        
        if target_lane == our_car.lane_id - 1:  # Left lane
            for car in surrounding['left_lane']:
                if car['distance'] < gap_m:
                    return False
        elif target_lane == our_car.lane_id + 1:  # Right lane
            for car in surrounding['right_lane']:
                if car['distance'] < gap_m:
                    return False
                    
        return True


class ProfessionalAutonomousDriver:
    """Professional autonomous driving system with proper overtaking logic"""
    
    def __init__(self, sensor_processor: SensorDataProcessor):
        self.sensor_processor = sensor_processor
        self.overtaking_state = None
        self.overtaking_timer = 0
        self.target_car_id = None
        self.target_lane = None
        
    def decide_action(self, our_car: Car, signal: TrafficSignal) -> str:
        """Main decision making function with proper overtaking logic"""
        surrounding = self.sensor_processor.get_surrounding_cars(our_car, [])
        
        # Check for traffic signal
        if self._should_stop_for_signal(our_car, signal):
            return "Stop for Signal"
        
        # Check for emergency situations
        emergency_action = self._check_emergency(our_car, surrounding)
        if emergency_action:
            return emergency_action
        
        # Handle ongoing overtaking with proper state machine
        if self.overtaking_state:
            return self._handle_overtaking(our_car, surrounding)
        
        # Check for overtaking opportunities
        overtake_action = self._check_overtaking_opportunity(our_car, surrounding)
        if overtake_action:
            return overtake_action
        
        # Normal driving decisions
        return self._normal_driving(our_car, surrounding)
    
    def _should_stop_for_signal(self, our_car: Car, signal: TrafficSignal) -> bool:
        """Check if we should stop for traffic signal"""
        if our_car.position > signal.position:
            return False
            
        distance_to_signal = signal.position - our_car.position
        stopping_distance = (our_car.speed ** 2) / (2 * 5.0)  # Using deceleration
        
        if signal.state == SignalState.RED and distance_to_signal < stopping_distance * 1.5:
            return True
        elif signal.state == SignalState.YELLOW and distance_to_signal < stopping_distance:
            return True
            
        return False
    
    def _check_emergency(self, our_car: Car, surrounding: Dict) -> Optional[str]:
        """Check for emergency situations requiring immediate action"""
        # Check for cars too close ahead
        for car in surrounding['ahead_same_lane']:
            if car['distance'] < 15:  # Very close
                return "Emergency Braking"
            elif car['distance'] < 30 and car['relative_speed'] > 15:  # Approaching fast
                return "Emergency Braking"
        
        return None
    
    def _check_overtaking_opportunity(self, our_car: Car, surrounding: Dict) -> Optional[str]:
        """Check if we should start overtaking - proper logic for all lanes"""
        # Find slowest car ahead in same lane
        slowest_ahead = None
        min_speed = float('inf')
        
        for car in surrounding['ahead_same_lane']:
            if car['distance'] < 80 and car['speed'] < min_speed:
                min_speed = car['speed']
                slowest_ahead = car
        
        # Only overtake if car is significantly slower (at least 20% slower)
        if slowest_ahead and slowest_ahead['speed'] < our_car.speed * 0.8:
            # Store original lane
            our_car.original_lane = our_car.lane_id
            
            # Determine best overtaking lane
            if our_car.lane_id == 1:  # Leftmost lane - can only go right
                if self.sensor_processor.is_lane_clear(2, our_car, 30):
                    self.overtaking_state = 'changing_right'
                    self.overtaking_timer = 0
                    self.target_car_id = slowest_ahead['car_id']
                    self.target_lane = 2
                    return "Initiating Overtake - Moving Right"
            elif our_car.lane_id == 3:  # Rightmost lane - can only go left
                if self.sensor_processor.is_lane_clear(2, our_car, 30):
                    self.overtaking_state = 'changing_left'
                    self.overtaking_timer = 0
                    self.target_car_id = slowest_ahead['car_id']
                    self.target_lane = 2
                    return "Initiating Overtake - Moving Left"
            else:  # Middle lane - prefer left, then right
                if self.sensor_processor.is_lane_clear(1, our_car, 30):
                    self.overtaking_state = 'changing_left'
                    self.overtaking_timer = 0
                    self.target_car_id = slowest_ahead['car_id']
                    self.target_lane = 1
                    return "Initiating Overtake - Moving Left"
                elif self.sensor_processor.is_lane_clear(3, our_car, 30):
                    self.overtaking_state = 'changing_right'
                    self.overtaking_timer = 0
                    self.target_car_id = slowest_ahead['car_id']
                    self.target_lane = 3
                    return "Initiating Overtake - Moving Right"
        
        return None
    
    def _handle_overtaking(self, our_car: Car, surrounding: Dict) -> str:
        """Handle ongoing overtaking maneuver with proper state machine"""
        self.overtaking_timer += 1
        
        if self.overtaking_state == 'changing_left':
            if self.overtaking_timer >= 2:  # 2 ticks to change lanes
                our_car.lane_id = self.target_lane
                self.overtaking_state = 'accelerating'
                self.overtaking_timer = 0
                return "Lane Changed - Accelerating"
            return "Changing to Left Lane"
            
        elif self.overtaking_state == 'changing_right':
            if self.overtaking_timer >= 2:
                our_car.lane_id = self.target_lane
                self.overtaking_state = 'accelerating'
                self.overtaking_timer = 0
                return "Lane Changed - Accelerating"
            return "Changing to Right Lane"
            
        elif self.overtaking_state == 'accelerating':
            if self.overtaking_timer >= 3:  # 3 ticks to accelerate
                self.overtaking_state = 'passing'
                self.overtaking_timer = 0
                return "Passing Slower Car"
            return "Accelerating to Pass"
            
        elif self.overtaking_state == 'passing':
            # Check if we've passed the target car
            target_passed = True
            for car in surrounding['behind_same_lane']:
                if car['car_id'] == self.target_car_id and car['distance'] < 40:
                    target_passed = False
                    break
            
            # Also check if we've moved far enough ahead
            if target_passed or self.overtaking_timer >= 8:  # Timeout after 8 ticks
                # Return to original lane
                if self.sensor_processor.is_lane_clear(our_car.original_lane, our_car, 25):
                    our_car.lane_id = our_car.original_lane
                    self.overtaking_state = None
                    self.target_car_id = None
                    self.target_lane = None
                    return "Overtaking Complete"
                else:
                    self.overtaking_state = 'waiting_to_return'
                    self.overtaking_timer = 0
                    return "Waiting to Return to Lane"
            return "Passing Slower Car"
            
        elif self.overtaking_state == 'waiting_to_return':
            if self.sensor_processor.is_lane_clear(our_car.original_lane, our_car, 25):
                our_car.lane_id = our_car.original_lane
                self.overtaking_state = None
                self.target_car_id = None
                self.target_lane = None
                return "Returned to Original Lane"
            return "Waiting to Return to Lane"
        
        return "Overtaking"
    
    def _normal_driving(self, our_car: Car, surrounding: Dict) -> str:
        """Normal driving decisions"""
        # Check for cars ahead in same lane
        closest_ahead = None
        min_distance = float('inf')
        
        for car in surrounding['ahead_same_lane']:
            if car['distance'] < min_distance:
                min_distance = car['distance']
                closest_ahead = car
        
        if closest_ahead:
            distance = closest_ahead['distance']
            speed_diff = our_car.speed - closest_ahead['speed']
            
            if distance < 40 and speed_diff > 5:
                return "Following Traffic"
            elif distance < 60 and speed_diff > 10:
                return "Slow Down - Following"
            elif distance > 80:
                return "Cruising"
        
        return "Cruising"


def init_cars() -> Dict[int, Car]:
    """Initialize cars with realistic positions and speeds"""
    rng = random.Random(42)
    cars = {
        0: Car(0, 2, 0.0, 65.0, max_speed=85.0, target_speed=70.0, original_lane=2),  # Our car O
        1: Car(1, 1, rng.uniform(120.0, 180.0), rng.uniform(50.0, 60.0), max_speed=65.0),  # Car A
        2: Car(2, 2, rng.uniform(150.0, 200.0), rng.uniform(35.0, 45.0), max_speed=50.0),  # Car B - slow
        3: Car(3, 3, rng.uniform(80.0, 140.0), rng.uniform(55.0, 70.0), max_speed=75.0),  # Car C
        4: Car(4, 1, rng.uniform(250.0, 320.0), rng.uniform(45.0, 55.0), max_speed=60.0),  # Car D
    }
    return cars


def update_other_cars(cars: Dict[int, Car], rng: random.Random) -> None:
    """Update other cars with realistic behavior"""
    for cid, car in cars.items():
        if cid == 0:  # Skip our car
            continue
            
        # Add some speed variation
        speed_change = rng.uniform(-0.8, 0.8)
        car.speed = max(25.0, min(car.speed + speed_change, car.max_speed))
        
        # Occasional lane changes for other cars
        if rng.random() < 0.015:  # 1.5% chance per tick
            if car.lane_id > 1 and rng.random() < 0.5:
                car.lane_id -= 1
            elif car.lane_id < 3 and rng.random() < 0.5:
                car.lane_id += 1
        
        car.update_position()


def main() -> None:
    st.set_page_config(page_title="Professional AV Simulation", layout="wide")
    st.title("Professional Autonomous Vehicle Simulation")
    st.markdown("*Advanced B-tree sensor data processing with intelligent overtaking*")

    # Initialize session state
    if "t" not in st.session_state:
        st.session_state.t = 0
        st.session_state.rng = random.Random(42)
        st.session_state.cars = init_cars()
        st.session_state.signal = TrafficSignal()
        st.session_state.manager = BTreeManager()
        st.session_state.logger = SingleFileLogger(
            "/Users/vidyadharpothula/Desktop/dsa_project/logs_professional"
        )
        st.session_state.detailed_logger = DetailedLogger(
            "/Users/vidyadharpothula/Desktop/dsa_project/logs_professional"
        )
        st.session_state.sensor_processor = SensorDataProcessor(st.session_state.manager)
        st.session_state.driver = ProfessionalAutonomousDriver(st.session_state.sensor_processor)
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.decision = "Initializing..."
        st.session_state.last_inserted_keys = []
        st.session_state.collision_detected = False

    # UI Layout
    left, right = st.columns([2, 3])

    with left:
        st.subheader("Controls")
        c1, c2, c3 = st.columns(3)
        
        if c1.button("Start", disabled=st.session_state.running and not st.session_state.paused):
            st.session_state.running = True
            st.session_state.paused = False
            
        pause_resume_label = "Pause" if not st.session_state.paused else "Resume"
        if c2.button(pause_resume_label, disabled=not st.session_state.running):
            st.session_state.paused = not st.session_state.paused
            
        if c3.button("Reset"):
            st.session_state.t = 0
            st.session_state.rng = random.Random(42)
            st.session_state.cars = init_cars()
            st.session_state.signal = TrafficSignal()
            st.session_state.manager = BTreeManager()
            st.session_state.logger = SingleFileLogger(
                "/Users/vidyadharpothula/Desktop/dsa_project/logs_professional"
            )
            st.session_state.detailed_logger = DetailedLogger(
                "/Users/vidyadharpothula/Desktop/dsa_project/logs_professional"
            )
            st.session_state.sensor_processor = SensorDataProcessor(st.session_state.manager)
            st.session_state.driver = ProfessionalAutonomousDriver(st.session_state.sensor_processor)
            st.session_state.running = False
            st.session_state.paused = False
            st.session_state.decision = "Reset Complete"
            st.session_state.last_inserted_keys = []
            st.session_state.collision_detected = False

        st.write(f"**Time:** {st.session_state.t} seconds")
        
        # Decision display
        decision_color = "green" if "Cruising" in st.session_state.decision else "orange" if "Following" in st.session_state.decision else "red"
        st.markdown(f"### Decision: <span style='color: {decision_color}'>{st.session_state.decision}</span>", unsafe_allow_html=True)
        
        # Traffic signal status
        signal = st.session_state.signal
        signal_color = "green" if signal.state == SignalState.GREEN else "orange" if signal.state == SignalState.YELLOW else "red"
        st.markdown(f"**Traffic Signal:** <span style='color: {signal_color}'>{signal.get_state_name()}</span>", unsafe_allow_html=True)

        # B-tree data
        st.subheader("B-tree Data")
        keys = st.session_state.manager.inorder_keys()
        st.code(", ".join(map(str, keys[-15:])) if keys else "(empty)")
        
        st.subheader("Last Inserted Keys")
        st.code(", ".join(map(str, st.session_state.last_inserted_keys)) if st.session_state.last_inserted_keys else "-")

        # Car status table
        st.subheader("Car Status")
        our_car = st.session_state.cars[0]
        rows = []
        for cid, car in st.session_state.cars.items():
            car_name = "Car O (Our Car)" if cid == 0 else f"Car {chr(ord('A') + cid - 1)}"
            distance = car.position - our_car.position if cid != 0 else 0
            rows.append({
                "Car": car_name,
                "Lane": car.lane_id,
                "Position": f"{car.position:.1f}m",
                "Speed": f"{car.speed:.1f} km/h",
                "Distance": f"{distance:.1f}m" if cid != 0 else "-"
            })
        
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Detailed logging display
        st.subheader("Detailed Logging")
        if st.session_state.t > 0:
            st.success(f"Logging active - {st.session_state.t} entries")
            st.info("Check logs_professional/ directory for detailed files")

    with right:
        st.subheader("3-Lane Highway View")
        
        # Create road visualization
        df = pd.DataFrame([
            {
                "car": "O" if cid == 0 else chr(ord('A') + cid - 1),
                "lane": car.lane_id,
                "position": car.position,
                "speed": car.speed
            }
            for cid, car in st.session_state.cars.items()
        ])
        
        road = alt.Chart(df).mark_point(size=200).encode(
            x=alt.X("position:Q", title="Position (m)", scale=alt.Scale(domain=[0, 1000])),
            y=alt.Y("lane:N", title="Lane", sort=[1, 2, 3]),
            color=alt.Color("car:N", scale=alt.Scale(range=["#FF4444", "#44AA44", "#4444FF", "#FFAA44", "#AA44FF"])),
            tooltip=["car", "lane", "position", "speed"]
        ).properties(height=300)
        
        # Add traffic signal
        signal_df = pd.DataFrame([{"position": st.session_state.signal.position, "lane": 2}])
        signal_chart = alt.Chart(signal_df).mark_rect(
            width=20, height=1, color="gray"
        ).encode(
            x="position:Q",
            y="lane:N"
        )
        
        st.altair_chart(road + signal_chart, use_container_width=True)
        
        # Distance over time chart
        if st.session_state.t > 0:
            st.subheader("Distance to Cars Over Time")
            history = st.session_state.manager.get_history()
            if history:
                df_history = pd.DataFrame(history, columns=["time", "distance"])
                chart = alt.Chart(df_history).mark_line(point=True).encode(
                    x=alt.X("time:Q", title="Time (s)"),
                    y=alt.Y("distance:Q", title="Distance (m)"),
                    tooltip=["time", "distance"]
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)

    # Simulation tick
    if st.session_state.running and not st.session_state.paused:
        cars = st.session_state.cars
        our_car = cars[0]
        signal = st.session_state.signal
        others = [c for cid, c in cars.items() if cid != 0]

        # Update traffic signal
        signal.update()

        # Update other cars
        update_other_cars(cars, st.session_state.rng)

        # Generate sensor readings and insert into B-tree
        insertion_order = list(cars.keys())
        st.session_state.rng.shuffle(insertion_order)
        last_keys = []
        
        for cid in insertion_order:
            if cid == 0:  # Skip our car for sensor readings
                continue
            car = cars[cid]
            distance = car.position - our_car.position
            key = car.car_id * 10000 + st.session_state.t  # Unique composite key
            
            record = SensorRecord(
                timestamp=st.session_state.t,
                object_id=car.car_id,
                object_type="car",
                position=(car.position, 0.0, 0.0),
                velocity=car.speed,
                distance_to_our_car=distance,
                lane_id=car.lane_id
            )
            
            st.session_state.manager.insert_with_key(key, record)
            st.session_state.logger.append_from_record(record)
            last_keys.append(key)
        
        st.session_state.last_inserted_keys = last_keys

        # Get surrounding cars data for decision making
        surrounding_data = st.session_state.sensor_processor.get_surrounding_cars(our_car, others)

        # Make driving decision using B-tree data
        decision = st.session_state.driver.decide_action(our_car, signal)
        st.session_state.decision = decision

        # Log detailed data
        st.session_state.detailed_logger.log_detailed_data(
            st.session_state.t, our_car, others, 
            st.session_state.manager.inorder_keys(), decision, surrounding_data
        )
        
        # Log decision with context
        st.session_state.detailed_logger.log_decision(
            st.session_state.t, decision, {
                "our_car_lane": our_car.lane_id,
                "our_car_speed": our_car.speed,
                "surrounding_cars": len(surrounding_data['all_nearby'])
            }
        )

        # Apply decision to our car
        if decision == "Stop for Signal":
            our_car.decelerate(dt=0.05)
        elif decision == "Emergency Braking":
            our_car.emergency_brake(dt=0.05)
        elif decision == "Following Traffic":
            our_car.maintain_speed(dt=0.05)
        elif decision == "Slow Down - Following":
            our_car.decelerate(dt=0.05)
        elif "Accelerating" in decision or "Passing" in decision:
            our_car.speed_up(dt=0.05)
        elif decision == "Cruising":
            our_car.accelerate(dt=0.05)
        else:
            our_car.maintain_speed(dt=0.05)

        # Update our car position
        our_car.update_position()
        
        # Handle road wrapping
        for car in cars.values():
            if car.position > 1000:
                car.position -= 1000

        # Check for collisions
        for cid, car in cars.items():
            if cid != 0 and abs(car.position - our_car.position) < 5 and car.lane_id == our_car.lane_id:
                st.session_state.collision_detected = True
                st.session_state.running = False
                st.error("COLLISION DETECTED! Simulation stopped.")

        st.session_state.t += 1
        time.sleep(0.05)  # Faster animation
        st.rerun()

    # Display collision warning
    if st.session_state.collision_detected:
        st.error("COLLISION DETECTED! Please reset the simulation.")


if __name__ == "__main__":
    main()
