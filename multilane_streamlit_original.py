import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto

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
    position: float = 500.0  # Position of the traffic signal on the road
    state: SignalState = SignalState.GREEN
    state_timer: int = 0
    cycle_times = {
        SignalState.GREEN: 100,  # 10 seconds
        SignalState.YELLOW: 30,  # 3 seconds
        SignalState.RED: 70      # 7 seconds
    }
    
    def update(self) -> None:
        self.state_timer += 1
        if self.state_timer >= self.cycle_times[self.state]:
            if self.state == SignalState.GREEN:
                self.state = SignalState.YELLOW
            elif self.state == SignalState.YELLOW:
                self.state = SignalState.RED
            else:  # RED
                self.state = SignalState.GREEN
            self.state_timer = 0
    
    def get_state_name(self) -> str:
        return self.state.name.title()


@dataclass
class Car:
    car_id: int
    lane_id: int  # 1, 2, or 3
    position: float
    speed: float  # km/h
    max_speed: float = 40.0  # km/h
    acceleration: float = 1.0  # m/s²
    deceleration: float = 3.0  # m/s²
    length: float = 4.5  # meters
    
    def update_position(self, dt: float = 0.1) -> None:
        self.position += self.speed * (dt / 3.6)  # Convert km/h to m/s
        
    def accelerate(self, dt: float = 0.1) -> None:
        self.speed = min(self.speed + self.acceleration * dt * 3.6, self.max_speed)
        
    def decelerate(self, dt: float = 0.1) -> None:
        self.speed = max(0, self.speed - self.deceleration * dt * 3.6)
        
    def maintain_speed(self, dt: float = 0.1) -> None:
        self.speed += random.uniform(-0.5, 0.5)
        self.speed = max(0, min(self.speed, self.max_speed))


def init_cars() -> Dict[int, Car]:
    rng = random.Random(42)
    cars = {
        0: Car(0, 1, 0.0, 60.0, max_speed=80.0),  # Our car - faster and capable
        1: Car(1, 1, rng.uniform(150.0, 300.0), rng.uniform(35.0, 45.0), max_speed=60.0),  # Medium speed
        2: Car(2, 1, rng.uniform(350.0, 450.0), rng.uniform(25.0, 35.0), max_speed=50.0),  # Slower car
        3: Car(3, 2, rng.uniform(100.0, 200.0), rng.uniform(50.0, 65.0), max_speed=75.0),  # Fast car
        4: Car(4, 3, rng.uniform(100.0, 500.0), rng.uniform(40.0, 55.0), max_speed=70.0),  # Good speed
    }
    return cars


def jitter_speed(speed: float, rng: random.Random) -> float:
    # Varied speed changes suitable for higher-speed simulation
    jitter = rng.uniform(-4.0, 4.0)
    new_speed = speed + jitter
    return max(10.0, min(80.0, new_speed))


def should_stop_for_signal(car: Car, signal: TrafficSignal, cars: List[Car]) -> bool:
    if car.position > signal.position:
        return False
        
    distance_to_signal = signal.position - car.position
    stopping_distance = (car.speed ** 2) / (2 * 3.0)

    if signal.state == SignalState.RED and distance_to_signal < stopping_distance * 1.5:
        return True

    for other in cars:
        if (other.lane_id == car.lane_id and 
            other.position > car.position and 
            other.position < signal.position and
            signal.state == SignalState.RED and
            (other.position - car.position) < stopping_distance * 1.2):
            return True
    
    return False


def get_surrounding_cars(car: Car, all_cars: List[Car], signal: TrafficSignal, manager: BTreeManager) -> dict:
    surrounding = {
        'ahead': [],
        'left_lane': [],
        'right_lane': [],
        'behind': []
    }
    
    recent_keys = manager.inorder_keys()[-100:]
    
    for key in recent_keys:
        record = manager.search(key)
        if not record or record.object_id == car.car_id:
            continue
            
        other_pos = record.position[0]
        other_lane = getattr(record, 'lane_id', 1)
        distance = other_pos - car.position
        
        if abs(distance) < 200:
            if other_lane == car.lane_id:
                if distance > 0:
                    surrounding['ahead'].append({
                        'position': other_pos,
                        'distance': distance,
                        'speed': record.velocity,
                        'lane': other_lane
                    })
                else:
                    surrounding['behind'].append({
                        'position': other_pos,
                        'distance': abs(distance),
                        'speed': record.velocity,
                        'lane': other_lane
                    })
            elif other_lane == car.lane_id - 1 and car.lane_id > 1:
                surrounding['left_lane'].append({
                    'position': other_pos,
                    'distance': abs(distance),
                    'speed': record.velocity,
                    'lane': other_lane
                })
            elif other_lane == car.lane_id + 1 and car.lane_id < 3:
                surrounding['right_lane'].append({
                    'position': other_pos,
                    'distance': abs(distance),
                    'speed': record.velocity,
                    'lane': other_lane
                })
    
    return surrounding


def is_lane_clear(surrounding_cars, lane_id, position, gap_m=20.0):
    """Check if a lane is clear for a car to move into using B-tree data."""
    # Check left lane
    if lane_id < surrounding_cars['lane_id']:
        for car in surrounding_cars['left_lane']:
            if abs(car['position'] - position) < gap_m:
                return False
    # Check right lane
    elif lane_id > surrounding_cars['lane_id']:
        for car in surrounding_cars['right_lane']:
            if abs(car['position'] - position) < gap_m:
                return False
    # Check same lane
    else:
        for car in surrounding_cars['ahead'] + surrounding_cars['behind']:
            if abs(car['position'] - position) < gap_m:
                return False
    return True


def lane_clear(desired_lane: int, our: Car, others: List[Car], gap_m: float = 25.0) -> bool:
    for c in others:
        if c.lane_id == desired_lane and abs(c.position - our.position) < gap_m:
            return False
    return True


def nearest_slower_ahead_same_lane(our: Car, others: List[Car]) -> Tuple[float, Optional[Car]]:
    candidates = [c for c in others if c.lane_id == our.lane_id and c.position > our.position and c.speed < our.speed]
    if not candidates:
        return float("inf"), None
    nearest = min(candidates, key=lambda c: c.position - our.position)
    return nearest.position - our.position, nearest


def decide_for_our_car(our: Car, others: List[Car], signal: TrafficSignal) -> str:
    if should_stop_for_signal(our, signal, others):
        return "Stop for Signal"
    
    # Get real-time sensor data from B-tree
    surrounding = get_surrounding_cars(our, others, signal, st.session_state.manager)
    
    # Find closest car ahead in current lane using sensor data
    closest_ahead = None
    min_distance = float('inf')
    
    # Also check for cars in other lanes that might need to be overtaken
    cars_in_other_lanes = []
    
    for car in others:
        # Check for cars ahead in same lane
        if car.lane_id == our.lane_id and car.position > our.position:
            distance = car.position - our.position
            if distance < min_distance:
                min_distance = distance
                closest_ahead = car
        
        # Check for cars in other lanes that might need to be overtaken
        elif car.lane_id != our.lane_id and abs(car.position - our.position) < 100:
            cars_in_other_lanes.append((car, abs(car.position - our.position)))
            
        # Specifically look for car 'C' (car_id 2) to overtake
        if car.car_id == 2 and car.lane_id != our.lane_id and abs(car.position - our.position) < 100:
            cars_in_other_lanes.append((car, abs(car.position - our.position)))
    
    # Handle ongoing overtaking maneuver with proper phases
    if st.session_state.get('overtaking', False):
        overtake_phase = st.session_state.get('overtake_phase', 'changing_lanes')
        
        if overtake_phase == 'changing_lanes':
            # After changing lanes, move to acceleration phase
            st.session_state.overtake_phase = 'accelerating'
            return "Changing Lanes to Overtake"
        elif overtake_phase == 'accelerating':
            # Increase speed during overtaking
            our.speed += 5.0  # Boost speed for overtaking
            st.session_state.overtake_phase = 'passing'
            return "Accelerating to Pass"
        elif overtake_phase == 'passing':
            # Check if we've passed the slow car
            slow_car_behind = False
            target_lane = st.session_state.get('target_lane', our.lane_id)
            original_lane = st.session_state.get('original_lane', our.lane_id)
            
            # Check for cars in the target lane that we're overtaking
            for car in others:
                if (car.lane_id == target_lane and
                    car.position < our.position and 
                    (our.position - car.position) > 30):  # Passed with safe margin
                    slow_car_behind = True
                    break
            
            if slow_car_behind:
                st.session_state.overtake_phase = 'returning'
                return "Passed Car - Preparing to Return"
            else:
                return "Passing Slower Car"
        elif overtake_phase == 'returning':
            # Check if it's safe to return to original lane
            original_lane = st.session_state.get('original_lane', our.lane_id)
            if lane_clear(original_lane, our, others, gap_m=30.0):
                # Return to original lane
                our.lane_id = original_lane
                # Complete the overtaking maneuver
                st.session_state.overtaking = False
                st.session_state.overtake_phase = None
                st.session_state.target_lane = None
                st.session_state.original_lane = None
                return "Completed Overtaking Maneuver"
            return "Returning to Original Lane"
    
    # Check if we need to initiate an overtaking maneuver
    if not st.session_state.get('overtaking', False):
        # If there's a slower car ahead in our lane
        if closest_ahead and closest_ahead.speed < our.speed * 0.8 and min_distance < 50:
            # Check if left lane is clear (if we're not already in leftmost lane)
            if our.lane_id > 1 and lane_clear(our.lane_id - 1, our, others):
                st.session_state.overtaking = True
                st.session_state.overtake_phase = 'changing_lanes'
                st.session_state.target_lane = our.lane_id - 1
                st.session_state.original_lane = our.lane_id
                our.lane_id = st.session_state.target_lane  # Move to target lane
                return "Initiating Overtake - Moving Left"
            # Check if right lane is clear (if we're not already in rightmost lane)
            elif our.lane_id < 3 and lane_clear(our.lane_id + 1, our, others):
                st.session_state.overtaking = True
                st.session_state.overtake_phase = 'changing_lanes'
                st.session_state.target_lane = our.lane_id + 1
                st.session_state.original_lane = our.lane_id
                our.lane_id = st.session_state.target_lane  # Move to target lane
                return "Initiating Overtake - Moving Right"
        
        # Check for cars in other lanes that might need to be overtaken
        for car, distance in sorted(cars_in_other_lanes, key=lambda x: x[1]):
            if car.speed < our.speed * 0.8 and distance < 50:
                # If car is in lane 1 and we're in lane 2, move to lane 1
                if car.lane_id == 1 and our.lane_id == 2 and lane_clear(1, our, others):
                    st.session_state.overtaking = True
                    st.session_state.overtake_phase = 'changing_lanes'
                    st.session_state.target_lane = 1
                    st.session_state.original_lane = our.lane_id
                    our.lane_id = st.session_state.target_lane  # Move to target lane
                    return "Initiating Overtake - Moving to Lane 1"
                # If car is in lane 3 and we're in lane 2, move to lane 3
                elif car.lane_id == 3 and our.lane_id == 2 and lane_clear(3, our, others):
                    st.session_state.overtaking = True
                    st.session_state.overtake_phase = 'changing_lanes'
                    st.session_state.target_lane = 3
                    st.session_state.original_lane = our.lane_id
                    our.lane_id = st.session_state.target_lane  # Move to target lane
                    return "Initiating Overtake - Moving to Lane 3"
    
    # Sensor-based decision making
    if closest_ahead:
        distance = closest_ahead.position - our.position
        speed_diff = our.speed - closest_ahead.speed
        
        # Emergency braking if too close
        if distance < 30:
            return "Emergency Braking"
        
        # Consider overtaking if car ahead is significantly slower
        if distance < 100 and speed_diff > 15 and closest_ahead.speed < our.speed * 0.75:
            return "Prepare to Overtake"
        
        # Slow down if approaching slower car
        if distance < 80 and speed_diff > 5:
            return "Slow Down - Following"
        
        # Maintain following distance
        if distance < 60:
            return "Following Traffic"
    
    # Check for cars in other lanes that might need to be overtaken
    for car, distance in sorted(cars_in_other_lanes, key=lambda x: x[1]):
        # If car C is nearby and in another lane, consider changing lanes to overtake
        if car.car_id == 3 and distance < 50 and car.speed < our.speed * 0.8:
            # Check if we can safely change lanes
            target_lane = car.lane_id
            if lane_clear(target_lane, our, others, gap_m=40.0):
                return "Prepare to Overtake"
    
    return "Cruising"


def main() -> None:
    st.set_page_config(page_title="3-Lane AV Simulation with Traffic Signals", layout="wide")
    st.title("3-Lane Autonomous Driving with Traffic Signals")

    # Initialize session state
    if "t" not in st.session_state:
        st.session_state.t = 0
        st.session_state.rng = random.Random(7)
        st.session_state.cars = init_cars()
        st.session_state.signal = TrafficSignal()
        st.session_state.manager = BTreeManager()
        st.session_state.logger = SingleFileLogger(
            "/Users/vidyadharpothula/dsa_project/logs_multilane"
        )
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.decision = "-"
        st.session_state.overtaking = False
        st.session_state.overtake_phase = None
        st.session_state.original_lane = 1
        st.session_state.lane_change_cooldown = 0
        st.session_state.last_inserted_keys = []

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
            st.session_state.rng = random.Random(7)
            st.session_state.cars = init_cars()
            st.session_state.signal = TrafficSignal()
            st.session_state.manager = BTreeManager()
            st.session_state.logger = SingleFileLogger(
                "/Users/vidyadharpothula/dsa_project/logs_multilane"
            )
            st.session_state.running = False
            st.session_state.paused = False
            st.session_state.decision = "-"
            st.session_state.overtaking = False
            st.session_state.overtake_phase = None
            st.session_state.original_lane = 1
            st.session_state.lane_change_cooldown = 0
            st.session_state.last_inserted_keys = []

        st.write(f"t = {st.session_state.t} s")
        st.markdown(f"### Decision: {st.session_state.decision}")

        st.subheader("B-tree inorder keys (last 60)")
        keys = st.session_state.manager.inorder_keys()
        st.code(", ".join(map(str, keys[-60:])) if keys else "(empty)")
        st.subheader("Last inserted keys (this tick)")
        st.code(", ".join(map(str, st.session_state.last_inserted_keys)) if st.session_state.last_inserted_keys else "-")

        st.subheader("Latest sensor table")
        our = st.session_state.cars[0]
        rows = [
            {
                "car": chr(ord("A") + cid - 1),
                "lane": car.lane_id,
                "distance": car.position - our.position,
                "speed": car.speed,
            }
            for cid, car in st.session_state.cars.items() if cid != 0
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with right:
        st.subheader("3-lane road view")
        df = pd.DataFrame(
            [
                {
                    "car": ("O" if cid == 0 else chr(ord("A") + cid - 1)),
                    "lane": car.lane_id,
                    "position": car.position,
                }
                for cid, car in st.session_state.cars.items()
            ]
        )
        road = alt.Chart(df).mark_point(size=200).encode(
            x=alt.X("position:Q", title="Position (m)", scale=alt.Scale(domain=[0, 1000])),
            y=alt.Y("lane:N", title="Lane", sort=[1, 2, 3]),
            color="car:N",
        ).properties(height=220)
        st.altair_chart(road, use_container_width=True)

    # Simulation tick
    if st.session_state.running and not st.session_state.paused:
        cars = st.session_state.cars
        our = cars[0]
        signal = st.session_state.signal
        others = [c for cid, c in cars.items() if cid != 0]

        # Update traffic signal
        signal.update()

        # Update other cars
        for cid, car in cars.items():
            if cid == 0:
                continue
            car.speed = jitter_speed(car.speed, st.session_state.rng)
            
            if cid in [1, 3]:
                if random.random() < 0.1:
                    car.speed += random.uniform(-2, 2)
                    car.speed = max(10, min(40, car.speed))
            
            # Get surrounding cars from B-tree for overtaking decisions
            surrounding = get_surrounding_cars(car, list(cars.values()), signal, st.session_state.manager)
            
            # Consider overtaking if car is significantly slower than others
            if random.random() < 0.05:
                # Check for slower cars ahead using B-tree data
                slower_car_ahead = False
                for car_data in surrounding['ahead']:
                    # Only consider overtaking if the car ahead is significantly slower
                    if car_data['distance'] < 50 and car_data['speed'] < car.speed * 0.8:
                        slower_car_ahead = True
                        break
                
                # Only change lanes if there's actually a slower car ahead
                if slower_car_ahead:
                    # Try to change lanes if possible, prioritizing left lane
                    if car.lane_id > 1 and not any(c['distance'] < 30 for c in surrounding['left_lane']):
                        car.lane_id -= 1  # Move to left lane
                        car.speed_up()  # Speed up to overtake
                    elif car.lane_id < 3 and not any(c['distance'] < 30 for c in surrounding['right_lane']):
                        car.lane_id += 1  # Move to right lane
                        car.speed_up()  # Speed up to overtake
            
            if should_stop_for_signal(car, signal, list(cars.values())):
                car.decelerate()
            else:
                car.maintain_speed()
            
            car.update_position()
            if car.position > 1000:
                car.position -= 1000
                
            if cid != 0 and abs(car.position - our.position) < 5 and car.lane_id == our.lane_id:
                st.session_state.collision_detected = True

        # Generate sensor readings
        insertion_order = list(cars.keys())
        st.session_state.rng.shuffle(insertion_order)
        last_keys: List[int] = []
        for cid in insertion_order:
            if cid == 0:
                continue
            car = cars[cid]
            distance = car.position - our.position
            key = car.car_id * 1000 + st.session_state.t
            rec = SensorRecord(
                timestamp=st.session_state.t,
                object_id=car.car_id,
                object_type="car",
                position=(car.position, 0.0, 0.0),
                velocity=car.speed,
                distance_to_our_car=distance,
                lane_id=car.lane_id
            )
            st.session_state.manager.insert_with_key(key, rec)
            st.session_state.logger.append_from_record(rec)
            last_keys.append(key)
        st.session_state.last_inserted_keys = last_keys

        # Update our car's decision
        decision = decide_for_our_car(our, others, signal)
        st.session_state.decision = decision

        # Handle lane changes
        if st.session_state.lane_change_cooldown > 0:
            st.session_state.lane_change_cooldown -= 1

        # Overtaking state machine
        if decision == "Prepare to Overtake" and st.session_state.lane_change_cooldown == 0:
            # Phase 1: Start overtaking - change to overtaking lane
            target_lanes = []
            if our.lane_id > 1:  # Can go left
                target_lanes.append(our.lane_id - 1)
            if our.lane_id < 3:  # Can go right  
                target_lanes.append(our.lane_id + 1)
            
            # Choose the clearest lane for overtaking
            for target_lane in target_lanes:
                if lane_clear(target_lane, our, others, gap_m=50.0):
                    st.session_state.original_lane = our.lane_id
                    our.lane_id = target_lane  # Actually change lanes visually
                    st.session_state.overtaking = True
                    st.session_state.overtake_phase = 'changing_lanes'
                    st.session_state.lane_change_cooldown = 20  # Time to change lanes
                    break
        
        # Handle overtaking phases
        elif st.session_state.get('overtaking', False):
            overtake_phase = st.session_state.get('overtake_phase', 'changing_lanes')
            
            if overtake_phase == 'changing_lanes' and st.session_state.lane_change_cooldown == 0:
                # Phase 2: Now accelerate to pass
                st.session_state.overtake_phase = 'accelerating'
                st.session_state.lane_change_cooldown = 15
            
            elif overtake_phase == 'accelerating' and st.session_state.lane_change_cooldown == 0:
                # Phase 3: Now actively passing
                st.session_state.overtake_phase = 'passing'
                st.session_state.lane_change_cooldown = 25  # Time to pass the car
            
            elif overtake_phase == 'passing' and st.session_state.lane_change_cooldown == 0:
                # Check if we've passed the slow car
                original_lane = st.session_state.get('original_lane', 1)
                slow_car_passed = False
                
                # Use B-tree to get surrounding cars data
                surrounding = get_surrounding_cars(our, others, signal, st.session_state.manager)
                
                # Find the car we're trying to overtake using B-tree data
                for car_data in surrounding['behind']:
                    if (car_data['lane'] == original_lane and 
                        car_data['speed'] < our.speed * 0.8):  # Car is significantly slower
                        # Check if we've passed it with enough margin (30 units)
                        if our.position > car_data['position'] + 30:
                            slow_car_passed = True
                            break
                
                # If not found in B-tree, check directly in the cars list
                if not slow_car_passed:
                    for car in others:
                        if (car.lane_id == original_lane and 
                            car.position < our.position + 50 and  # Car is nearby
                            car.speed < our.speed * 0.8):  # Car is significantly slower
                            # Check if we've passed it with enough margin (30 units)
                            if our.position > car.position + 30:
                                slow_car_passed = True
                                break
                
                if slow_car_passed:
                    st.session_state.overtake_phase = 'returning'
                    st.session_state.lane_change_cooldown = 15
                else:
                    # Continue passing
                    st.session_state.lane_change_cooldown = 10
            
            elif overtake_phase == 'returning' and st.session_state.lane_change_cooldown == 0:
                # Phase 4: Return to original lane
                original_lane = st.session_state.get('original_lane', 1)
                if lane_clear(original_lane, our, others, gap_m=40.0):
                    our.lane_id = original_lane  # Actually return to original lane
                    st.session_state.overtaking = False
                    st.session_state.overtake_phase = None
                    st.session_state.lane_change_cooldown = 15

        # Enhanced speed control based on sensor decisions and overtaking phases
        if decision == "Stop for Signal":
            our.decelerate(dt=0.1)
        elif decision == "Emergency Braking":
            our.decelerate(dt=0.2)  # Hard braking
        elif decision == "Slow Down - Following":
            our.decelerate(dt=0.08)  # Moderate braking
        elif decision == "Following Traffic":
            our.maintain_speed(dt=0.1)  # Match traffic speed
        elif decision == "Prepare to Overtake":
            our.accelerate(dt=0.1)  # Speed up to prepare
        elif decision == "Changing Lanes to Overtake":
            our.maintain_speed(dt=0.1)  # Steady speed while changing lanes
        elif decision == "Accelerating to Pass":
            our.accelerate(dt=0.15)  # Speed up significantly
        elif decision == "Passing Slower Car":
            our.accelerate(dt=0.12)  # Maintain high speed while passing
        elif decision == "Passed Car - Preparing to Return":
            our.maintain_speed(dt=0.1)  # Maintain speed before returning
        elif decision == "Returning to Original Lane":
            our.decelerate(dt=0.05)  # Slow down slightly while changing back
        elif decision == "Cruising":
            our.accelerate(dt=0.1)  # Accelerate to target speed
        else:
            our.maintain_speed(dt=0.1)  # Default behavior
            
        our.update_position()
        if our.position > 1000:
            our.position -= 1000

        st.session_state.t += 1
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()