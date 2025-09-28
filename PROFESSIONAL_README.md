# 🚗 Professional Autonomous Vehicle Simulation

A sophisticated autonomous vehicle simulation that demonstrates advanced B-tree data structure usage for real-time sensor data processing and intelligent driving decisions.

## 🌟 Key Features

### 🧠 Advanced B-tree Integration
- **Real-time sensor data indexing** using composite keys (car_id * 10000 + timestamp)
- **Efficient search and retrieval** of surrounding vehicle data
- **In-order traversal** for chronological data analysis
- **Memory-efficient storage** with proper node splitting and balancing

### 🚙 Professional Overtaking System
- **Multi-phase overtaking state machine**:
  1. **Detection**: Identifies slower vehicles ahead
  2. **Lane Change**: Safely changes to overtaking lane
  3. **Acceleration**: Increases speed to pass
  4. **Passing**: Maintains speed while overtaking
  5. **Return**: Safely returns to original lane
- **Intelligent lane selection** (prefers left lane for overtaking)
- **Safety checks** using B-tree sensor data for lane clearance

### 🛣️ Realistic Traffic Simulation
- **3-lane highway** with proper lane management
- **Traffic signals** with realistic timing cycles
- **Dynamic vehicle behavior** with speed variations and lane changes
- **Collision detection** and emergency braking
- **Realistic physics** with proper acceleration/deceleration

### 📊 Professional UI
- **Real-time visualization** of all vehicles and lanes
- **Live decision display** with color-coded status
- **B-tree data monitoring** showing inserted keys
- **Distance tracking** over time
- **Traffic signal status** with visual indicators

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas altair
```

### Run the Simulation
```bash
# Auto-launch with browser opening
python3 run_app.py

# Or run directly
streamlit run professional_simulation.py
```

### Test the System
```bash
# Run comprehensive tests
python3 test_btree.py
```

## 🏗️ Architecture

### Core Components

#### 1. **BTreeManager** (`btree_core.py`)
- Manages B-tree operations for sensor data
- Handles insertions, searches, and traversals
- Maintains chronological order of sensor readings

#### 2. **SensorDataProcessor** (`professional_simulation.py`)
- Processes B-tree data for decision making
- Identifies surrounding vehicles by lane and distance
- Provides lane clearance checks for safe maneuvers

#### 3. **AutonomousDriver** (`professional_simulation.py`)
- Implements intelligent driving decisions
- Manages overtaking state machine
- Handles emergency situations and traffic signals

#### 4. **Car Class** (`professional_simulation.py`)
- Realistic vehicle physics and behavior
- Speed control with acceleration/deceleration
- Lane management and position updates

### Data Flow

```
Sensor Readings → B-tree Insert → Data Processing → Decision Making → Vehicle Control
     ↓                ↓                ↓                ↓                ↓
  JSONL Logs    Composite Keys    Surrounding Cars   State Machine   Position Update
```

## 🎯 B-tree Implementation Details

### Composite Key Strategy
```python
key = car_id * 10000 + timestamp
```
- **Car ID**: Ensures unique identification
- **Timestamp**: Maintains chronological order
- **Scalability**: Supports up to 10,000 cars per second

### Sensor Data Structure
```python
@dataclass
class SensorRecord:
    timestamp: int
    object_id: int
    object_type: str
    position: Tuple[float, float, float]
    velocity: float
    distance_to_our_car: float
    lane_id: int
```

### B-tree Operations
- **Insert**: O(log n) complexity for sensor data insertion
- **Search**: O(log n) complexity for vehicle lookup
- **Traversal**: O(n) complexity for chronological data access
- **Latest**: O(log n) complexity for most recent data

## 🧪 Testing

The system includes comprehensive tests covering:

### B-tree Operations
- ✅ Insert operations with composite keys
- ✅ Search functionality
- ✅ In-order traversal
- ✅ Latest record retrieval

### Sensor Processing
- ✅ Surrounding vehicle detection
- ✅ Lane clearance verification
- ✅ Distance calculations

### Autonomous Driving
- ✅ Decision making logic
- ✅ Overtaking state machine
- ✅ Emergency handling
- ✅ Traffic signal compliance

## 📈 Performance Features

### Real-time Processing
- **1Hz sensor updates** with immediate B-tree insertion
- **Sub-second decision making** using cached sensor data
- **Efficient memory usage** with B-tree node management

### Scalability
- **Multi-vehicle support** with unique identification
- **Extensible lane system** (currently 3 lanes)
- **Configurable parameters** for different scenarios

## 🎮 Controls

### Simulation Controls
- **▶️ Start**: Begin the simulation
- **⏸️ Pause/Resume**: Control simulation flow
- **🔄 Reset**: Restart with fresh state

### Real-time Monitoring
- **Decision Display**: Shows current driving action
- **B-tree Keys**: Displays recent sensor data keys
- **Car Status**: Real-time vehicle positions and speeds
- **Traffic Signal**: Current signal state

## 🔧 Configuration

### Vehicle Parameters
```python
max_speed: float = 80.0      # Maximum speed (km/h)
acceleration: float = 2.0    # Acceleration (m/s²)
deceleration: float = 4.0    # Deceleration (m/s²)
```

### Overtaking Parameters
```python
overtaking_distance: float = 100.0    # Detection distance (m)
lane_change_gap: float = 30.0         # Required gap for lane change (m)
overtaking_speed_boost: float = 1.5   # Speed multiplier during overtaking
```

### Traffic Signal Timing
```python
GREEN: 100 ticks    # ~10 seconds
YELLOW: 30 ticks    # ~3 seconds  
RED: 70 ticks       # ~7 seconds
```

## 🚨 Safety Features

### Collision Avoidance
- **Emergency braking** when vehicles are too close
- **Safe following distance** maintenance
- **Lane change safety checks** using B-tree data

### Traffic Compliance
- **Traffic signal adherence** with proper stopping distances
- **Speed limit compliance** with realistic acceleration
- **Lane discipline** with proper lane change procedures

## 📊 Data Logging

### JSONL Format
```json
{"timestamp": 123, "object_id": 1, "object_type": "car", "position": [150.0, 0.0, 0.0], "velocity": 50.0, "distance_to_our_car": 50.0, "lane_id": 2}
```

### Log Locations
- **Professional logs**: `logs_professional/readings.jsonl`
- **Timestamped directories** for different runs
- **Append-only format** for data integrity

## 🎓 Educational Value

This simulation demonstrates:

1. **Data Structures**: B-tree implementation and usage
2. **Algorithms**: Search, insertion, and traversal operations
3. **Real-time Systems**: Sensor data processing and decision making
4. **State Machines**: Overtaking maneuver management
5. **Software Architecture**: Modular design with clear separation of concerns

## 🔮 Future Enhancements

- **Machine Learning**: AI-based decision making
- **Multi-lane highways**: Support for more lanes
- **Weather conditions**: Rain, fog, and visibility effects
- **Vehicle types**: Trucks, motorcycles, and emergency vehicles
- **Network simulation**: V2V and V2I communication

---

**Built with ❤️ for demonstrating advanced data structures in autonomous vehicle applications**
