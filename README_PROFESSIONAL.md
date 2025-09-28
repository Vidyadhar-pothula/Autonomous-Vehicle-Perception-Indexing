# Professional Autonomous Vehicle Simulation

A sophisticated autonomous vehicle simulation demonstrating advanced B-tree data structure usage for real-time sensor data processing and intelligent driving decisions.

## Key Features

### Advanced B-tree Integration
- **Real-time sensor data indexing** using composite keys (car_id * 10000 + timestamp)
- **Efficient search and retrieval** of surrounding vehicle data
- **In-order traversal** for chronological data analysis
- **Memory-efficient storage** with proper node splitting and balancing

### Professional Overtaking System
- **Multi-phase overtaking state machine**:
  1. **Detection**: Identifies slower vehicles ahead
  2. **Lane Change**: Safely changes to overtaking lane
  3. **Acceleration**: Increases speed to pass
  4. **Passing**: Maintains speed while overtaking
  5. **Return**: Safely returns to original lane
- **Intelligent lane selection** (prefers left lane for overtaking)
- **Safety checks** using B-tree sensor data for lane clearance
- **Works properly in all lanes** (leftmost, middle, rightmost)

### Realistic Traffic Simulation
- **3-lane highway** with proper lane management
- **Traffic signals** with realistic timing cycles
- **Dynamic vehicle behavior** with speed variations and lane changes
- **Collision detection** and emergency braking
- **Realistic physics** with proper acceleration/deceleration

### Professional UI
- **Clean, professional interface** without emojis
- **Real-time visualization** of all vehicles and lanes
- **Live decision display** with color-coded status
- **B-tree data monitoring** showing inserted keys
- **Distance tracking** over time
- **Traffic signal status** with visual indicators

### Enhanced Logging System
- **Detailed JSONL logging** for every simulation tick
- **B-tree data tracking** with surrounding car analysis
- **Decision logging** with context and timestamps
- **Visible logging status** in the UI
- **Separate log files** for different data types

## Quick Start

### Prerequisites
```bash
pip install streamlit pandas altair
```

### Run the Simulation

#### Option 1: Local Access
```bash
python3 run_app.py
```

#### Option 2: Public Access (Network Accessible)
```bash
python3 deploy_public.py
```

#### Option 3: Direct Streamlit
```bash
streamlit run professional_av_simulation.py --server.address=0.0.0.0 --server.port=8501
```

### Test the System
```bash
python3 test_professional_simulation.py
```

## Architecture

### Core Components

#### 1. **BTreeManager** (`btree_core.py`)
- Manages B-tree operations for sensor data
- Handles insertions, searches, and traversals
- Maintains chronological order of sensor readings

#### 2. **SensorDataProcessor** (`professional_av_simulation.py`)
- Processes B-tree data for decision making
- Identifies surrounding vehicles by lane and distance
- Provides lane clearance checks for safe maneuvers

#### 3. **ProfessionalAutonomousDriver** (`professional_av_simulation.py`)
- Implements intelligent driving decisions
- Manages overtaking state machine
- Handles emergency situations and traffic signals

#### 4. **DetailedLogger** (`professional_av_simulation.py`)
- Enhanced logging system for detailed data tracking
- Logs B-tree data, decisions, and surrounding car information
- Creates separate JSONL files for different data types

#### 5. **Car Class** (`professional_av_simulation.py`)
- Realistic vehicle physics and behavior
- Speed control with acceleration/deceleration
- Lane management and position updates

### Data Flow

```
Sensor Readings → B-tree Insert → Data Processing → Decision Making → Vehicle Control
     ↓                ↓                ↓                ↓                ↓
  JSONL Logs    Composite Keys    Surrounding Cars   State Machine   Position Update
```

## B-tree Implementation Details

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

## Overtaking Logic

### Lane-Specific Overtaking Rules

#### Leftmost Lane (Lane 1)
- Can only overtake by moving right to Lane 2
- Checks for clearance in right lane before initiating

#### Middle Lane (Lane 2)
- Prefers left lane (Lane 1) for overtaking
- Falls back to right lane (Lane 3) if left is not clear

#### Rightmost Lane (Lane 3)
- Can only overtake by moving left to Lane 2
- Checks for clearance in left lane before initiating

### Overtaking State Machine
1. **Detection**: Identifies slower vehicles (20% speed difference)
2. **Lane Change**: 2 ticks to change lanes safely
3. **Acceleration**: 3 ticks to increase speed
4. **Passing**: Maintains speed until target is passed
5. **Return**: Returns to original lane when safe

## Logging System

### Detailed Log Format
```json
{
  "timestamp": 123,
  "our_car": {
    "id": 0,
    "lane": 2,
    "position": 150.0,
    "speed": 65.0,
    "original_lane": 2
  },
  "other_cars": [...],
  "btree_data": {
    "total_keys": 50,
    "recent_keys": [1, 2, 3, 4, 5],
    "surrounding_cars_count": {...}
  },
  "decision": "Initiating Overtake - Moving Left",
  "surrounding_details": {...}
}
```

### Log Files
- **detailed_log.jsonl**: Complete simulation data
- **decisions.jsonl**: Driving decisions with context
- **readings.jsonl**: Basic sensor readings

## Performance Features

### Real-time Processing
- **20Hz simulation updates** (0.05s intervals)
- **Sub-second decision making** using cached sensor data
- **Efficient memory usage** with B-tree node management

### Scalability
- **Multi-vehicle support** with unique identification
- **Extensible lane system** (currently 3 lanes)
- **Configurable parameters** for different scenarios

## Controls

### Simulation Controls
- **Start**: Begin the simulation
- **Pause/Resume**: Control simulation flow
- **Reset**: Restart with fresh state

### Real-time Monitoring
- **Decision Display**: Shows current driving action
- **B-tree Keys**: Displays recent sensor data keys
- **Car Status**: Real-time vehicle positions and speeds
- **Traffic Signal**: Current signal state
- **Logging Status**: Shows active logging status

## Configuration

### Vehicle Parameters
```python
max_speed: float = 80.0      # Maximum speed (km/h)
acceleration: float = 3.0    # Acceleration (m/s²)
deceleration: float = 5.0    # Deceleration (m/s²)
```

### Overtaking Parameters
```python
overtaking_distance: float = 80.0    # Detection distance (m)
lane_change_gap: float = 25.0        # Required gap for lane change (m)
overtaking_speed_boost: float = 2.0  # Speed multiplier during overtaking
```

### Traffic Signal Timing
```python
GREEN: 80 ticks    # ~4 seconds
YELLOW: 20 ticks   # ~1 second  
RED: 60 ticks      # ~3 seconds
```

## Safety Features

### Collision Avoidance
- **Emergency braking** when vehicles are too close (< 15m)
- **Safe following distance** maintenance
- **Lane change safety checks** using B-tree data

### Traffic Compliance
- **Traffic signal adherence** with proper stopping distances
- **Speed limit compliance** with realistic acceleration
- **Lane discipline** with proper lane change procedures

## Public Access

### Network Accessibility
The simulation can be made publicly accessible using:
```bash
python3 deploy_public.py
```

This will:
- Bind to all network interfaces (0.0.0.0)
- Make the simulation accessible from other devices on the network
- Display the public IP address for access
- Disable CORS and XSRF protection for public access

### Access URLs
- **Local**: http://localhost:8501
- **Public**: http://[YOUR_IP]:8501

## Testing

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
- ✅ Overtaking state machine for all lanes
- ✅ Emergency handling
- ✅ Traffic signal compliance

### Logging System
- ✅ Detailed data logging
- ✅ Decision logging with context
- ✅ File creation and content verification

## Educational Value

This simulation demonstrates:

1. **Data Structures**: B-tree implementation and usage
2. **Algorithms**: Search, insertion, and traversal operations
3. **Real-time Systems**: Sensor data processing and decision making
4. **State Machines**: Overtaking maneuver management
5. **Software Architecture**: Modular design with clear separation of concerns
6. **Logging Systems**: Comprehensive data tracking and analysis

## Future Enhancements

- **Machine Learning**: AI-based decision making
- **Multi-lane highways**: Support for more lanes
- **Weather conditions**: Rain, fog, and visibility effects
- **Vehicle types**: Trucks, motorcycles, and emergency vehicles
- **Network simulation**: V2V and V2I communication
- **Performance optimization**: GPU acceleration for large-scale simulations

---

**Built for demonstrating advanced data structures in autonomous vehicle applications**
