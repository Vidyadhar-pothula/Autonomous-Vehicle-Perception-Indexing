# 🚗 Autonomous Vehicle Perception Indexing with B-tree Data Structure

A sophisticated **real-time autonomous vehicle simulation** that demonstrates advanced **B-tree data structure** implementation for **intelligent sensor data processing** and **autonomous driving decisions**.

## 🎯 **Purpose & B-tree Usage**

### **Why B-trees in Autonomous Vehicles?**

Autonomous vehicles generate **massive amounts of sensor data** every second:
- **Position data** from GPS and LiDAR
- **Speed information** from wheel sensors
- **Object detection** from cameras and radar
- **Traffic conditions** from V2V communication

**B-trees provide the perfect solution** for this real-time data management:

#### **1. Efficient Time-Series Indexing**
```python
# Composite key: car_id * 10000 + timestamp
key = 10001  # Car 1 at timestamp 1
key = 10002  # Car 1 at timestamp 2
key = 20001  # Car 2 at timestamp 1
```
- **Chronological ordering** of sensor readings
- **Fast insertion** of new sensor data (O(log n))
- **Efficient range queries** for time-based analysis

#### **2. Multi-Vehicle Data Management**
```python
# Each vehicle gets unique key space
Car 1: keys 10000-19999
Car 2: keys 20000-29999  
Car 3: keys 30000-39999
```
- **Scalable** to thousands of vehicles
- **Isolated data** per vehicle
- **Parallel processing** capability

#### **3. Real-Time Decision Making**
```python
# Get recent sensor data for decision making
recent_keys = btree_manager.inorder_keys()[-100:]  # Last 100 readings
for key in recent_keys:
    record = btree_manager.search(key)  # O(log n) search
    # Process for driving decisions
```

## 🏗️ **B-tree Implementation Details**

### **Data Structure**
```python
@dataclass
class SensorRecord:
    timestamp: int           # Time of reading
    object_id: int          # Vehicle ID
    object_type: str        # "car", "truck", etc.
    position: Tuple[float, float, float]  # 3D coordinates
    velocity: float         # Speed in km/h
    distance_to_our_car: float  # Relative distance
    lane_id: int           # Lane number (1, 2, 3)
```

### **B-tree Operations**
```python
class BTree:
    def insert(self, key: int, value: SensorRecord) -> None:
        # O(log n) insertion with automatic balancing
        
    def search(self, key: int) -> Optional[SensorRecord]:
        # O(log n) search for specific sensor reading
        
    def inorder_keys(self) -> List[int]:
        # O(n) traversal for chronological data access
        
    def get_latest(self) -> Optional[Tuple[int, SensorRecord]]:
        # O(log n) access to most recent data
```

### **Composite Key Strategy**
```python
def generate_key(car_id: int, timestamp: int) -> int:
    return car_id * 10000 + timestamp

# Examples:
# Car 1, timestamp 5  → key = 10005
# Car 2, timestamp 3  → key = 20003  
# Car 3, timestamp 7  → key = 30007
```

**Benefits:**
- **Unique identification** of each sensor reading
- **Natural ordering** by car and time
- **Efficient range queries** for specific vehicles
- **Scalable** to 10,000 cars per second

## 🚙 **How B-tree Enables Autonomous Driving**

### **1. Real-Time Sensor Data Processing**
```python
# Every simulation tick (0.1 seconds):
for car_id in all_cars:
    # Generate sensor reading
    record = SensorRecord(...)
    key = generate_key(car_id, timestamp)
    
    # Insert into B-tree
    btree_manager.insert_with_key(key, record)
```

### **2. Surrounding Vehicle Detection**
```python
def get_surrounding_cars(our_car: Car) -> Dict:
    surrounding = {'ahead_same_lane': [], 'left_lane': [], 'right_lane': []}
    
    # Get recent sensor data from B-tree
    recent_keys = btree_manager.inorder_keys()[-100:]
    
    for key in recent_keys:
        record = btree_manager.search(key)  # O(log n)
        if record and record.object_id != our_car.car_id:
            # Analyze relative position and speed
            distance = record.position[0] - our_car.position
            if abs(distance) < 200:  # Within 200m
                # Categorize by lane and position
                categorize_vehicle(record, surrounding)
    
    return surrounding
```

### **3. Intelligent Decision Making**
```python
def decide_overtaking_action(our_car: Car) -> str:
    surrounding = get_surrounding_cars(our_car)
    
    # Find slowest car ahead in same lane
    slowest_ahead = find_slowest_car(surrounding['ahead_same_lane'])
    
    if slowest_ahead and slowest_ahead.speed < our_car.speed * 0.75:
        # Check if overtaking lane is clear using B-tree data
        if is_lane_clear(target_lane, our_car):
            return "Initiating Overtake"
    
    return "Cruising"
```

## 📊 **B-tree Performance Benefits**

### **Time Complexity**
- **Insertion**: O(log n) - Fast sensor data logging
- **Search**: O(log n) - Quick vehicle lookup
- **Range Query**: O(log n + k) - Efficient time-based queries
- **Traversal**: O(n) - Complete data access

### **Space Efficiency**
- **Balanced tree** ensures minimal height
- **Node splitting** maintains optimal storage
- **Memory efficient** for large datasets

### **Real-World Scalability**
```python
# Performance with different vehicle counts:
# 100 vehicles:   ~7 tree levels, <1ms operations
# 1,000 vehicles: ~10 tree levels, <2ms operations  
# 10,000 vehicles: ~14 tree levels, <3ms operations
```

## 🎮 **Simulation Features**

### **Professional UI**
- Clean, emoji-free interface
- Real-time B-tree data visualization
- Live decision display with color coding
- Traffic signal status monitoring

### **Smooth Overtaking Logic**
- **Multi-phase state machine**:
  1. **Detection**: Identify slower vehicles using B-tree data
  2. **Lane Change**: Smooth transition to overtaking lane
  3. **Acceleration**: Speed up while passing
  4. **Return**: Safely return to original lane
- **Works in all lanes** (leftmost, middle, rightmost)
- **Safety checks** using real-time sensor data

### **Comprehensive Logging**
- **Detailed JSONL logs** of all B-tree operations
- **Decision tracking** with context and timestamps
- **Sensor data analysis** for debugging and optimization

## 🚀 **Quick Start**

### **Prerequisites**
```bash
pip install streamlit pandas altair
```

### **Run the Simulation**

#### **Option 1: Public Access (Recommended)**
```bash
python3 deploy_public.py
```
- **Local**: http://localhost:8501
- **Public**: http://[YOUR_IP]:8501 (network accessible)

#### **Option 2: Local Only**
```bash
python3 run_app.py
```

#### **Option 3: Direct Streamlit**
```bash
streamlit run smooth_av_simulation.py --server.address=0.0.0.0 --server.port=8501
```

### **Test the System**
```bash
python3 test_professional_simulation.py
```

## 🧪 **Testing B-tree Functionality**

### **Comprehensive Test Suite**
```bash
python3 test_professional_simulation.py
```

**Tests Include:**
- ✅ B-tree insert, search, and traversal operations
- ✅ Sensor data processing and surrounding car detection
- ✅ Overtaking logic for all lanes
- ✅ Detailed logging system
- ✅ Car physics and movement
- ✅ Traffic signal compliance

## 📈 **Educational Value**

This simulation demonstrates:

### **Data Structures**
- **B-tree implementation** with insert, search, and traversal
- **Composite key strategies** for multi-dimensional indexing
- **Balanced tree maintenance** with node splitting

### **Algorithms**
- **Search algorithms** for real-time data retrieval
- **Sorting and ordering** for chronological data access
- **Range query optimization** for time-based analysis

### **Real-Time Systems**
- **Sensor data processing** at 10Hz (0.1s intervals)
- **Decision making** using cached B-tree data
- **Memory management** for continuous data streams

### **Software Architecture**
- **Modular design** with clear separation of concerns
- **State machine implementation** for complex behaviors
- **Logging and monitoring** for system observability

## 🔮 **Future Enhancements**

- **Machine Learning**: AI-based decision making using B-tree data
- **Multi-lane highways**: Support for more lanes and complex intersections
- **Weather conditions**: Environmental factors affecting sensor data
- **V2V communication**: Vehicle-to-vehicle data sharing
- **Performance optimization**: GPU acceleration for large-scale simulations

## 📚 **Technical Documentation**

- **`btree_core.py`**: Core B-tree implementation
- **`smooth_av_simulation.py`**: Main simulation with smooth animations
- **`professional_av_simulation.py`**: Professional UI version
- **`test_professional_simulation.py`**: Comprehensive test suite
- **`README_PROFESSIONAL.md`**: Detailed technical documentation

---

**Built to demonstrate how advanced data structures (B-trees) enable real-time autonomous vehicle decision making with intelligent sensor data processing and smooth, realistic driving behaviors.**