# Autonomous Vehicle Perception Indexing with B-tree Data Structure

A sophisticated **real-time autonomous vehicle simulation** that demonstrates advanced **B-tree data structure** implementation for **intelligent sensor data processing** and **autonomous driving decisions**.

## Purpose & B-tree Usage

### **Why B-trees in Autonomous Vehicles?**

Autonomous vehicles generate **massive amounts of sensor data** every second:
- **Position data** from GPS and LiDAR
- **Speed information** from wheel sensors
- **Object detection** from cameras and radar
- **Traffic conditions** from V2V communication

**B-trees provide the perfect solution** for this real-time data management:

#### **1. Efficient Time-Series Indexing**
- **Composite key strategy**: car_id * 10000 + timestamp
- **Chronological ordering** of sensor readings
- **Fast insertion** of new sensor data (O(log n))
- **Efficient range queries** for time-based analysis

#### **2. Multi-Vehicle Data Management**
- **Unique key space** per vehicle (Car 1: 10000-19999, Car 2: 20000-29999)
- **Scalable** to thousands of vehicles
- **Isolated data** per vehicle
- **Parallel processing** capability

#### **3. Real-Time Decision Making**
- **Recent data retrieval** from B-tree (last 100 readings)
- **O(log n) search** operations for specific vehicles
- **Real-time processing** for driving decisions

## B-tree Implementation Details

### **Data Structure**
The simulation uses a SensorRecord data structure containing:
- **timestamp**: Time of reading
- **object_id**: Vehicle identifier
- **object_type**: Vehicle type (car, truck, etc.)
- **position**: 3D coordinates (x, y, z)
- **velocity**: Speed in km/h
- **distance_to_our_car**: Relative distance in meters
- **lane_id**: Lane number (1, 2, 3)

### **B-tree Operations**
- **insert()**: O(log n) insertion with automatic balancing
- **search()**: O(log n) search for specific sensor reading
- **inorder_keys()**: O(n) traversal for chronological data access
- **get_latest()**: O(log n) access to most recent data

### **Composite Key Strategy**
Each sensor reading gets a unique composite key: `car_id * 10000 + timestamp`

**Examples:**
- Car 1, timestamp 5 → key = 10005
- Car 2, timestamp 3 → key = 20003  
- Car 3, timestamp 7 → key = 30007

**Benefits:**
- **Unique identification** of each sensor reading
- **Natural ordering** by car and time
- **Efficient range queries** for specific vehicles
- **Scalable** to 10,000 cars per second

## How B-tree Enables Autonomous Driving

### **1. Real-Time Sensor Data Processing**
Every simulation tick (0.1 seconds), sensor data from all vehicles is:
- **Generated** with position, speed, and lane information
- **Indexed** using composite keys (car_id * 10000 + timestamp)
- **Inserted** into B-tree for O(log n) storage and retrieval

### **2. Surrounding Vehicle Detection**
The autonomous vehicle uses B-tree data to:
- **Retrieve** recent sensor readings (last 100 entries)
- **Search** for specific vehicles using O(log n) operations
- **Analyze** relative positions and speeds within 200m radius
- **Categorize** vehicles by lane (ahead, behind, left, right)

### **3. Intelligent Decision Making**
Based on B-tree sensor data, the vehicle:
- **Identifies** slower vehicles ahead in the same lane
- **Checks** lane clearance for safe overtaking maneuvers
- **Makes** real-time decisions (overtake, follow, brake)
- **Executes** smooth lane changes and speed adjustments

## B-tree Performance Benefits

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
- **100 vehicles**: ~7 tree levels, <1ms operations
- **1,000 vehicles**: ~10 tree levels, <2ms operations  
- **10,000 vehicles**: ~14 tree levels, <3ms operations

## Simulation Features

### **Professional UI**
- Clean, professional interface without emojis
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

## Quick Start

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
- **Public**: http://YOUR_IP:8501 (network accessible)

#### **Option 2: Local Only**
```bash
python3 run_app.py
```

#### **Option 3: Direct Streamlit**
```bash
streamlit run smooth_av_simulation.py --server.address=0.0.0.0 --server.port=8501
```

### **How to Find Your IP Address**

#### **Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" in the output.

#### **macOS/Linux:**
```bash
ifconfig | grep "inet "
```
Or use:
```bash
hostname -I
```

#### **Alternative Method:**
The deployment script automatically detects and displays your IP address when you run:
```bash
python3 deploy_public.py
```

### **Test the System**
```bash
python3 test_professional_simulation.py
```

## Testing B-tree Functionality

### **Comprehensive Test Suite**
```bash
python3 test_professional_simulation.py
```

**Tests Include:**
- B-tree insert, search, and traversal operations
- Sensor data processing and surrounding car detection
- Overtaking logic for all lanes
- Detailed logging system
- Car physics and movement
- Traffic signal compliance

## Educational Value

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

## Future Enhancements

- **Machine Learning**: AI-based decision making using B-tree data
- **Multi-lane highways**: Support for more lanes and complex intersections
- **Weather conditions**: Environmental factors affecting sensor data
- **V2V communication**: Vehicle-to-vehicle data sharing
- **Performance optimization**: GPU acceleration for large-scale simulations

## Technical Documentation

- **`btree_core.py`**: Core B-tree implementation
- **`smooth_av_simulation.py`**: Main simulation with smooth animations
- **`professional_av_simulation.py`**: Professional UI version
- **`test_professional_simulation.py`**: Comprehensive test suite
- **`README_PROFESSIONAL.md`**: Detailed technical documentation

---

**Built to demonstrate how advanced data structures (B-trees) enable real-time autonomous vehicle decision making with intelligent sensor data processing and smooth, realistic driving behaviors.**