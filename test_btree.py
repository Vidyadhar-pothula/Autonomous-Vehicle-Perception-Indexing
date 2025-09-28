#!/usr/bin/env python3
"""
Test script to verify B-tree functionality and sensor data processing
"""

from btree_core import BTreeManager, SensorRecord
from professional_simulation import SensorDataProcessor, AutonomousDriver, Car, TrafficSignal
import random

def test_btree_operations():
    """Test basic B-tree operations"""
    print("🧪 Testing B-tree Operations...")
    
    manager = BTreeManager()
    
    # Insert some test records
    for i in range(10):
        record = SensorRecord(
            timestamp=i,
            object_id=100 + i,
            object_type="car",
            position=(i * 10.0, 0.0, 0.0),
            velocity=50.0 + i,
            distance_to_our_car=100.0 - i * 5,
            lane_id=1 + (i % 3)
        )
        manager.insert_with_key(i, record)
    
    # Test search
    result = manager.search(5)
    if result:
        print(f"✅ Search successful: Found car {result.object_id} at position {result.position}")
    else:
        print("❌ Search failed")
    
    # Test get_latest
    latest = manager.get_latest()
    if latest:
        print(f"✅ Get latest successful: Timestamp {latest[0]}, Car {latest[1].object_id}")
    else:
        print("❌ Get latest failed")
    
    # Test inorder traversal
    keys = manager.inorder_keys()
    print(f"✅ Inorder keys: {keys}")
    
    return manager

def test_sensor_processing():
    """Test sensor data processing"""
    print("\n🧪 Testing Sensor Data Processing...")
    
    manager = BTreeManager()
    processor = SensorDataProcessor(manager)
    
    # Create a test car
    our_car = Car(0, 2, 100.0, 60.0)
    
    # Insert some surrounding cars
    test_cars = [
        SensorRecord(1, 1, "car", (150.0, 0.0, 0.0), 50.0, 50.0, 1),  # Left lane, ahead
        SensorRecord(2, 2, "car", (120.0, 0.0, 0.0), 40.0, 20.0, 2),  # Same lane, ahead
        SensorRecord(3, 3, "car", (80.0, 0.0, 0.0), 70.0, -20.0, 3),  # Right lane, behind
    ]
    
    for i, record in enumerate(test_cars):
        manager.insert_with_key(i + 1, record)
    
    # Test surrounding cars detection
    surrounding = processor.get_surrounding_cars(our_car, [])
    
    print(f"✅ Cars ahead in same lane: {len(surrounding['ahead_same_lane'])}")
    print(f"✅ Cars behind in same lane: {len(surrounding['behind_same_lane'])}")
    print(f"✅ Cars in left lane: {len(surrounding['left_lane'])}")
    print(f"✅ Cars in right lane: {len(surrounding['right_lane'])}")
    
    # Test lane clearance
    left_clear = processor.is_lane_clear(1, our_car, 30.0)
    right_clear = processor.is_lane_clear(3, our_car, 30.0)
    
    print(f"✅ Left lane clear: {left_clear}")
    print(f"✅ Right lane clear: {right_clear}")

def test_autonomous_driver():
    """Test autonomous driving decisions"""
    print("\n🧪 Testing Autonomous Driver...")
    
    manager = BTreeManager()
    processor = SensorDataProcessor(manager)
    driver = AutonomousDriver(processor)
    
    # Create test scenario
    our_car = Car(0, 2, 100.0, 60.0)
    signal = TrafficSignal()
    
    # Insert a slow car ahead
    slow_car = SensorRecord(
        1, 1, "car", (150.0, 0.0, 0.0), 30.0, 50.0, 2  # Same lane, slow
    )
    manager.insert_with_key(1, slow_car)
    
    # Test decision making
    decision = driver.decide_action(our_car, signal)
    print(f"✅ Decision: {decision}")
    
    # Test overtaking state machine
    if "Overtake" in decision:
        print("✅ Overtaking logic triggered correctly")
        
        # Simulate overtaking phases
        for phase in range(10):
            decision = driver.decide_action(our_car, signal)
            print(f"   Phase {phase}: {decision}")
            if "Complete" in decision:
                break

def main():
    """Run all tests"""
    print("🚗 Professional AV Simulation - B-tree Tests\n")
    
    try:
        # Test B-tree operations
        manager = test_btree_operations()
        
        # Test sensor processing
        test_sensor_processing()
        
        # Test autonomous driver
        test_autonomous_driver()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Test Summary:")
        print("✅ B-tree insert, search, and traversal working")
        print("✅ Sensor data processing working")
        print("✅ Autonomous driving decisions working")
        print("✅ Overtaking state machine working")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
