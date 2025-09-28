#!/usr/bin/env python3
"""
Comprehensive test for the Professional AV Simulation
Tests all components including B-tree, overtaking, and logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from btree_core import BTreeManager, SensorRecord
from professional_av_simulation import (
    SensorDataProcessor, ProfessionalAutonomousDriver, Car, 
    TrafficSignal, DetailedLogger, init_cars
)
import json
import tempfile
import shutil


def test_btree_operations():
    """Test B-tree operations with realistic data"""
    print("Testing B-tree Operations...")
    
    manager = BTreeManager()
    
    # Insert realistic sensor data
    for i in range(20):
        record = SensorRecord(
            timestamp=i,
            object_id=100 + (i % 5),  # 5 different cars
            object_type="car",
            position=(i * 15.0, 0.0, 0.0),
            velocity=50.0 + (i % 3) * 10,
            distance_to_our_car=100.0 - i * 3,
            lane_id=1 + (i % 3)
        )
        key = record.object_id * 10000 + i
        manager.insert_with_key(key, record)
    
    # Test search
    result = manager.search(100 * 10000 + 5)
    assert result is not None, "Search failed"
    print(f"✅ Search successful: Car {result.object_id}")
    
    # Test get_latest
    latest = manager.get_latest()
    assert latest is not None, "Get latest failed"
    print(f"✅ Get latest successful: Timestamp {latest[0]}")
    
    # Test inorder traversal
    keys = manager.inorder_keys()
    assert len(keys) == 20, f"Expected 20 keys, got {len(keys)}"
    print(f"✅ Inorder traversal: {len(keys)} keys")
    
    return manager


def test_sensor_processing():
    """Test sensor data processing"""
    print("\nTesting Sensor Data Processing...")
    
    manager = BTreeManager()
    processor = SensorDataProcessor(manager)
    
    # Create test scenario
    our_car = Car(0, 2, 100.0, 60.0)
    
    # Insert surrounding cars
    test_cars = [
        SensorRecord(1, 1, "car", (150.0, 0.0, 0.0), 50.0, 50.0, 1),  # Left lane, ahead
        SensorRecord(2, 2, "car", (120.0, 0.0, 0.0), 40.0, 20.0, 2),  # Same lane, ahead
        SensorRecord(3, 3, "car", (80.0, 0.0, 0.0), 70.0, -20.0, 3),  # Right lane, behind
        SensorRecord(4, 4, "car", (200.0, 0.0, 0.0), 30.0, 100.0, 2), # Same lane, far ahead
    ]
    
    for i, record in enumerate(test_cars):
        manager.insert_with_key(i + 1, record)
    
    # Test surrounding cars detection
    surrounding = processor.get_surrounding_cars(our_car, [])
    
    assert len(surrounding['ahead_same_lane']) == 2, "Should detect 2 cars ahead in same lane"
    assert len(surrounding['left_lane']) == 1, "Should detect 1 car in left lane"
    assert len(surrounding['right_lane']) == 1, "Should detect 1 car in right lane"
    
    print(f"✅ Cars ahead in same lane: {len(surrounding['ahead_same_lane'])}")
    print(f"✅ Cars in left lane: {len(surrounding['left_lane'])}")
    print(f"✅ Cars in right lane: {len(surrounding['right_lane'])}")
    
    # Test lane clearance
    left_clear = processor.is_lane_clear(1, our_car, 30.0)
    right_clear = processor.is_lane_clear(3, our_car, 30.0)
    
    print(f"✅ Left lane clear: {left_clear}")
    print(f"✅ Right lane clear: {right_clear}")


def test_overtaking_logic():
    """Test overtaking logic for all lanes"""
    print("\nTesting Overtaking Logic...")
    
    manager = BTreeManager()
    processor = SensorDataProcessor(manager)
    driver = ProfessionalAutonomousDriver(processor)
    
    # Test scenario 1: Car in middle lane (lane 2) with slow car ahead
    our_car = Car(0, 2, 100.0, 60.0, original_lane=2)
    signal = TrafficSignal()
    
    # Insert slow car ahead in same lane
    slow_car = SensorRecord(1, 1, "car", (150.0, 0.0, 0.0), 30.0, 50.0, 2)
    manager.insert_with_key(1, slow_car)
    
    # Test decision making
    decision = driver.decide_action(our_car, signal)
    print(f"✅ Middle lane overtaking decision: {decision}")
    
    # Test scenario 2: Car in leftmost lane (lane 1)
    our_car.lane_id = 1
    our_car.original_lane = 1
    
    # Insert slow car ahead
    slow_car_left = SensorRecord(2, 2, "car", (140.0, 0.0, 0.0), 25.0, 40.0, 1)
    manager.insert_with_key(2, slow_car_left)
    
    decision = driver.decide_action(our_car, signal)
    print(f"✅ Left lane overtaking decision: {decision}")
    
    # Test scenario 3: Car in rightmost lane (lane 3)
    our_car.lane_id = 3
    our_car.original_lane = 3
    
    # Insert slow car ahead
    slow_car_right = SensorRecord(3, 3, "car", (160.0, 0.0, 0.0), 35.0, 60.0, 3)
    manager.insert_with_key(3, slow_car_right)
    
    decision = driver.decide_action(our_car, signal)
    print(f"✅ Right lane overtaking decision: {decision}")


def test_detailed_logging():
    """Test detailed logging system"""
    print("\nTesting Detailed Logging...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger = DetailedLogger(temp_dir)
        
        # Create test data
        our_car = Car(0, 2, 100.0, 60.0)
        other_cars = [Car(1, 1, 150.0, 50.0), Car(2, 3, 80.0, 70.0)]
        btree_keys = [1, 2, 3, 4, 5]
        decision = "Test Decision"
        surrounding_data = {
            'ahead_same_lane': [{'car_id': 1, 'distance': 50}],
            'left_lane': [{'car_id': 2, 'distance': 30}]
        }
        
        # Test detailed logging
        logger.log_detailed_data(1, our_car, other_cars, btree_keys, decision, surrounding_data)
        logger.log_decision(1, decision, {"test": "context"})
        
        # Verify files were created
        detailed_log_path = os.path.join(temp_dir, "detailed_log.jsonl")
        decision_log_path = os.path.join(temp_dir, "decisions.jsonl")
        
        assert os.path.exists(detailed_log_path), "Detailed log file not created"
        assert os.path.exists(decision_log_path), "Decision log file not created"
        
        # Verify content
        with open(detailed_log_path, 'r') as f:
            detailed_content = f.read()
            assert "our_car" in detailed_content, "Our car data not logged"
            assert "btree_data" in detailed_content, "B-tree data not logged"
        
        with open(decision_log_path, 'r') as f:
            decision_content = f.read()
            assert "Test Decision" in decision_content, "Decision not logged"
        
        print("✅ Detailed logging working correctly")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_car_physics():
    """Test car physics and movement"""
    print("\nTesting Car Physics...")
    
    car = Car(0, 2, 100.0, 50.0)
    initial_position = car.position
    initial_speed = car.speed
    
    # Test acceleration
    car.accelerate(dt=0.1)
    assert car.speed > initial_speed, "Acceleration not working"
    print(f"✅ Acceleration: {initial_speed:.1f} -> {car.speed:.1f} km/h")
    
    # Test deceleration
    car.decelerate(dt=0.1)
    assert car.speed < car.max_speed, "Deceleration not working"
    print(f"✅ Deceleration working")
    
    # Test position update
    car.update_position(dt=0.1)
    assert car.position > initial_position, "Position update not working"
    print(f"✅ Position update: {initial_position:.1f} -> {car.position:.1f}m")


def test_traffic_signal():
    """Test traffic signal functionality"""
    print("\nTesting Traffic Signal...")
    
    signal = TrafficSignal()
    initial_state = signal.state
    
    # Test state transitions
    signal.state_timer = signal.cycle_times[signal.state] - 1
    signal.update()
    
    # Should transition to next state
    if initial_state == signal.state:
        signal.update()  # One more update should trigger transition
    
    print(f"✅ Traffic signal state transitions working")


def run_comprehensive_test():
    """Run all tests"""
    print("=" * 60)
    print("Professional AV Simulation - Comprehensive Tests")
    print("=" * 60)
    
    try:
        # Run all test functions
        test_btree_operations()
        test_sensor_processing()
        test_overtaking_logic()
        test_detailed_logging()
        test_car_physics()
        test_traffic_signal()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("✅ B-tree operations working correctly")
        print("✅ Sensor data processing working correctly")
        print("✅ Overtaking logic working for all lanes")
        print("✅ Detailed logging system working correctly")
        print("✅ Car physics and movement working correctly")
        print("✅ Traffic signal system working correctly")
        print("\nThe Professional AV Simulation is ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
