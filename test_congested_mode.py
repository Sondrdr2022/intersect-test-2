#!/usr/bin/env python3
"""
Test script for congested mode functionality in SmartTrafficController.
This tests the congestion detection and mode switching without requiring SUMO.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock traci module since we're testing without SUMO
class MockTraci:
    class simulation:
        @staticmethod
        def getTime():
            return 100.0
    
    class lane:
        @staticmethod
        def getIDList():
            return []
    
    class trafficlight:
        @staticmethod
        def getPhase(tl_id):
            return 0

sys.modules['traci'] = MockTraci()

# Now we can import the class
from Lane2 import SmartTrafficController

def test_congestion_detection():
    """Test congestion detection with different lane data scenarios"""
    print("Testing congestion detection...")
    
    # Create controller instance with minimal initialization
    controller = SmartTrafficController(mode="test")
    
    # Test 1: No congestion - low queue and waiting times
    lane_data_normal = {
        'lane1': {'queue_length': 2, 'waiting_time': 10},
        'lane2': {'queue_length': 1, 'waiting_time': 15},
        'lane3': {'queue_length': 3, 'waiting_time': 20}
    }
    
    result = controller.detect_congestion(lane_data_normal)
    assert not result, "Expected no congestion for normal traffic"
    print("✓ Normal traffic correctly detected as not congested")
    
    # Test 2: Congestion due to high queue lengths
    lane_data_high_queue = {
        'lane1': {'queue_length': 10, 'waiting_time': 30},
        'lane2': {'queue_length': 12, 'waiting_time': 25},
        'lane3': {'queue_length': 8, 'waiting_time': 35}
    }
    
    result = controller.detect_congestion(lane_data_high_queue)
    assert result, "Expected congestion for high queue lengths"
    print("✓ High queue lengths correctly detected as congested")
    
    # Test 3: Congestion due to high waiting times
    lane_data_high_wait = {
        'lane1': {'queue_length': 3, 'waiting_time': 80},
        'lane2': {'queue_length': 2, 'waiting_time': 90},
        'lane3': {'queue_length': 4, 'waiting_time': 70}
    }
    
    result = controller.detect_congestion(lane_data_high_wait)
    assert result, "Expected congestion for high waiting times"
    print("✓ High waiting times correctly detected as congested")
    
    # Test 4: Empty lane data
    result = controller.detect_congestion({})
    assert not result, "Expected no congestion for empty data"
    print("✓ Empty data correctly handled")

def test_mode_switching():
    """Test congested mode switching functionality"""
    print("\nTesting mode switching...")
    
    controller = SmartTrafficController(mode="test")
    
    # Store original parameters
    original_min_green = controller.adaptive_params['min_green']
    original_max_green = controller.adaptive_params['max_green']
    original_flow_weight = controller.adaptive_params['flow_weight']
    
    # Test switching to congested mode
    assert not controller.is_congested_mode, "Should start in normal mode"
    controller.set_congested_mode(True)
    assert controller.is_congested_mode, "Should be in congested mode"
    assert controller.adaptive_params['min_green'] > original_min_green, "Min green should increase"
    assert controller.adaptive_params['max_green'] > original_max_green, "Max green should increase"
    assert controller.adaptive_params['flow_weight'] > original_flow_weight, "Flow weight should increase"
    print("✓ Congested mode activation successful")
    
    # Test switching back to normal mode
    controller.set_congested_mode(False)
    assert not controller.is_congested_mode, "Should be back in normal mode"
    assert controller.adaptive_params['min_green'] == original_min_green, "Min green should restore"
    assert controller.adaptive_params['max_green'] == original_max_green, "Max green should restore"
    assert controller.adaptive_params['flow_weight'] == original_flow_weight, "Flow weight should restore"
    print("✓ Normal mode restoration successful")

def test_parameter_values():
    """Test that congested mode parameters are correctly calculated"""
    print("\nTesting parameter values...")
    
    controller = SmartTrafficController(mode="test")
    
    # Check congested parameters are higher than normal
    assert controller.congested_adaptive_params['min_green'] > controller.normal_adaptive_params['min_green']
    assert controller.congested_adaptive_params['max_green'] > controller.normal_adaptive_params['max_green']
    assert controller.congested_adaptive_params['flow_weight'] > controller.normal_adaptive_params['flow_weight']
    assert controller.congested_adaptive_params['queue_weight'] > controller.normal_adaptive_params['queue_weight']
    
    # Check bounds are respected
    assert controller.congested_adaptive_params['min_green'] <= 90
    assert controller.congested_adaptive_params['max_green'] <= 120
    assert controller.congested_adaptive_params['flow_weight'] <= 1.0
    assert controller.congested_adaptive_params['queue_weight'] <= 1.0
    
    print("✓ Parameter values correctly calculated and bounded")

if __name__ == "__main__":
    try:
        test_congestion_detection()
        test_mode_switching()
        test_parameter_values()
        print("\n🎉 All tests passed! Congested mode implementation is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)