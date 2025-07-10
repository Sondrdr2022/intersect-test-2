#!/usr/bin/env python3
"""
Integration test for congested mode with run_step workflow.
This tests the full integration of congested mode with the existing system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enhanced Mock traci module for integration testing
class MockTraci:
    _time = 0
    _lane_data = {
        'lane1': {'queue': 5, 'waiting': 30, 'vehicles': 3},
        'lane2': {'queue': 4, 'waiting': 25, 'vehicles': 2},
        'lane3': {'queue': 6, 'waiting': 35, 'vehicles': 4}
    }
    
    @classmethod
    def set_congested_scenario(cls):
        """Set up a congested traffic scenario"""
        cls._lane_data = {
            'lane1': {'queue': 12, 'waiting': 80, 'vehicles': 8},
            'lane2': {'queue': 10, 'waiting': 70, 'vehicles': 6},
            'lane3': {'queue': 15, 'waiting': 90, 'vehicles': 10}
        }
    
    @classmethod
    def set_normal_scenario(cls):
        """Set up a normal traffic scenario"""
        cls._lane_data = {
            'lane1': {'queue': 3, 'waiting': 20, 'vehicles': 2},
            'lane2': {'queue': 2, 'waiting': 15, 'vehicles': 1},
            'lane3': {'queue': 4, 'waiting': 25, 'vehicles': 3}
        }
    
    class simulation:
        @classmethod
        def getTime(cls):
            MockTraci._time += 1
            return MockTraci._time
    
    class lane:
        @classmethod
        def getIDList(cls):
            return list(MockTraci._lane_data.keys())
        
        @classmethod
        def getLength(cls, lane_id):
            return 100.0
        
        @classmethod
        def getLastStepHaltingNumber(cls, lane_id):
            return MockTraci._lane_data.get(lane_id, {}).get('queue', 0)
        
        @classmethod
        def getWaitingTime(cls, lane_id):
            return MockTraci._lane_data.get(lane_id, {}).get('waiting', 0)
        
        @classmethod
        def getLastStepVehicleNumber(cls, lane_id):
            return MockTraci._lane_data.get(lane_id, {}).get('vehicles', 0)
        
        @classmethod
        def getLastStepMeanSpeed(cls, lane_id):
            return 10.0
        
        @classmethod
        def getEdgeID(cls, lane_id):
            return f"edge_{lane_id}"
        
        @classmethod
        def getLastStepVehicleIDs(cls, lane_id):
            return []
    
    class trafficlight:
        @classmethod
        def getIDList(cls):
            return ['tl1']
        
        @classmethod
        def getPhase(cls, tl_id):
            return 0
        
        @classmethod
        def getControlledLanes(cls, tl_id):
            return list(MockTraci._lane_data.keys())

sys.modules['traci'] = MockTraci()

# Now import and test
from Lane2 import SmartTrafficController

def test_run_step_integration():
    """Test full run_step integration with congested mode"""
    print("Testing run_step integration with congested mode...")
    
    controller = SmartTrafficController(mode="test")
    
    # Test 1: Normal scenario
    print("\n1. Testing normal traffic scenario...")
    MockTraci.set_normal_scenario()
    
    initial_mode = controller.is_congested_mode
    controller.run_step()
    
    # Should remain in normal mode
    assert not controller.is_congested_mode, "Should stay in normal mode for normal traffic"
    print("✓ Normal traffic handled correctly")
    
    # Test 2: Congested scenario
    print("\n2. Testing congested traffic scenario...")
    MockTraci.set_congested_scenario()
    
    controller.run_step()
    
    # Should switch to congested mode
    assert controller.is_congested_mode, "Should switch to congested mode for heavy traffic"
    print("✓ Congested traffic correctly triggers congested mode")
    
    # Verify parameters were adjusted
    assert controller.adaptive_params['min_green'] > 20, "Min green should be increased"
    assert controller.adaptive_params['max_green'] > 50, "Max green should be increased"
    assert controller.adaptive_params['flow_weight'] > 0.8, "Flow weight should be increased"
    print("✓ Parameters correctly adjusted for congested mode")
    
    # Test 3: Return to normal
    print("\n3. Testing return to normal traffic...")
    MockTraci.set_normal_scenario()
    
    controller.run_step()
    
    # Should return to normal mode
    assert not controller.is_congested_mode, "Should return to normal mode"
    assert controller.adaptive_params['min_green'] == 20, "Min green should be restored"
    assert controller.adaptive_params['max_green'] == 50, "Max green should be restored"
    assert controller.adaptive_params['flow_weight'] == 0.8, "Flow weight should be restored"
    print("✓ Normal mode correctly restored")

def test_parameter_persistence():
    """Test that parameters persist correctly between mode switches"""
    print("\nTesting parameter persistence...")
    
    controller = SmartTrafficController(mode="test")
    
    # Store original values
    original_params = controller.adaptive_params.copy()
    
    # Switch to congested mode
    MockTraci.set_congested_scenario()
    controller.run_step()
    congested_params = controller.adaptive_params.copy()
    
    # Switch back to normal
    MockTraci.set_normal_scenario()
    controller.run_step()
    restored_params = controller.adaptive_params.copy()
    
    # Verify restoration
    assert original_params == restored_params, "Parameters should be exactly restored"
    assert congested_params != restored_params, "Congested parameters should be different"
    print("✓ Parameter persistence works correctly")

def test_multiple_mode_switches():
    """Test multiple rapid mode switches"""
    print("\nTesting multiple mode switches...")
    
    controller = SmartTrafficController(mode="test")
    
    # Simulate multiple switches
    scenarios = [
        (MockTraci.set_normal_scenario, False),
        (MockTraci.set_congested_scenario, True),
        (MockTraci.set_normal_scenario, False),
        (MockTraci.set_congested_scenario, True),
        (MockTraci.set_normal_scenario, False)
    ]
    
    for i, (scenario_func, expected_congested) in enumerate(scenarios):
        scenario_func()
        controller.run_step()
        assert controller.is_congested_mode == expected_congested, f"Switch {i+1} failed"
    
    print("✓ Multiple mode switches work correctly")

if __name__ == "__main__":
    try:
        test_run_step_integration()
        test_parameter_persistence()
        test_multiple_mode_switches()
        print("\n🎉 All integration tests passed! Congested mode is fully integrated.")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)