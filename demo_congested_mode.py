#!/usr/bin/env python3
"""
Demonstration of congested mode feature.
Shows how the SmartTrafficController automatically adapts to traffic conditions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple mock for demonstration
class MockTraci:
    class simulation:
        @staticmethod
        def getTime():
            return 100.0
    
    class lane:
        @staticmethod
        def getIDList():
            return []

sys.modules['traci'] = MockTraci()

from Lane2 import SmartTrafficController

def demonstrate_congested_mode():
    print("🚦 SmartTrafficController Congested Mode Demonstration")
    print("=" * 60)
    
    # Create controller
    controller = SmartTrafficController(mode="demo")
    
    print("\n📊 Initial Configuration:")
    print(f"   Normal mode: {not controller.is_congested_mode}")
    print(f"   Min green time: {controller.adaptive_params['min_green']}s")
    print(f"   Max green time: {controller.adaptive_params['max_green']}s")
    print(f"   Flow weight: {controller.adaptive_params['flow_weight']}")
    print(f"   Queue weight: {controller.adaptive_params['queue_weight']}")
    
    # Simulate normal traffic
    print("\n🟢 Simulating Normal Traffic:")
    normal_traffic = {
        'main_st_1': {'queue_length': 3, 'waiting_time': 25},
        'main_st_2': {'queue_length': 2, 'waiting_time': 20},
        'side_st_1': {'queue_length': 4, 'waiting_time': 30},
        'side_st_2': {'queue_length': 1, 'waiting_time': 15}
    }
    
    is_congested = controller.detect_congestion(normal_traffic)
    print(f"   Average queue length: {sum(d['queue_length'] for d in normal_traffic.values()) / len(normal_traffic):.1f}")
    print(f"   Average waiting time: {sum(d['waiting_time'] for d in normal_traffic.values()) / len(normal_traffic):.1f}s")
    print(f"   Congestion detected: {is_congested}")
    
    # Simulate congested traffic
    print("\n🔴 Simulating Congested Traffic:")
    congested_traffic = {
        'main_st_1': {'queue_length': 12, 'waiting_time': 85},
        'main_st_2': {'queue_length': 10, 'waiting_time': 75},
        'side_st_1': {'queue_length': 15, 'waiting_time': 95},
        'side_st_2': {'queue_length': 8, 'waiting_time': 65}
    }
    
    is_congested = controller.detect_congestion(congested_traffic)
    print(f"   Average queue length: {sum(d['queue_length'] for d in congested_traffic.values()) / len(congested_traffic):.1f}")
    print(f"   Average waiting time: {sum(d['waiting_time'] for d in congested_traffic.values()) / len(congested_traffic):.1f}s")
    print(f"   Congestion detected: {is_congested}")
    
    # Activate congested mode
    print("\n⚡ Activating Congested Mode:")
    controller.set_congested_mode(True)
    
    print(f"\n📈 Updated Configuration:")
    print(f"   Congested mode: {controller.is_congested_mode}")
    print(f"   Min green time: {controller.adaptive_params['min_green']}s (+{controller.adaptive_params['min_green'] - 20}s)")
    print(f"   Max green time: {controller.adaptive_params['max_green']}s (+{controller.adaptive_params['max_green'] - 50}s)")
    print(f"   Flow weight: {controller.adaptive_params['flow_weight']} (+{controller.adaptive_params['flow_weight'] - 0.8:.1f})")
    print(f"   Queue weight: {controller.adaptive_params['queue_weight']} (+{controller.adaptive_params['queue_weight'] - 0.5:.2f})")
    
    # Show the difference
    print("\n🎯 Key Benefits:")
    print("   ✓ Longer green times reduce phase switching overhead")
    print("   ✓ Higher flow weight prioritizes vehicle throughput")
    print("   ✓ Higher queue weight gives more attention to backed-up lanes")
    print("   ✓ RL agent continues learning with adjusted parameters")
    
    # Return to normal
    print("\n⬇️ Traffic Subsiding - Returning to Normal Mode:")
    controller.set_congested_mode(False)
    
    print(f"\n✅ Restored Configuration:")
    print(f"   Normal mode: {not controller.is_congested_mode}")
    print(f"   Min green time: {controller.adaptive_params['min_green']}s")
    print(f"   Max green time: {controller.adaptive_params['max_green']}s")
    print(f"   Flow weight: {controller.adaptive_params['flow_weight']}")
    print(f"   Queue weight: {controller.adaptive_params['queue_weight']}")
    
    print("\n🎉 Demonstration Complete!")
    print("The controller automatically adapts to traffic conditions while preserving RL learning.")

if __name__ == "__main__":
    demonstrate_congested_mode()