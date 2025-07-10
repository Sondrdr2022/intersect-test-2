# Congested Mode Feature Documentation

## Overview
The SmartTrafficController now includes a congested mode feature that automatically detects traffic congestion and adapts traffic light parameters to improve throughput during heavy traffic periods.

## Key Features

### Congestion Detection
- **Method**: `detect_congestion(lane_data)`
- **Logic**: Analyzes average queue length and waiting time across all lanes
- **Thresholds**: 
  - Average queue length ≥ 8 vehicles
  - Average waiting time ≥ 60 seconds
- **Returns**: `True` if congestion detected, `False` otherwise

### Mode Switching
- **Method**: `set_congested_mode(enabled: bool)`
- **Automatic**: Called automatically in `run_step()` based on congestion detection
- **Safe**: Preserves all RL/Q-table mechanisms and ongoing learning

### Parameter Adjustments in Congested Mode
- **Green Time**: Increased min_green (×1.5) and max_green (×1.8)
- **Flow Priority**: Increased flow_weight (×1.4) and queue_weight (×1.3)
- **Wait Priority**: Increased wait_weight (×1.2)
- **Bounds**: All weights capped at 1.0, green times bounded appropriately

## Usage
The feature works automatically - no code changes needed for existing implementations. The controller will:
1. Detect congestion every simulation step
2. Switch to congested mode when congestion is detected
3. Revert to normal mode when congestion subsides
4. Log mode changes with relevant parameter values

## Implementation Details
- **Normal Parameters**: Stored in `self.normal_adaptive_params`
- **Congested Parameters**: Stored in `self.congested_adaptive_params`
- **Current State**: Tracked in `self.is_congested_mode`
- **RL Preservation**: Q-table and learning mechanisms remain unchanged

## Benefits
- **Adaptive**: Automatically responds to changing traffic conditions
- **Throughput**: Improves traffic flow during congestion
- **Learning**: Continues RL training in both modes
- **Safe**: Non-destructive parameter switching