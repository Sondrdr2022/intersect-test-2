import os
import sys
import traci
import numpy as np
import pandas as pd
from collections import defaultdict
import math
import time
import datetime
import pickle
import json
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set SUMO_HOME environment variable
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

def detect_turning_lanes_with_traci(sumo_cfg_path):
    sumo_binary = "sumo"
    sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], "bin", sumo_binary), 
                "-c", sumo_cfg_path, "--start", "--quit-on-end", "--no-step-log"]
    traci.start(sumo_cmd)
    left_turn_lanes = set()
    right_turn_lanes = set()
    for lane_id in traci.lane.getIDList():
        links = traci.lane.getLinks(lane_id)
        for conn in links:
            if len(conn) > 6:
                if conn[6] == 'l':
                    left_turn_lanes.add(lane_id)
                if conn[6] == 'r':
                    right_turn_lanes.add(lane_id)
            elif len(conn) > 3:
                if conn[3] == 'l':
                    left_turn_lanes.add(lane_id)
                if conn[3] == 'r':
                    right_turn_lanes.add(lane_id)
    traci.close()
    return left_turn_lanes, right_turn_lanes

class EnhancedQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 q_table_file="enhanced_q_table.pkl", mode="train"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.training_data = []
        self.q_table_file = q_table_file
        self._loaded_training_count = 0
        self.reward_components = []
        self.reward_history = []
        self.learning_rate_decay = 0.999
        self.min_learning_rate = 0.01
        self.consecutive_no_improvement = 0
        self.max_no_improvement = 100
        self.mode = mode  # "train", "eval", or "adaptive"
        
        # Cache for frequently accessed states
        self.state_key_cache = {}
        self.max_cache_size = 10000

        if self.mode == "eval":
            self.epsilon = 0.0
        elif self.mode == "adaptive":
            self.epsilon = 0.01

    def is_valid_state(self, state):
        if isinstance(state, (list, np.ndarray)):
            state_array = np.array(state)
            if np.isnan(state_array).any() or np.isinf(state_array).any():
                return False
            if (np.abs(state_array) > 100).any():
                return False
            if not isinstance(state, np.ndarray):
                return False
            if len(state) != self.state_size:
                return False
        return True

    def get_action(self, state, lane_id=None):
        state_key = self._state_to_key(state, lane_id)
        if self.mode == "train":
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.action_size)
            else:
                return np.argmax(self.q_table[state_key])
        else:
            # eval or adaptive: exploit only
            return np.argmax(self.q_table[state_key])

    def _state_to_key(self, state, lane_id=None):
        """Convert state to hashable key with caching for performance"""
        try:
            # Check cache first for better performance
            cache_key = (id(state), lane_id) if lane_id else id(state)
            if cache_key in self.state_key_cache:
                return self.state_key_cache[cache_key]
                
            if isinstance(state, np.ndarray):
                key = tuple(np.round(state, 2))
            elif isinstance(state, list):
                state_array = np.round(np.array(state), 2)
                key = tuple(state_array.tolist())
            else:
                key = tuple(state) if hasattr(state, '__iter__') else (state,)
                
            if lane_id is not None:
                result_key = (lane_id, key)
            else:
                result_key = key
                
            # Store in cache
            if len(self.state_key_cache) >= self.max_cache_size:
                # Clear half the cache if full
                keys_to_remove = list(self.state_key_cache.keys())[:self.max_cache_size//2]
                for k in keys_to_remove:
                    self.state_key_cache.pop(k, None)
                    
            self.state_key_cache[cache_key] = result_key
            return result_key
            
        except Exception:
            return (lane_id, (0,)) if lane_id is not None else (0,)

    def update_q_table(self, state, action, reward, next_state, lane_info=None):
        if self.mode == "eval":
            return  # No learning in eval mode
        if not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        lane_id = lane_info['lane_id'] if lane_info and 'lane_id' in lane_info else None
        state_key = self._state_to_key(state, lane_id)
        next_state_key = self._state_to_key(next_state, lane_id)
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        if np.isnan(new_q) or np.isinf(new_q):
            new_q = current_q
        self.q_table[state_key][action] = new_q
        
        # Only store essential data to reduce memory usage
        essential_info = {
            'state_key': state_key,
            'action': action,
            'reward': reward,
            'q_value': new_q,
            'timestamp': time.time(),
            'lane_id': lane_id
        }
        
        # Add lane info selectively
        if lane_info:
            if 'reward_components' in lane_info:
                essential_info['reward_components'] = lane_info['reward_components']
            if 'raw_reward' in lane_info:
                essential_info['raw_reward'] = lane_info['raw_reward']
                
        self.training_data.append(essential_info)
        
        # Trim training data if it gets too large
        if len(self.training_data) > 20000:
            self.training_data = self.training_data[-10000:]
            
        self._update_adaptive_parameters(reward)

    def _update_adaptive_parameters(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        if self.mode == "train":
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if len(self.reward_history) >= 50:
            recent_avg = np.mean(self.reward_history[-50:])
            older_avg = np.mean(self.reward_history[-100:-50])
            if recent_avg <= older_avg:
                self.consecutive_no_improvement += 1
            else:
                self.consecutive_no_improvement = 0
            if self.consecutive_no_improvement > self.max_no_improvement:
                self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
                self.consecutive_no_improvement = 0

    def load_model(self, filepath=None):
        if filepath is None:
            filepath = self.q_table_file
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(self.action_size))
                loaded_q_table = model_data.get('q_table', {})
                self.q_table.update(loaded_q_table)
                self._loaded_training_count = len(model_data.get('training_data', []))
                params = model_data.get('params', {})
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.discount_factor = params.get('discount_factor', self.discount_factor)
                self.epsilon = params.get('epsilon', self.epsilon)
                adaptive_params = model_data.get('adaptive_params', {})
                print(f"Loaded Q-table with {len(self.q_table)} states from {filepath}")
                if adaptive_params:
                    print(f"📋 Loaded adaptive parameters from previous run")
                return True, adaptive_params
        except Exception as e:
            print(f"Error loading model: {e}")
        print("No existing Q-table, starting fresh")
        return False, {}

    def save_model(self, filepath=None, adaptive_params=None):
        if filepath is None:
            filepath = self.q_table_file
        try:
            if os.path.exists(filepath):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{filepath}.bak_{timestamp}"
                for _ in range(3):
                    try:
                        os.rename(filepath, backup_path)
                        break
                    except Exception as e:
                        print(f"Retrying backup: {e}")
                        time.sleep(0.5)
            model_data = {
                'q_table': dict(self.q_table),
                'params': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon': self.epsilon,
                    'epsilon_decay': self.epsilon_decay,
                    'min_epsilon': self.min_epsilon
                },
                'metadata': {
                    'last_updated': datetime.datetime.now().isoformat(),
                    'training_count': len(self.training_data)
                }
            }
            
            # Only store last 100 training entries to save memory
            if self.training_data:
                model_data['training_data'] = self.training_data[-100:]
                model_data['metadata']['average_reward'] = np.mean([x['reward'] for x in self.training_data[-100:]])
            
            if adaptive_params:
                model_data['adaptive_params'] = adaptive_params.copy()
                print(f"💾 Saving adaptive parameters")
                
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            print(f"✅ Model saved with {len(self.q_table)} states")
            
            # Don't clear training data here - just let it be trimmed as needed
            
        except Exception as e:
            print(f"Error saving model: {e}")

class SmartTrafficController:
    def __init__(self, state_size=14, action_size=5, sumocfg_path=None, mode="train"):
        self.rl_agent = EnhancedQLearningAgent(state_size=12, action_size=5, mode=mode)
        self.mode = mode
        self.step_count = 0
        self.lane_scores = defaultdict(int)
        self.lane_states = defaultdict(lambda: "UNKNOWN")
        self.consecutive_states = defaultdict(int)
        self.lane_to_tl = {}
        self.edge_to_routes = defaultdict(set)
        self.previous_states = {}
        self.previous_actions = {}
        self.current_episode = 0
        self.last_phase_change = defaultdict(float)
        self.last_green_time = defaultdict(float)
        self.phase_utilization = defaultdict(int)
        self.priority_vehicles = defaultdict(bool)
        self.ambulance_active = defaultdict(lambda: False)
        self.ambulance_start_time = defaultdict(float)
        self.left_turn_lanes = set()
        self.right_turn_lanes = set()
        self.tl_logic_cache = {}
        self.lane_cache = {}
        self.vehicle_cache = {}
        self.last_lane_vehicles = defaultdict(set)
        self.last_arrival_time = defaultdict(float)
        self.log_frequency = 500  # Log only every N steps
        self.subscription_initialized = False
        
        # Network data caches
        self.lane_lengths = {}
        self.controlled_lanes_cache = {}
        
        default_adaptive_params = {
            'min_green': 20,
            'max_green': 50,
            'starvation_threshold': 120,
            'reward_scale': 50,
            'queue_weight': 0.5,
            'wait_weight': 0.2,
            'flow_weight': 0.8,
            'speed_weight': 0.1,
            'left_turn_priority': 1.5      
        }
        success, loaded_adaptive_params = self.rl_agent.load_model()
        if success and loaded_adaptive_params:
            self.adaptive_params = loaded_adaptive_params.copy()
            print(f"🔄 Loaded adaptive parameters: {self.adaptive_params}")
        else:
            self.adaptive_params = default_adaptive_params.copy()
            print("🆕 Using default adaptive parameters")
        self.norm_bounds = {
            'queue': 20,
            'wait': 120,
            'density': 2.0,
            'speed': 20,
            'flow': 20,
            'time_since_green': 120,
            'arrival_rate': 10
        }
        
        if sumocfg_path is not None:
            try:
                left_lanes, right_lanes = detect_turning_lanes_with_traci(sumocfg_path)
                self.left_turn_lanes = left_lanes
                self.right_turn_lanes = right_lanes
                print(f"Auto-detected left-turn lanes: {len(self.left_turn_lanes)}")
                print(f"Auto-detected right-turn lanes: {len(self.right_turn_lanes)}")
            except Exception as e:
                print(f"Error detecting turning lanes: {e}")
        else:
            self.left_turn_lanes = set()
            self.right_turn_lanes = set()
    
    def _setup_subscriptions(self):
        """Set up TraCI subscriptions for efficient data collection"""
        try:
            if self.subscription_initialized:
                return
                
            print("Setting up TraCI subscriptions for efficient data collection...")
            
            # 1. Lane subscriptions - subscribe to all lanes at once
            lanes = traci.lane.getIDList()
            
            # Cache lane lengths
            for lane_id in lanes:
                self.lane_lengths[lane_id] = traci.lane.getLength(lane_id)
                
            # Subscribe to lane variables
            for lane_id in lanes:
                traci.lane.subscribe(lane_id, [
                    traci.constants.LAST_STEP_VEHICLE_NUMBER,
                    traci.constants.LAST_STEP_MEAN_SPEED,
                    traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    traci.constants.LAST_STEP_VEHICLE_ID_LIST,
                    traci.constants.VAR_WAITING_TIME
                ])
            
            # 2. Traffic light subscriptions
            for tl_id in traci.trafficlight.getIDList():
                traci.trafficlight.subscribe(tl_id, [
                    traci.constants.TL_CURRENT_PHASE,
                    traci.constants.TL_CURRENT_PROGRAM,
                    traci.constants.VAR_NEXT_SWITCH
                ])
                
                # Cache controlled lanes
                self.controlled_lanes_cache[tl_id] = traci.trafficlight.getControlledLanes(tl_id)
                
                # Build lane to traffic light mapping
                for lane in self.controlled_lanes_cache[tl_id]:
                    self.lane_to_tl[lane] = tl_id
                
                # Cache traffic light logic once
                self.tl_logic_cache[tl_id] = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            
            # 3. Vehicle type subscription (for emergency vehicles)
            vehicle_types = set()
            for v_id in traci.vehicle.getIDList():
                vehicle_types.add(traci.vehicle.getVehicleClass(v_id))
                
            # Subscribe to all vehicles (will auto-update for new vehicles)
            if traci.vehicle.getIDCount() > 0:
                traci.vehicle.subscribeContext("", traci.constants.CMD_GET_VEHICLE_VARIABLE, 0,
                                              [traci.constants.VAR_VEHICLECLASS, 
                                               traci.constants.VAR_ROUTE_ID,
                                               traci.constants.VAR_LANE_ID])
            
            self.subscription_initialized = True
            print(f"✅ Subscriptions set up for {len(lanes)} lanes and {len(self.controlled_lanes_cache)} traffic lights")
            
        except Exception as e:
            print(f"Error setting up subscriptions: {e}")

    def _collect_enhanced_lane_data_batched(self):
        """Collect comprehensive lane data using TraCI subscriptions for efficiency"""
        lane_data = {}
        try:
            current_time = traci.simulation.getTime()
            
            # Get all subscription results at once (much faster than individual calls)
            lane_subscriptions = traci.lane.getSubscriptionResults()
            vehicle_context = None
            
            # Try to get vehicle context data if available
            try:
                if traci.vehicle.getIDCount() > 0:
                    vehicle_context = traci.vehicle.getContextSubscriptionResults("")
            except:
                vehicle_context = None
                
            # Pre-compute edge-level aggregates
            edge_queues = defaultdict(float)
            edge_flows = defaultdict(float)
            route_queues = defaultdict(float)
            route_flows = defaultdict(float)
            
            # Process all lanes at once
            for lane_id, data in lane_subscriptions.items():
                if lane_id not in self.lane_lengths or self.lane_lengths[lane_id] <= 0:
                    continue
                
                lane_length = self.lane_lengths[lane_id]
                vehicle_count = data[traci.constants.LAST_STEP_VEHICLE_NUMBER]
                queue_length = data[traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER]
                mean_speed = data[traci.constants.LAST_STEP_MEAN_SPEED]
                waiting_time = data[traci.constants.VAR_WAITING_TIME]
                vehicle_ids = data[traci.constants.LAST_STEP_VEHICLE_ID_LIST]
                
                edge_id = traci.lane.getEdgeID(lane_id)
                
                # Check for ambulances without querying each vehicle
                ambulance_detected = False
                route_id = ""
                
                # Use vehicle context data if available (much faster)
                if vehicle_context and vehicle_ids:
                    for vid in vehicle_ids:
                        if vid in vehicle_context:
                            v_data = vehicle_context[vid]
                            
                            # Get first vehicle's route as sample
                            if not route_id and traci.constants.VAR_ROUTE_ID in v_data:
                                route_id = v_data[traci.constants.VAR_ROUTE_ID]
                                
                            # Check for emergency vehicles
                            if traci.constants.VAR_VEHICLECLASS in v_data:
                                if v_data[traci.constants.VAR_VEHICLECLASS] in ['emergency', 'authority']:
                                    ambulance_detected = True
                                    break
                
                # Store lane data
                lane_data[lane_id] = {
                    'queue_length': queue_length,
                    'waiting_time': waiting_time,
                    'density': vehicle_count / lane_length,
                    'mean_speed': mean_speed,
                    'flow': vehicle_count,
                    'lane_id': lane_id,
                    'edge_id': edge_id,
                    'route_id': route_id,
                    'ambulance': ambulance_detected,
                    'left_turn': lane_id in self.left_turn_lanes,
                    'right_turn': lane_id in self.right_turn_lanes,
                    'tl_id': self.lane_to_tl.get(lane_id, ''),
                    'arrival_rate': self._calculate_arrival_rate_efficient(lane_id, vehicle_ids, current_time)
                }
                
                # Update edge aggregates
                edge_queues[edge_id] += queue_length
                edge_flows[edge_id] += vehicle_count
                
                # Update route aggregates
                if route_id:
                    route_queues[route_id] += queue_length
                    route_flows[route_id] += vehicle_count
            
            # Second pass: add aggregated metrics (can be done in one pass if optimizing further)
            for lane_id, data in lane_data.items():
                data['queue_route'] = route_queues.get(data.get('route_id', ''), 0)
                data['flow_route'] = route_flows.get(data.get('route_id', ''), 0)
                
        except Exception as e:
            print(f"Error in _collect_enhanced_lane_data_batched: {e}")
            
        return lane_data
    
    def _calculate_arrival_rate_efficient(self, lane_id, current_vehicles, current_time):
        """Calculate arrival rate efficiently using sets"""
        try:
            current_vehicles_set = set(current_vehicles)
            prev_vehicles = self.last_lane_vehicles.get(lane_id, set())
            arrivals = current_vehicles_set - prev_vehicles
            last_time = self.last_arrival_time.get(lane_id, current_time-1)
            
            # Avoid division by zero
            delta_time = max(1e-3, current_time - last_time)
            arrival_rate = len(arrivals) / delta_time  # vehicles per second

            # Update memory for next call
            self.last_lane_vehicles[lane_id] = current_vehicles_set
            self.last_arrival_time[lane_id] = current_time
            return arrival_rate
        except Exception as e:
            if self.step_count % self.log_frequency == 0:
                print(f"Error calculating arrival rate for {lane_id}: {e}")
            return 0.0

    def run_step(self):
        """Main simulation step function with optimizations"""
        try:
            self.step_count += 1
            
            # Setup subscriptions on first step
            if not self.subscription_initialized:
                self._setup_subscriptions()
                
            current_time = traci.simulation.getTime()
            
            # Get lane data using efficient batched method
            lane_data = self._collect_enhanced_lane_data_batched()
            if not lane_data:
                return
                
            # Update lane status every step but only for actively used lanes
            lane_status = self._update_lane_status_and_score(lane_data)
            
            # Adjust traffic lights
            self._adjust_traffic_lights_efficient(lane_data, lane_status, current_time)
            
            # Process RL learning only every few steps to reduce overhead
            if self.step_count % 3 == 0:  # Reduced frequency for learning
                self._process_rl_learning(lane_data, current_time)
                
            # Log status periodically rather than every step
            if self.step_count % self.log_frequency == 0:
                print(f"Step {self.step_count}: Processing {len(lane_data)} lanes, {len(self.lane_to_tl)} traffic lights")
                
        except Exception as e:
            print(f"Error in run_step: {e}")

    def _update_lane_status_and_score(self, lane_data):
        """Enhanced lane status update with reduced processing"""
        status = {}
        try:
            # Filter to only process lanes with significant activity
            active_lanes = {lane_id: data for lane_id, data in lane_data.items() 
                          if data['flow'] > 0 or data['queue_length'] > 0}
            
            for lane_id, data in active_lanes.items():
                # Get normalized metrics
                queue_norm = data['queue_length'] / self.norm_bounds['queue']
                wait_norm = data['waiting_time'] / self.norm_bounds['wait']
                speed_norm = data['mean_speed'] / self.norm_bounds['speed']
                flow_norm = data['flow'] / self.norm_bounds['flow']
                arrival_norm = data['arrival_rate'] / self.norm_bounds['arrival_rate']
                
                # Calculate composite score
                composite_score = (
                    self.adaptive_params['queue_weight'] * queue_norm +
                    self.adaptive_params['wait_weight'] * wait_norm +
                    (1 - min(speed_norm, 1.0)) +  # Inverse of speed
                    (1 - min(flow_norm, 1.0)) +   # Inverse of flow
                    arrival_norm * 0.5            # Predictive component
                )
                
                # Determine status based on dynamic thresholds
                if composite_score > 0.8:  # BAD threshold
                    status[lane_id] = "BAD"
                    delta = -max(2, min(8, int(composite_score * 8)))
                elif composite_score < 0.3:  # GOOD threshold
                    status[lane_id] = "GOOD"
                    delta = max(2, min(8, int((1 - composite_score) * 8)))
                else:
                    status[lane_id] = "NORMAL"
                    delta = 0

                # Update lane score with momentum
                if self.lane_states[lane_id] == status[lane_id]:
                    self.consecutive_states[lane_id] += 1
                    momentum_factor = min(3.0, 1.0 + self.consecutive_states[lane_id] * 0.1)
                    delta *= momentum_factor              
                else:
                    self.lane_states[lane_id] = status[lane_id]
                    self.consecutive_states[lane_id] = 1
                
                self.lane_scores[lane_id] += delta

                # Decay toward zero if NORMAL
                if status[lane_id] == "NORMAL":
                    decay_factor = 1.5 if composite_score < 0.4 else 1.0
                    if self.lane_scores[lane_id] > 0:
                        self.lane_scores[lane_id] = max(0, self.lane_scores[lane_id] - decay_factor)
                    elif self.lane_scores[lane_id] < 0:
                        self.lane_scores[lane_id] = min(0, self.lane_scores[lane_id] + decay_factor)
                
                # Ensure score stays within reasonable bounds
                self.lane_scores[lane_id] = max(-50, min(50, self.lane_scores[lane_id]))
                
        except Exception as e:
            print(f"Error in _update_lane_status_and_score: {e}")
            
        return status

    def _adjust_traffic_lights_efficient(self, lane_data, lane_status, current_time):
        """Optimized traffic light adjustment with priority handling"""
        try:
            tl_subscriptions = traci.trafficlight.getSubscriptionResults()
            
            # Process only traffic lights that need updates
            active_tls = {}
            for tl_id, data in tl_subscriptions.items():
                next_switch = data.get(traci.constants.VAR_NEXT_SWITCH, current_time)
                # Only process if switch time is close or we have an emergency
                if next_switch - current_time <= 5:
                    active_tls[tl_id] = data
                    
            # Add any traffic lights that have emergency vehicles
            for lane_id, data in lane_data.items():
                if data['ambulance'] and lane_id in self.lane_to_tl:
                    tl_id = self.lane_to_tl[lane_id]
                    if tl_id not in active_tls and tl_id in tl_subscriptions:
                        active_tls[tl_id] = tl_subscriptions[tl_id]
            
            # Process only active traffic lights
            for tl_id, tl_data in active_tls.items():
                try:
                    controlled_lanes = self.controlled_lanes_cache.get(tl_id, [])
                    if not controlled_lanes:
                        continue
                    
                    # Check for priority conditions
                    priority_handled = self._handle_priority_conditions_efficient(
                        tl_id, controlled_lanes, lane_data, current_time)
                    
                    # Only proceed with normal control if no priority was handled
                    if not priority_handled:
                        self._perform_normal_control_efficient(
                            tl_id, controlled_lanes, lane_data, current_time)
                        
                except Exception as e:
                    print(f"Error adjusting traffic light {tl_id}: {e}")
                    
        except Exception as e:
            print(f"Error in _adjust_traffic_lights_efficient: {e}")

    def _handle_priority_conditions_efficient(self, tl_id, controlled_lanes, lane_data, current_time):
        """Efficiently handle priority conditions (ambulances, left turns)"""
        # Check for ambulances first (highest priority)
        ambulance_lanes = []
        left_turn_lanes = []
        
        # Process lane data in one pass
        for lane_id in controlled_lanes:
            if lane_id not in lane_data:
                continue
                
            if lane_data[lane_id]['ambulance']:
                ambulance_lanes.append(lane_id)
            elif (lane_data[lane_id]['left_turn'] and 
                  (lane_data[lane_id]['queue_length'] > 3 or 
                   lane_data[lane_id]['waiting_time'] > 10)):
                left_turn_lanes.append(lane_id)
        
        if ambulance_lanes:
            self._handle_ambulance_priority(tl_id, ambulance_lanes, current_time)
            return True
            
        if left_turn_lanes:
            return self._handle_protected_left_turn(tl_id, left_turn_lanes, lane_data, current_time)
            
        return False

    def _handle_ambulance_priority(self, tl_id, ambulance_lanes, current_time):
        """Handle ambulance priority with timeout"""
        try:
            if not self.ambulance_active[tl_id]:
                # New ambulance detected - activate priority
                ambulance_lane = ambulance_lanes[0]
                phase_index = self._find_phase_for_lane(tl_id, ambulance_lane)
                
                if phase_index is not None:
                    # Set green for ambulance lane with extended duration
                    traci.trafficlight.setPhase(tl_id, phase_index)
                    traci.trafficlight.setPhaseDuration(tl_id, 30)
                    
                    self.ambulance_active[tl_id] = True
                    self.ambulance_start_time[tl_id] = current_time
                    
                    # Log only once per traffic light
                    if self.step_count % self.log_frequency == 0:
                        print(f"🚑 AMBULANCE PRIORITY: Green at {tl_id}")
            else:
                # Check if priority should be released
                if current_time - self.ambulance_start_time[tl_id] > 30:  # 30s timeout
                    self.ambulance_active[tl_id] = False
                    
        except Exception as e:
            print(f"Error in _handle_ambulance_priority: {e}")

    def _handle_protected_left_turn(self, tl_id, left_turn_lanes, lane_data, current_time):
        """Handle protected left turn phase activation (exclusive left-turn green)"""
        try:
            # Find the left turn lane with highest priority
            left_turn_lanes.sort(key=lambda x: (
                lane_data[x]['queue_length'],
                lane_data[x]['waiting_time']
            ), reverse=True)
            
            target_lane = left_turn_lanes[0]
            current_phase = traci.trafficlight.getPhase(tl_id)
            
            # Find the best protected left-turn phase
            left_turn_phase = self._find_best_left_turn_phase(tl_id, target_lane, lane_data)
            
            if left_turn_phase is not None and current_phase != left_turn_phase:
                # Calculate duration based on queue and waiting time
                queue_length = lane_data[target_lane]['queue_length']
                waiting_time = lane_data[target_lane]['waiting_time']
                base_duration = 5 + min(queue_length * 0.5, 10) + min(waiting_time * 0.1, 5)
                
                # Apply adaptive parameters
                duration = min(
                    max(base_duration, self.adaptive_params['min_green']),
                    self.adaptive_params['max_green']
                )
                
                # Set the protected left-turn phase
                traci.trafficlight.setPhase(tl_id, left_turn_phase)
                traci.trafficlight.setPhaseDuration(tl_id, duration)
                
                # Only log occasionally
                if self.step_count % self.log_frequency == 0:
                    print(f"↩️ PROTECTED LEFT TURN: at {tl_id} (duration={duration:.1f}s)")
                
                # Update tracking variables
                self.last_green_time[target_lane] = current_time
                self.last_phase_change[tl_id] = current_time
                self.phase_utilization[(tl_id, left_turn_phase)] += 1
                
                return True  # Indicate that left turn was handled
                
        except Exception as e:
            print(f"Error in _handle_protected_left_turn: {e}")
        return False

    def _find_best_left_turn_phase(self, tl_id, left_turn_lane, lane_data):
        """Finds the best phase for protected left turns"""
        try:
            logic = self.tl_logic_cache.get(tl_id)
            if not logic:
                return None

            controlled_lanes = self.controlled_lanes_cache.get(tl_id, [])
            if not controlled_lanes:
                return None
                
            target_idx = controlled_lanes.index(left_turn_lane)
            
            best_phase = None
            best_score = -1
            
            for phase_idx, phase in enumerate(logic.phases):
                state = phase.state.upper()
                
                # Skip if this phase doesn't give green to our left-turn lane
                if target_idx >= len(state) or state[target_idx] != 'G':
                    continue
                    
                # Calculate phase score
                score = 0
                
                # Prefer exclusive left-turn phases (where only left turns are green)
                is_exclusive = True
                for i, signal in enumerate(state):
                    if i != target_idx and signal != 'r' and signal != 'R':
                        is_exclusive = False
                        break
                        
                if is_exclusive:
                    score += 20  # Bonus for exclusive left-turn phase
                
                # Consider other lanes in this phase
                for i, lane in enumerate(controlled_lanes):
                    if i >= len(state):
                        continue
                        
                    if state[i] == 'G' and lane in lane_data:
                        # Penalize phases that give green to congested opposing lanes
                        if lane != left_turn_lane and lane_data[lane]['queue_length'] > 5:
                            score -= lane_data[lane]['queue_length'] * 0.5
                            
                # Track the best scoring phase
                if score > best_score:
                    best_score = score
                    best_phase = phase_idx
                    
            return best_phase
            
        except Exception as e:
            print(f"Error in _find_best_left_turn_phase: {e}")
            return None

    def _perform_normal_control_efficient(self, tl_id, controlled_lanes, lane_data, current_time):
        """Perform normal RL-based traffic control with optimizations"""
        try:
            # Filter to active lanes
            active_lanes = [l for l in controlled_lanes if l in lane_data]
            if not active_lanes:
                return
                
            # Select target lane using cached score data
            target_lane = self._select_target_lane_efficient(tl_id, active_lanes, lane_data, current_time)
            
            if not target_lane:
                return
                
            # Get state and RL action
            state = self._create_state_vector_efficient(target_lane, lane_data)
            if not self.rl_agent.is_valid_state(state):
                return
                
            action = self.rl_agent.get_action(state, lane_id=target_lane)
            
            # Execute action with minimum phase duration enforcement
            if current_time - self.last_phase_change.get(tl_id, 0) >= 5:  # 5s minimum
                self._execute_control_action_efficient(tl_id, target_lane, action, lane_data, current_time)
                
        except Exception as e:
            print(f"Error in _perform_normal_control_efficient: {e}")

    def _select_target_lane_efficient(self, tl_id, controlled_lanes, lane_data, current_time):
        """Lane selection with efficient scoring"""
        candidate_lanes = []
        max_queue = 0
        max_wait = 0
        max_arrival = 0
        
        # Find maximum values in one pass
        for lane in controlled_lanes:
            if lane in lane_data:
                data = lane_data[lane]
                max_queue = max(max_queue, data['queue_length'])
                max_wait = max(max_wait, data['waiting_time'])
                max_arrival = max(max_arrival, data['arrival_rate'])
        
        # Avoid division by zero
        max_queue = max(max_queue, 1)
        max_wait = max(max_wait, 1)
        max_arrival = max(max_arrival, 0.1)
        
        # Score and sort lanes
        for lane in controlled_lanes:
            if lane in lane_data:
                data = lane_data[lane]
                
                # Get base score from cached lane scores
                base_score = self.lane_scores.get(lane, 0)
                
                # Calculate normalized factors
                queue_factor = (data['queue_length'] / max_queue) * 10
                wait_factor = (data['waiting_time'] / max_wait) * 5
                arrival_factor = (data.get('arrival_rate', 0) / max_arrival) * 8
                
                # Starvation prevention
                starvation_factor = 0
                last_green = self.last_green_time.get(lane, 0)
                if current_time - last_green > self.adaptive_params['starvation_threshold']:
                    starvation_factor = min(15, (current_time - last_green - 
                                            self.adaptive_params['starvation_threshold']) * 0.3)
                
                # Emergency vehicle priority
                emergency_boost = 20 if data['ambulance'] else 0
                
                # Phase efficiency from cache
                phase_efficiency = 1.0  # Default if not in cache
                current_phase = traci.trafficlight.getPhase(tl_id)
                phase_key = (tl_id, current_phase)
                if phase_key in self.phase_utilization:
                    total = sum(count for (t, _), count in self.phase_utilization.items() if t == tl_id)
                    if total > 0:
                        phase_efficiency = self.phase_utilization.get(phase_key, 0) / total
                
                # Total priority score
                total_score = (
                    base_score + 
                    queue_factor + 
                    wait_factor + 
                    arrival_factor + 
                    starvation_factor + 
                    emergency_boost +
                    phase_efficiency * 5
                )
                
                candidate_lanes.append((lane, total_score))
        
        if not candidate_lanes:
            return None
            
        # Select lane with highest priority
        candidate_lanes.sort(key=lambda x: x[1], reverse=True)
        return candidate_lanes[0][0]

    def _execute_control_action_efficient(self, tl_id, target_lane, action, lane_data, current_time):
        """Execute control action with minimal API calls"""
        try:
            # Get current phase from subscription data if possible
            current_phase = traci.trafficlight.getPhase(tl_id)
            target_phase = self._find_phase_for_lane(tl_id, target_lane)
            
            if target_phase is None:
                target_phase = current_phase  # Fallback
                
            action_executed = False
            
            if action == 0:  # Set green for target lane
                if current_phase != target_phase:
                    green_time = self._calculate_dynamic_green(lane_data[target_lane])
                    traci.trafficlight.setPhase(tl_id, target_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, green_time)
                    self.last_phase_change[tl_id] = current_time
                    self.last_green_time[target_lane] = current_time
                    action_executed = True
                    
            elif action == 1:  # Next phase
                phase_count = len(self.tl_logic_cache.get(tl_id, {}).phases)
                if phase_count > 0:
                    next_phase = (current_phase + 1) % phase_count
                    traci.trafficlight.setPhase(tl_id, next_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                    self.last_phase_change[tl_id] = current_time
                    action_executed = True
                    
            elif action == 2:  # Extend current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    extension = min(15, self.adaptive_params['max_green'] - remaining)
                    if extension > 0:
                        traci.trafficlight.setPhaseDuration(tl_id, remaining + extension)
                        action_executed = True
                except Exception:
                    pass
                    
            elif action == 3:  # Shorten current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    if remaining > self.adaptive_params['min_green'] + 5:
                        reduction = min(5, remaining - self.adaptive_params['min_green'])
                        traci.trafficlight.setPhaseDuration(tl_id, remaining - reduction)
                        action_executed = True
                except Exception:
                    pass
                    
            elif action == 4:  # Balanced phase switch
                balanced_phase = self._get_balanced_phase_efficient(tl_id, lane_data)
                if balanced_phase != current_phase:
                    traci.trafficlight.setPhase(tl_id, balanced_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                    self.last_phase_change[tl_id] = current_time
                    action_executed = True
                    
            # Update phase utilization stats
            if action_executed:
                new_phase = traci.trafficlight.getPhase(tl_id)
                self.phase_utilization[(tl_id, new_phase)] += 1
                
                # Log only occasionally
                if self.step_count % self.log_frequency == 0:
                    action_names = {0: "Green", 1: "Next", 2: "Extend", 3: "Shorten", 4: "Balanced"}
                    print(f"🚦 TL {tl_id}: {action_names.get(action, 'Unknown')} action")
            
        except Exception as e:
            print(f"Error in _execute_control_action_efficient: {e}")

    def _get_balanced_phase_efficient(self, tl_id, lane_data):
        """Get balanced phase with minimal calculation"""
        try:
            logic = self.tl_logic_cache.get(tl_id)
            if not logic:
                return 0
                
            controlled_lanes = self.controlled_lanes_cache.get(tl_id, [])
            if not controlled_lanes:
                return 0
                
            # Score each phase - calculate once and reuse
            phase_scores = []
            
            for phase_idx, phase in enumerate(logic.phases):
                state = phase.state
                score = 0
                
                # Check active lanes with green in this phase
                for lane_idx, lane in enumerate(controlled_lanes):
                    if lane_idx < len(state) and state[lane_idx].upper() == 'G' and lane in lane_data:
                        data = lane_data[lane]
                        phase_score = (data['queue_length'] * 0.5 + 
                                      data['waiting_time'] * 0.1 +
                                      (1 - min(data['mean_speed'] / self.norm_bounds['speed'], 1)) * 10)
                        score += phase_score
                
                phase_scores.append((phase_idx, score))
            
            # Return highest scoring phase
            if phase_scores:
                best_phase = max(phase_scores, key=lambda x: x[1])
                return best_phase[0]
                
            return 0
        except Exception as e:
            print(f"Error in _get_balanced_phase_efficient: {e}")
            return 0

    def _find_phase_for_lane(self, tl_id, target_lane):
        """Find phase that gives green to target lane using cache"""
        try:
            # Check cache first
            cache_key = (tl_id, target_lane)
            if cache_key in self.lane_cache:
                return self.lane_cache[cache_key]
            
            logic = self.tl_logic_cache.get(tl_id)
            if not logic:
                return 0
                
            controlled_lanes = self.controlled_lanes_cache.get(tl_id, [])
            if not controlled_lanes or target_lane not in controlled_lanes:
                return 0
                
            target_idx = controlled_lanes.index(target_lane)
            
            for phase_idx, phase in enumerate(logic.phases):
                state = phase.state
                if target_idx < len(state) and state[target_idx].upper() == 'G':
                    # Store in cache
                    self.lane_cache[cache_key] = phase_idx
                    return phase_idx
                    
        except Exception as e:
            pass
            
        return 0

    def _calculate_dynamic_green(self, lane_data):
        """Calculate dynamic green time based on lane conditions"""
        base_time = self.adaptive_params['min_green']
        queue_factor = min(lane_data['queue_length'] * 0.7, 15)
        density_factor = min(lane_data['density'] * 5, 10)
        emergency_bonus = 10 if lane_data['ambulance'] else 0
        
        total_time = (base_time + queue_factor + density_factor + emergency_bonus)
        return min(max(total_time, self.adaptive_params['min_green']), 
                 self.adaptive_params['max_green'])

    def _process_rl_learning(self, lane_data, current_time):
        """Process RL learning with optimizations"""
        # Process only a subset of lanes each time to distribute computation
        subset_size = min(20, len(lane_data))  # Process at most 20 lanes per step
        lanes_to_process = list(lane_data.keys())
        
        # Randomly sample lanes if we have too many
        if len(lanes_to_process) > subset_size:
            import random
            lanes_to_process = random.sample(lanes_to_process, subset_size)
        
        for lane_id in lanes_to_process:
            data = lane_data[lane_id]
            state = self._create_state_vector_efficient(lane_id, lane_data)
            
            if not self.rl_agent.is_valid_state(state):
                continue
                
            action = self.rl_agent.get_action(state, lane_id=lane_id)
            
            # Calculate reward if we have previous state
            if lane_id in self.previous_states and lane_id in self.previous_actions:
                reward, reward_components, raw_reward = self._calculate_reward_efficient(
                    lane_id, lane_data, self.previous_actions[lane_id], current_time)
                
                # Only log rewards occasionally and for significant values
                if self.step_count % self.log_frequency == 0 and abs(reward) > 0.5:
                    print(f"🏆 Lane {lane_id}: reward {reward:.2f}")
                
                # Update Q-table
                self.rl_agent.update_q_table(
                    self.previous_states[lane_id],
                    self.previous_actions[lane_id],
                    reward,
                    state,
                    lane_info={
                        'lane_id': lane_id,
                        'edge_id': data['edge_id'],
                        'route_id': data['route_id'],
                        'queue_length': data['queue_length'],
                        'waiting_time': data['waiting_time'],
                        'density': data['density'],
                        'mean_speed': data['mean_speed'],
                        'flow': data['flow'],
                        'queue_route': data['queue_route'],
                        'flow_route': data['flow_route'],
                        'ambulance': data['ambulance'],
                        'tl_id': self.lane_to_tl.get(lane_id, ''),
                        'phase_id': traci.trafficlight.getPhase(self.lane_to_tl.get(lane_id, '')) 
                                  if lane_id in self.lane_to_tl else -1,
                        'reward_components': reward_components,
                        'raw_reward': int(round(raw_reward)),
                        'left_turn': data.get('left_turn', False)
                    }
                )
            
            # Store current state and action
            self.previous_states[lane_id] = state
            self.previous_actions[lane_id] = action

    def _create_state_vector_efficient(self, lane_id, lane_data):
        """Create state vector efficiently using normalized metrics"""
        try:
            if lane_id not in lane_data:
                return np.zeros(12)
                
            data = lane_data[lane_id]
            tl_id = self.lane_to_tl.get(lane_id, "")
            
            # Normalized lane metrics
            queue_norm = min(data['queue_length'] / self.norm_bounds['queue'], 1.0)
            wait_norm = min(data['waiting_time'] / self.norm_bounds['wait'], 1.0)
            density_norm = min(data['density'] / self.norm_bounds['density'], 1.0)
            speed_norm = min(data['mean_speed'] / self.norm_bounds['speed'], 1.0)
            flow_norm = min(data['flow'] / self.norm_bounds['flow'], 1.0)
            arrival_norm = min(data['arrival_rate'] / self.norm_bounds['arrival_rate'], 1.0)

            # Route-level metrics
            route_queue_norm = min(data['queue_route'] / (self.norm_bounds['queue'] * 3), 1.0)
            route_flow_norm = min(data['flow_route'] / (self.norm_bounds['flow'] * 3), 1.0)
            
            # Traffic light context
            current_phase = 0
            phase_norm = 0.0
            phase_efficiency = 0.0

            if tl_id:
                try:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    logic = self.tl_logic_cache.get(tl_id)
                    if logic:
                        num_phases = len(logic.phases)
                        phase_norm = current_phase / max(num_phases-1, 1)
                        
                        # Calculate efficiency from cache
                        total = sum(count for (t, _), count in self.phase_utilization.items() if t == tl_id)
                        if total > 0:
                            phase_key = (tl_id, current_phase)
                            phase_efficiency = self.phase_utilization.get(phase_key, 0) / total
                except:
                    pass
            
            # Time since last green
            last_green = self.last_green_time.get(lane_id, 0)
            time_since_green = min((traci.simulation.getTime() - last_green) / 
                                 self.norm_bounds['time_since_green'], 1.0)
            
            # Create state vector
            state = np.array([
                queue_norm,
                wait_norm,
                density_norm,
                speed_norm,
                flow_norm,
                route_queue_norm,
                route_flow_norm,
                phase_norm,
                time_since_green,
                float(data['ambulance']),
                self.lane_scores.get(lane_id, 0) / 100, 
                phase_efficiency
            ])
            
            # Ensure no invalid values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            
            return state
            
        except Exception as e:
            if self.step_count % self.log_frequency == 0:
                print(f"Error creating state vector for {lane_id}: {e}")
            return np.zeros(12)

    def _calculate_reward_efficient(self, lane_id, lane_data, action_taken, current_time):
        """Calculate reward efficiently using cached data"""
        try:
            data = lane_data[lane_id]
            left_turn_factor = 1.5 if data['left_turn'] else 1.0

            # Core components
            queue_penalty = -min(data['queue_length'] * self.adaptive_params['queue_weight'] * left_turn_factor, 30)
            wait_penalty = -min(data['waiting_time'] * self.adaptive_params['wait_weight'] * left_turn_factor, 20)
            throughput_reward = min(data['flow'] * self.adaptive_params['flow_weight'], 25)
            speed_reward = min(data['mean_speed'] * self.adaptive_params['speed_weight'], 15)
            
            # Action bonus
            action_bonus = 0
            if action_taken == 0:  # Green light action
                if data['queue_length'] > 5:
                    action_bonus = min(data['queue_length'] * 0.7, 20)
                    
            # Starvation prevention
            starvation_penalty = 0
            last_green = self.last_green_time.get(lane_id, 0)
            if current_time - last_green > self.adaptive_params['starvation_threshold']:
                starvation_penalty = -min(30, (current_time - last_green - 
                                          self.adaptive_params['starvation_threshold']) * 0.5)
            
            # Priority vehicle handling
            ambulance_bonus = 25 if data['ambulance'] else 0

            # Efficiency bonus
            efficiency_bonus = 0
            if data['queue_length'] < 3 and data['mean_speed'] > 5:
                efficiency_bonus = 15
            
            # Composite reward
            total_reward = (
                queue_penalty + 
                wait_penalty + 
                throughput_reward + 
                speed_reward + 
                action_bonus + 
                starvation_penalty + 
                ambulance_bonus +
                efficiency_bonus
            )
            
            # Normalize and validate
            normalized_reward = total_reward / self.adaptive_params['reward_scale']
            normalized_reward = max(-1.0, min(1.0, normalized_reward))
            
            if np.isnan(normalized_reward) or np.isinf(normalized_reward):
                normalized_reward = 0.0
                
            # Return simplified component dict to reduce memory
            reward_components = {
                'queue': queue_penalty,
                'wait': wait_penalty,
                'flow': throughput_reward,
                'speed': speed_reward,
                'action': action_bonus,
                'starve': starvation_penalty,
                'emergency': ambulance_bonus
            }
            
            return normalized_reward, reward_components, total_reward
            
        except Exception as e:
            if self.step_count % self.log_frequency == 0:
                print(f"Error calculating reward for {lane_id}: {e}")
            return 0.0, {}, 0.0

    def end_episode(self):
        """Finalize episode and save data with optimizations"""
        try:
            # Save RL model periodically, not every episode
            if self.step_count % 5000 == 0:
                self.rl_agent.save_model(adaptive_params=self.adaptive_params)
            
            # Clear subscription data
            self.subscription_initialized = False
            
            # Reset episode-specific state
            self.previous_states.clear()
            self.previous_actions.clear()
            
            # Trim histories to prevent memory growth
            if len(self.rl_agent.training_data) > 10000:
                # Keep only the most recent 10,000 entries
                self.rl_agent.training_data = self.rl_agent.training_data[-10000:]
            
            print(f"✅ Episode {self.current_episode} completed after {self.step_count} steps")
            
        except Exception as e:
            print(f"Error ending episode: {e}")


def start_enhanced_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1, num_retries=1, retry_delay=1, mode="train"):
    controller = None
    try:
        controller = SmartTrafficController(sumocfg_path=sumocfg_path, mode=mode)
        for episode in range(episodes):
            print(f"\n{'='*50}")
            print(f"🚦 STARTING ENHANCED EPISODE {episode + 1}/{episodes}")
            print(f"{'='*50}")
            sumo_binary = "sumo-gui" if use_gui else "sumo"
            sumo_cmd = [
                os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary),
                '-c', sumocfg_path,
                '--start', '--quit-on-end',
                '--no-step-log',  # Reduce console output
                '--no-warnings'   # Reduce warnings
            ]
            traci.start(sumo_cmd)
            controller.current_episode = episode + 1
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                if max_steps and step >= max_steps:
                    print(f"Reached max steps ({max_steps}), ending episode.")
                    break
                try:
                    traci.simulationStep()
                    controller.run_step()
                    step += 1
                    if step % 1000 == 0:
                        print(f"Episode {episode + 1}: Step {step} completed")
                except Exception as e:
                    print(f"Error in simulation step {step}: {e}")
                    break
            print(f"Episode {episode + 1} completed after {step} steps")
            controller.end_episode()
            traci.close()
            if episode < episodes - 1:
                time.sleep(2)
        print(f"\n🎉 All {episodes} episodes completed!")
    except Exception as e:
        print(f"Error in enhanced simulation: {e}")
    finally:
        try:
            traci.close()
        except:
            pass
        print("Simulation resources cleaned up")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run enhanced SUMO RL traffic simulation")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of retries if connection fails')
    parser.add_argument('--retry-delay', type=int, default=1, help='Delay in seconds between retries')
    parser.add_argument('--mode', choices=['train', 'eval', 'adaptive'], default='train',
                        help='Controller mode: train (explore+learn), eval (exploit only), adaptive (exploit+learn)')
    args = parser.parse_args()
    start_enhanced_simulation(
        args.sumo, 
        args.gui, 
        args.max_steps, 
        args.episodes, 
        args.num_retries, 
        args.retry_delay,
        mode=args.mode
    )