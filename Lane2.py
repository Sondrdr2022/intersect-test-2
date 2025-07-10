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
    sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], "bin", sumo_binary), "-c", sumo_cfg_path, "--start", "--quit-on-end"]
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
            # Existing checks plus:
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
        try:
            if isinstance(state, np.ndarray):
                key = tuple(np.round(state, 2))
            elif isinstance(state, list):
                state_array = np.round(np.array(state), 2)
                key = tuple(state_array.tolist())
            else:
                key = tuple(state) if hasattr(state, '__iter__') else (state,)
            if lane_id is not None:
                return (lane_id, key)
            return key
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
        entry = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'reward': reward,
            'reward_components': lane_info.get('reward_components', {}),
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'q_value': new_q,
            'timestamp': time.time(),
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'lane_id': lane_id,
            'action_name': self._get_action_name(action),
            'raw_reward': lane_info.get('raw_reward', None)
        }
        if lane_info:
            entry.update(lane_info)
        self.training_data.append(entry)
        self._update_adaptive_parameters(reward)

    def _get_action_name(self, action):
        action_names = {
            0: "Set Green",
            1: "Next Phase",
            2: "Extend Phase",
            3: "Shorten Phase",
            4: "Balanced Phase"
        }
        return action_names.get(action, f"Unknown Action {action}")

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
                'training_data': self.training_data,
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
                    'training_count': len(self.training_data),
                    'average_reward': np.mean([x['reward'] for x in self.training_data[-100:]]) if self.training_data else 0,
                    'reward_components': [x.get('reward_components', {}) for x in self.training_data[-100:]]
                }
            }
            if adaptive_params:
                model_data['adaptive_params'] = adaptive_params.copy()
                print(f"💾 Saving adaptive parameters: {adaptive_params}")
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Model saved with {len(self.training_data)} training entries")
            self.training_data = []
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
        self.last_lane_vehicles = defaultdict(set)
        self.last_arrival_time = defaultdict(float)
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
        
        # Congested mode parameters
        self.is_congested_mode = False
        self.normal_adaptive_params = self.adaptive_params.copy()
        self.congested_adaptive_params = {
            'min_green': min(90, self.normal_adaptive_params['min_green'] * 1.5),
            'max_green': min(120, self.normal_adaptive_params['max_green'] * 1.8),
            'starvation_threshold': self.normal_adaptive_params['starvation_threshold'],
            'reward_scale': self.normal_adaptive_params['reward_scale'],
            'queue_weight': min(1.0, self.normal_adaptive_params['queue_weight'] * 1.3),
            'wait_weight': min(1.0, self.normal_adaptive_params['wait_weight'] * 1.2),
            'flow_weight': min(1.0, self.normal_adaptive_params['flow_weight'] * 1.4),
            'speed_weight': self.normal_adaptive_params['speed_weight'],
            'left_turn_priority': self.normal_adaptive_params['left_turn_priority']
        }
        
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
                print(f"Auto-detected left-turn lanes: {sorted(self.left_turn_lanes)}")
                print(f"Auto-detected right-turn lanes: {sorted(self.right_turn_lanes)}")
            except Exception as e:
                print(f"Error detecting turning lanes: {e}")
        else:
            self.left_turn_lanes = set()
            self.right_turn_lanes = set()
            self._init_left_turn_lanes()
    
    def detect_congestion(self, lane_data):
        """
        Detect congestion based on average queue length and waiting time across all lanes.
        Returns True if congestion is detected.
        """
        if not lane_data:
            return False
            
        try:
            total_queue = 0
            total_waiting = 0
            valid_lanes = 0
            
            # Calculate averages across all lanes
            for lane_id, data in lane_data.items():
                if data['queue_length'] >= 0 and data['waiting_time'] >= 0:
                    total_queue += data['queue_length']
                    total_waiting += data['waiting_time']
                    valid_lanes += 1
            
            if valid_lanes == 0:
                return False
                
            avg_queue = total_queue / valid_lanes
            avg_waiting = total_waiting / valid_lanes
            
            # Congestion thresholds
            queue_threshold = 8.0  # Average queue length threshold
            waiting_threshold = 60.0  # Average waiting time threshold (seconds)
            
            # Detect congestion if either threshold is exceeded
            is_congested = avg_queue >= queue_threshold or avg_waiting >= waiting_threshold
            
            return is_congested
            
        except Exception as e:
            print(f"Error in detect_congestion: {e}")
            return False
    
    def set_congested_mode(self, enabled: bool):
        """
        Change adaptive parameters according to congested mode state.
        Args:
            enabled: True to enable congested mode, False for normal mode
        """
        try:
            if enabled and not self.is_congested_mode:
                # Switch to congested mode
                self.is_congested_mode = True
                self.adaptive_params = self.congested_adaptive_params.copy()
                print("🚨 CONGESTED MODE ACTIVATED - Increased green times and throughput priority")
                print(f"   Min green: {self.adaptive_params['min_green']}s, Max green: {self.adaptive_params['max_green']}s")
                print(f"   Flow weight: {self.adaptive_params['flow_weight']}, Queue weight: {self.adaptive_params['queue_weight']}")
                
            elif not enabled and self.is_congested_mode:
                # Switch to normal mode
                self.is_congested_mode = False
                self.adaptive_params = self.normal_adaptive_params.copy()
                print("✅ NORMAL MODE RESTORED - Standard adaptive parameters")
                print(f"   Min green: {self.adaptive_params['min_green']}s, Max green: {self.adaptive_params['max_green']}s")
                print(f"   Flow weight: {self.adaptive_params['flow_weight']}, Queue weight: {self.adaptive_params['queue_weight']}")
                
        except Exception as e:
            print(f"Error in set_congested_mode: {e}")
    
    def _init_left_turn_lanes(self):
        """Scan and cache all left-turn lanes based on SUMO network connections."""
        try:
            self.left_turn_lanes.clear()
            lanes = traci.lane.getIDList()
            for lane_id in lanes:
                connections = traci.lane.getLinks(lane_id)
                for conn in connections:
                    # Check both possible direction indicators
                    if (len(conn) > 6 and conn[6] == 'l') or (len(conn) > 3 and conn[3] == 'l'):
                        self.left_turn_lanes.add(lane_id)
                        break
            print(f"Auto-detected left-turn lanes: {sorted(self.left_turn_lanes)}")
        except Exception as e:
            print(f"Error initializing left-turn lanes: {e}")

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
            logic = self._get_traffic_light_logic(tl_id)
            
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
                
                print(f"↩️ PROTECTED LEFT TURN: Activated for {target_lane} at {tl_id} "
                    f"(queue={queue_length}, wait={waiting_time:.1f}s, duration={duration:.1f}s)")
                
                # Update tracking variables
                self.last_green_time[target_lane] = current_time
                self.last_phase_change[tl_id] = current_time
                self.phase_utilization[(tl_id, left_turn_phase)] += 1
                
                return True  # Indicate that left turn was handled
                
        except Exception as e:
            print(f"Error in _handle_protected_left_turn: {e}")
        return False

    def _find_best_left_turn_phase(self, tl_id, left_turn_lane, lane_data):
        """
        Finds the best phase for protected left turns, considering:
        - Exclusive left-turn phases (only left turns get green)
        - Permissive phases (left turns share green with straight traffic)
        """
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return None

            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
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

    def _is_left_turn_lane(self, lane_id):
        """Check if the lane is a left-turn lane."""
        return lane_id in self.left_turn_lanes


    def _get_traffic_light_logic(self, tl_id):
        """Get traffic light logic with caching"""
        if tl_id not in self.tl_logic_cache:
            try:
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                self.tl_logic_cache[tl_id] = logic
            except Exception as e:
                print(f"Error getting logic for TL {tl_id}: {e}")
                return None
        return self.tl_logic_cache[tl_id]

    def _get_phase_count(self, tl_id):
        """Get number of phases for traffic light"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if logic:
                return len(logic.phases)
            else:
                # Fallback: try to get from current program
                program = traci.trafficlight.getProgram(tl_id)
                return len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id))
        except Exception as e:
            print(f"Error getting phase count for {tl_id}: {e}")
            return 4  # Default fallback

    def _get_phase_name(self, tl_id, phase_idx):
        """Get phase name"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if logic and phase_idx < len(logic.phases):
                return getattr(logic.phases[phase_idx], 'name', f'phase_{phase_idx}')
        except Exception as e:
            print(f"Error getting phase name for {tl_id}[{phase_idx}]: {e}")
        return f'phase_{phase_idx}'

    def run_step(self):
        try:
            self.step_count += 1
            current_time = traci.simulation.getTime()
            lane_data = self._collect_enhanced_lane_data()
            if not lane_data:
                return
            
            # Check for congestion and switch modes as needed
            is_congested = self.detect_congestion(lane_data)
            self.set_congested_mode(is_congested)
            
            lane_status = self._update_lane_status_and_score(lane_data)
            self._adjust_traffic_lights(lane_data, lane_status, current_time)
            self._process_rl_learning(lane_data, current_time)
        except Exception as e:
            print(f"Error in run_step: {e}")

    def _collect_enhanced_lane_data(self):
        """Collect comprehensive lane data with route information"""
        lane_data = {}
        try:
            lanes = traci.lane.getIDList()
            edge_queues = defaultdict(float)
            edge_flows = defaultdict(float)
            route_queues = defaultdict(float)
            route_flows = defaultdict(float)
            
            # First pass: collect basic lane data and aggregate edge/route metrics
            for lane_id in lanes:
                try:
                    lane_length = traci.lane.getLength(lane_id)
                    if lane_length <= 0:
                        continue
                        
                    # Basic lane metrics
                    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                    waiting_time = traci.lane.getWaitingTime(lane_id)
                    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                    mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                    
                    # Edge and route information
                    edge_id = traci.lane.getEdgeID(lane_id)
                    route_id = self._get_route_for_lane(lane_id)
                    
                    # Priority vehicle detection
                    ambulance_detected = self._detect_priority_vehicles(lane_id)

                    # --- LEFT/RIGHT TURN DETECTION ---
                    is_left_turn = lane_id in self.left_turn_lanes
                    is_right_turn = lane_id in self.right_turn_lanes
                    # Debug print for verification
                    
                    
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
                        'left_turn': is_left_turn,       # <-- ASSIGN HERE!
                        'right_turn': is_right_turn,     # <-- ASSIGN HERE!
                        'tl_id': self.lane_to_tl.get(lane_id, '')
                    }
                    
                    # Update edge aggregates
                    edge_queues[edge_id] += queue_length
                    edge_flows[edge_id] += vehicle_count
                    
                    # Update route aggregates if available
                    if route_id:
                        route_queues[route_id] += queue_length
                        route_flows[route_id] += vehicle_count
                        
                except Exception as e:
                    print(f"Error collecting data for lane {lane_id}: {e}")
                    continue
                    
            # Second pass: add aggregated metrics
            for lane_id, data in lane_data.items():
                data['queue_route'] = route_queues.get(data.get('route_id', ''), 0)
                data['flow_route'] = route_flows.get(data.get('route_id', ''), 0)
                
        except Exception as e:
            print(f"Error in _collect_enhanced_lane_data: {e}")
            
        return lane_data
    def _get_route_for_lane(self, lane_id):
        """Get route ID for vehicles in the lane"""
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            if vehicles:
                return traci.vehicle.getRouteID(vehicles[0])
        except:
            pass
        return ""

    def _detect_priority_vehicles(self, lane_id):
        """Detect priority vehicles (ambulances, emergency) in lane"""
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            for vid in vehicles:
                if traci.vehicle.getVehicleClass(vid) in ['emergency', 'authority']:
                    return True
        except:
            pass
        return False

    def _update_lane_status_and_score(self, lane_data):
        """Enhanced lane status update with dynamic thresholds"""
        status = {}
        try:
            for lane_id, data in lane_data.items():
                # Get normalized metrics
                queue_norm = data['queue_length'] / self.norm_bounds['queue']
                wait_norm = data['waiting_time'] / self.norm_bounds['wait']
                speed_norm = data['mean_speed'] / self.norm_bounds['speed']
                flow_norm = data['flow'] / self.norm_bounds['flow']

                # Calculate arrival rate (vehicles per second)
                arrival_rate = self._calculate_arrival_rate(lane_id)
                arrival_norm = arrival_rate / self.norm_bounds['arrival_rate']
                
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

    def _calculate_arrival_rate(self, lane_id):
        """Calculate real-time vehicle arrival rate for a lane using TraCI only (Method 1: Delta Count)."""
        try:
            now = traci.simulation.getTime()
            current_vehicles = set(traci.lane.getLastStepVehicleIDs(lane_id))
            prev_vehicles = self.last_lane_vehicles.get(lane_id, set())
            arrivals = current_vehicles - prev_vehicles
            last_time = self.last_arrival_time.get(lane_id, now-1)
            # Avoid division by zero
            delta_time = max(1e-3, now - last_time)
            arrival_rate = len(arrivals) / delta_time  # vehicles per second

            # Update memory for next call
            self.last_lane_vehicles[lane_id] = current_vehicles
            self.last_arrival_time[lane_id] = now
            return arrival_rate
        except Exception as e:
            print(f"Error calculating arrival rate for {lane_id}: {e}")
            return 0.0
    
    
    def _select_target_lane(self, tl_id, controlled_lanes, lane_data, current_time):
        """Enhanced lane selection with multi-factor prioritization"""
        candidate_lanes = []
        max_queue = 0
        max_wait = 0
        max_arrival = 0
        
        # First pass: find maximum values for normalization
        for lane in controlled_lanes:
            if lane in lane_data:
                data = lane_data[lane]
                max_queue = max(max_queue, data['queue_length'])
                max_wait = max(max_wait, data['waiting_time'])
                # Estimate arrival rate if not available
                if 'arrival_rate' not in data:
                    data['arrival_rate'] = self._calculate_arrival_rate(lane)
                max_arrival = max(max_arrival, data['arrival_rate'])
        
        # Avoid division by zero
        max_queue = max(max_queue, 1)
        max_wait = max(max_wait, 1)
        max_arrival = max(max_arrival, 0.1)
        
        for lane in controlled_lanes:
            if lane in lane_data:
                data = lane_data[lane]
                
                # Base score from lane scoring system
                score = self.lane_scores.get(lane, 0)
                
                # Normalized factors
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
                
                # Phase efficiency consideration
                current_phase = traci.trafficlight.getPhase(tl_id)
                phase_efficiency = self._get_phase_efficiency(tl_id, current_phase)
                
                # Total priority score
                total_score = (
                    score + 
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
    
    def _get_phase_efficiency(self, tl_id, phase_index):
        """Calculate phase efficiency based on historical utilization"""
        try:
            total_utilization = sum(
                count for (tl, phase), count in self.phase_utilization.items()
                if tl == tl_id
            )
            
            if total_utilization == 0:
                return 1.0  # Default efficiency
            
            phase_key = (tl_id, phase_index)
            phase_count = self.phase_utilization.get(phase_key, 0)
            efficiency = phase_count / total_utilization
            
            # Normalize to 0-1 range
            return min(1.0, max(0.1, efficiency))
        except:
            return 1.0

    def _adjust_traffic_lights(self, lane_data, lane_status, current_time):
        """Enhanced traffic light adjustment with priority handling"""
        try:
            tl_ids = traci.trafficlight.getIDList()
            
            for tl_id in tl_ids:
                try:
                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    
                    # Update lane to traffic light mapping
                    for lane in controlled_lanes:
                        self.lane_to_tl[lane] = tl_id
                    
                    # Check for priority conditions
                    priority_handled = self._handle_priority_conditions(tl_id, controlled_lanes, lane_data, current_time)
                    
                    # Only proceed with normal control if no priority was handled
                    if not priority_handled:
                        self._perform_normal_control(tl_id, controlled_lanes, lane_data, current_time)
                        
                except Exception as e:
                    print(f"Error adjusting traffic light {tl_id}: {e}")
                    
        except Exception as e:
            print(f"Error in _adjust_traffic_lights: {e}")

    def _handle_priority_conditions(self, tl_id, controlled_lanes, lane_data, current_time):
        """Handle priority conditions (ambulances, left turns)"""
        # Check for ambulances first (highest priority)
        ambulance_lanes = [lane for lane in controlled_lanes 
                        if lane in lane_data and lane_data[lane]['ambulance']]
        
        if ambulance_lanes:
            self._handle_ambulance_priority(tl_id, ambulance_lanes, current_time)
            return True
            
        # Check for left turns with significant queues
        left_turn_lanes = [lane for lane in controlled_lanes 
                        if lane in lane_data and lane_data[lane]['left_turn'] 
                        and (lane_data[lane]['queue_length'] > 3 or 
                            lane_data[lane]['waiting_time'] > 10)]
        
        if left_turn_lanes:
            return self._handle_protected_left_turn(tl_id, left_turn_lanes, lane_data, current_time)
            
        return False  # No left-turn priority logic

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
                    print(f"🚑 AMBULANCE PRIORITY: Green for {ambulance_lane} at {tl_id}")
            else:
                # Check if priority should be released
                if current_time - self.ambulance_start_time[tl_id] > 30:  # 30s timeout
                    self.ambulance_active[tl_id] = False
                    print(f"🚑 AMBULANCE CLEARED: Released priority at {tl_id}")
                    
        except Exception as e:
            print(f"Error in _handle_ambulance_priority: {e}")

    # --- REMOVED: All left-turn related control, reward, and phase-finding logic ---

    def _perform_normal_control(self, tl_id, controlled_lanes, lane_data, current_time):
        """Perform normal RL-based traffic control"""
        try:
            # Select target lane based on multiple factors
            target_lane = self._select_target_lane(tl_id, controlled_lanes, lane_data, current_time)
            
            if not target_lane:
                return
                
            # Get state and RL action
            state = self._create_state_vector(target_lane, lane_data)
            if not self.rl_agent.is_valid_state(state):
                return
                
            action = self.rl_agent.get_action(state, lane_id=target_lane)
            
            # Execute action with minimum phase duration enforcement
            if current_time - self.last_phase_change.get(tl_id, 0) >= 5:  # 5s minimum
                self._execute_control_action(tl_id, target_lane, action, lane_data, current_time)
                
        except Exception as e:
            print(f"Error in _perform_normal_control: {e}")

    def _execute_control_action(self, tl_id, target_lane, action, lane_data, current_time):
        """Execute the selected control action"""
        try:
            # Ensure traffic light exists
            if tl_id not in traci.trafficlight.getIDList():
                print(f"⚠️ Traffic light {tl_id} not found")
                return
            
            # Get current phase and find target phase for the lane
            current_phase = traci.trafficlight.getPhase(tl_id)
            target_phase = self._find_phase_for_lane(tl_id, target_lane)
            
            if target_phase is None:
                target_phase = current_phase  # Fallback to current phase
                
            if action == 0:  # Set green for target lane
                if current_phase != target_phase:
                    green_time = self._calculate_dynamic_green(lane_data[target_lane])
                    traci.trafficlight.setPhase(tl_id, target_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, green_time)
                    self.last_phase_change[tl_id] = current_time
                    self.last_green_time[target_lane] = current_time
                    print(f"🟢 RL ACTION: Green for {target_lane} (duration={green_time}s)")
                    
            elif action == 1:  # Next phase
                phase_count = self._get_phase_count(tl_id)
                next_phase = (current_phase + 1) % phase_count
                traci.trafficlight.setPhase(tl_id, next_phase)
                traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                self.last_phase_change[tl_id] = current_time
                print(f"⏭️ RL ACTION: Next phase ({next_phase})")
                
            elif action == 2:  # Extend current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    extension = min(15, self.adaptive_params['max_green'] - remaining)
                    if extension > 0:
                        traci.trafficlight.setPhaseDuration(tl_id, remaining + extension)
                        print(f"⏱️ RL ACTION: Extended phase by {extension}s")
                except Exception as e:
                    print(f"Could not extend phase: {e}")
                    
            elif action == 3:  # Shorten current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    if remaining > self.adaptive_params['min_green'] + 5:
                        reduction = min(5, remaining - self.adaptive_params['min_green'])
                        traci.trafficlight.setPhaseDuration(tl_id, remaining - reduction)
                        print(f"⏳ RL ACTION: Shortened phase by {reduction}s")
                except Exception as e:
                    print(f"Could not shorten phase: {e}")
                    
            elif action == 4:  # Balanced phase switch
                balanced_phase = self._get_balanced_phase(tl_id, lane_data)
                if balanced_phase != current_phase:
                    traci.trafficlight.setPhase(tl_id, balanced_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                    self.last_phase_change[tl_id] = current_time
                    print(f"⚖️ RL ACTION: Balanced phase ({balanced_phase})")
                    
            # Update phase utilization stats
            new_phase = traci.trafficlight.getPhase(tl_id)
            self.phase_utilization[(tl_id, new_phase)] += 1
            
        except Exception as e:
            print(f"Error in _execute_control_action: {e}")

    def _get_balanced_phase(self, tl_id, lane_data):
        """Get phase that balances traffic flow"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return 0
                
            best_phase = 0
            best_score = -1
            
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for phase_idx in range(len(logic.phases)):
                phase_score = 0
                
                for lane_idx, lane in enumerate(controlled_lanes):
                    if lane in lane_data:
                        # Only consider lanes that get green in this phase
                        state = logic.phases[phase_idx].state
                        if lane_idx < len(state) and state[lane_idx].upper() == 'G':
                            data = lane_data[lane]
                            phase_score += (data['queue_length'] * 0.5 + 
                                          data['waiting_time'] * 0.1 +
                                          (1 - min(data['mean_speed'] / self.norm_bounds['speed'], 1)) * 10)
                
                if phase_score > best_score:
                    best_score = phase_score
                    best_phase = phase_idx
                    
            return best_phase
        except Exception as e:
            print(f"Error in _get_balanced_phase: {e}")
            return 0

    def _calculate_dynamic_green(self, lane_data):
        """Calculate dynamic green time based on lane conditions"""
        base_time = self.adaptive_params['min_green']
        queue_factor = min(lane_data['queue_length'] * 0.7, 15)
        density_factor = min(lane_data['density'] * 5, 10)
        emergency_bonus = 10 if lane_data['ambulance'] else 0
        
        total_time = (base_time + queue_factor + density_factor + 
                     emergency_bonus)
        
        return min(max(total_time, self.adaptive_params['min_green']), 
                 self.adaptive_params['max_green'])

    def _find_phase_for_lane(self, tl_id, target_lane):
        """Find phase that gives green to target lane"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return 0
                
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for phase_idx, phase in enumerate(logic.phases):
                state = phase.state
                for lane_idx, lane in enumerate(controlled_lanes):
                    if lane == target_lane and lane_idx < len(state) and state[lane_idx].upper() == 'G':
                        return phase_idx
        except Exception as e:
            print(f"Error finding phase for lane {target_lane}: {e}")
        return 0

    def _process_rl_learning(self, lane_data, current_time):
        """Process RL learning for each lane"""
        for lane_id, data in lane_data.items():
            state = self._create_state_vector(lane_id, lane_data)
            
            if not self.rl_agent.is_valid_state(state):
                continue
                
            action = self.rl_agent.get_action(state, lane_id=lane_id)
            
            # Calculate reward if we have previous state
            reward = 0
            reward_components = {}
            if lane_id in self.previous_states and lane_id in self.previous_actions:
                reward, reward_components, raw_reward = self._calculate_reward(lane_id, lane_data, 
                                                    self.previous_actions[lane_id], current_time)
                
                # Log reward components
                print(f"🏆 Lane {lane_id} reward components:")
                for comp, value in reward_components.items():
                    print(f"  - {comp}: {value:.2f}")
                print(f"  Total reward: {reward:.2f}")
                
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
                        'phase_id': traci.trafficlight.getPhase(self.lane_to_tl.get(lane_id, '')) if lane_id in self.lane_to_tl else -1,
                        'epsilon': self.rl_agent.epsilon,
                        'learning_rate': self.rl_agent.learning_rate,
                        'adaptive_params': self.adaptive_params.copy(),
                        'simulation_time': current_time,
                        'reward_components': reward_components,
                        'raw_reward': int(round(raw_reward)),
                        'left_turn': data.get('left_turn', False)
                        }
                )
            
            # Store current state and action
            self.previous_states[lane_id] = state
            self.previous_actions[lane_id] = action
            

    def _create_state_vector(self, lane_id, lane_data):
        """Create comprehensive state vector"""
        try:
            data = lane_data[lane_id]
            tl_id = self.lane_to_tl.get(lane_id, "")
            
            # Normalized lane metrics
            queue_norm = min(data['queue_length'] / self.norm_bounds['queue'], 1.0)
            wait_norm = min(data['waiting_time'] / self.norm_bounds['wait'], 1.0)
            density_norm = min(data['density'] / self.norm_bounds['density'], 1.0)
            speed_norm = min(data['mean_speed'] / self.norm_bounds['speed'], 1.0)
            flow_norm = min(data['flow'] / self.norm_bounds['flow'], 1.0)
            
            if 'arrival_rate' not in data:
                data['arrival_rate'] = self._calculate_arrival_rate(lane_id)
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
                    num_phases = self._get_phase_count(tl_id)
                    phase_norm = current_phase / max(num_phases-1, 1)
                    phase_efficiency = self._get_phase_efficiency(tl_id, current_phase)

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
                self.lane_scores.get(lane_id, 0) / 100, phase_efficiency  # Normalized lane score
            ])
            
            # Ensure no invalid values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            
            return state
            
        except Exception as e:
            print(f"Error creating state vector for {lane_id}: {e}")
            return np.zeros(13)  # Reduced state size

    def _calculate_reward(self, lane_id, lane_data, action_taken, current_time):
        """Calculate comprehensive reward signal with detailed components"""
        try:
            data = lane_data[lane_id]
            left_turn_factor = 1.5 if data['left_turn'] else 1.0

            # Core components
            queue_penalty = -min(data['queue_length'] * self.adaptive_params['queue_weight'] * left_turn_factor, 30)
            wait_penalty = -min(data['waiting_time'] * self.adaptive_params['wait_weight'] * left_turn_factor, 20)
            throughput_reward = min(data['flow'] * self.adaptive_params['flow_weight'], 25)
            speed_reward = min(data['mean_speed'] * self.adaptive_params['speed_weight'], 15)
            # Bonus for handling left turns effectively
            left_turn_bonus = 0
            if data['left_turn'] and action_taken == 0:  # Green light action
                if data['queue_length'] < 2:  # Successfully cleared left turn queue
                    left_turn_bonus = 15
            # Action effectiveness
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
                
            # Return both the reward and its components
            reward_components = {
                'queue_penalty': queue_penalty,
                'wait_penalty': wait_penalty,
                'throughput_reward': throughput_reward,
                'speed_reward': speed_reward,
                'action_bonus': action_bonus,
                'starvation_penalty': starvation_penalty,
                'ambulance_bonus': ambulance_bonus,
                'total_raw': total_reward,
                'normalized': normalized_reward
            }
            
            return normalized_reward, reward_components, total_reward
            
        except Exception as e:
            print(f"Error calculating reward for {lane_id}: {e}")
            return 0.0, {}, 0.0

    def end_episode(self):
        """Finalize episode and save data"""
        try:
            # Save RL model with updated adaptive parameters
            if self.step_count % 1000 == 0:
                self.rl_agent.save_model(adaptive_params=self.adaptive_params)
            
            # Reset episode-specific state
            self.previous_states.clear()
            self.previous_actions.clear()
            
            print(f"✅ Episode {self.current_episode} completed")
            
        except Exception as e:
            print(f"Error ending episode: {e}")

    def _update_adaptive_parameters(self, performance_stats):
        """Dynamically adjust control parameters based on performance"""
        try:
            avg_reward = performance_stats.get('avg_reward', 0)
            
            # Adjust green time parameters based on queue performance
            if avg_reward > 0.6:  # High queues
                self.adaptive_params['min_green'] = min(15, self.adaptive_params['min_green'] + 1)
                self.adaptive_params['max_green'] = min(90, self.adaptive_params['max_green'] + 5)
            elif avg_reward < 0.3:  # Low queues
                self.adaptive_params['min_green'] = max(5, self.adaptive_params['min_green'] - 1)
                self.adaptive_params['max_green'] = max(30, self.adaptive_params['max_green'] - 5)
                
            print("🔄 Updated adaptive parameters:", self.adaptive_params)
            
        except Exception as e:
            print(f"Error updating adaptive parameters: {e}")

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
                '--start', '--quit-on-end'
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
                    if step % 200 == 0:
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