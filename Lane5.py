import os
import sys
import traci
import numpy as np
import time
import datetime
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import traceback

# SUMO_HOME setup
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

def subscribe_lanes(lane_id_list):
    for lane_id in lane_id_list:
        traci.lane.subscribe(lane_id, [
            traci.constants.LAST_STEP_VEHICLE_NUMBER,
            traci.constants.LAST_STEP_MEAN_SPEED,
            traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
        ])

def detect_turning_lanes_with_traci():
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
    return left_turn_lanes, right_turn_lanes

class EnhancedQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 q_table_file="enhanced_q_table.pkl", mode="train", adaptive_params=None):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.training_data = []
        self.q_table_file = q_table_file
        self._loaded_training_count = 0
        self.reward_components = []
        self.reward_history = []
        self.learning_rate_decay = 0.999
        self.min_learning_rate = 0.01
        self.consecutive_no_improvement = 0
        self.max_no_improvement = 100
        self.mode = mode

        # ADD THIS BLOCK:
        self.adaptive_params = adaptive_params or {
            'min_green': 30,
            'max_green': 80,
            'starvation_threshold': 40,
            'reward_scale': 40,
            'queue_weight': 0.6,
            'wait_weight': 0.3,
            'flow_weight': 0.5,
            'speed_weight': 0.2,
            'left_turn_priority': 1.2
        }

        if self.mode == "eval":
            self.epsilon = 0.0
        elif self.mode == "adaptive":
            self.epsilon = 0.01
            
        print(f"AGENT INIT: mode={self.mode}, epsilon={self.epsilon}")

    def is_valid_state(self, state):
        if not isinstance(state, (list, np.ndarray)):
            return False
        state_array = np.array(state)
        if np.isnan(state_array).any() or np.isinf(state_array).any():
            return False
        if (np.abs(state_array) > 100).any():
            return False
        if state_array.size != self.state_size:
            return False
        # Prevent all-zeros (likely dummy state)
        if np.all(state_array == 0):
            return False
        return True

    def get_action(self, state, tl_id=None, action_size=None):
        state_key = self._state_to_key(state, tl_id)
        if action_size is None:
            action_size = self.action_size
        state_key = self._state_to_key(state, tl_id)
        if state_key not in self.q_table or len(self.q_table[state_key]) < action_size:
            self.q_table[state_key] = np.zeros(action_size)
        if self.mode == "train":
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, action_size)
            else:
                return np.argmax(self.q_table[state_key][:action_size])
        else:
            return np.argmax(self.q_table[state_key][:action_size])

    def _state_to_key(self, state, tl_id=None):
        try:
            if isinstance(state, np.ndarray):
                key = tuple(np.round(state, 2))
            elif isinstance(state, list):
                state_array = np.round(np.array(state), 2)
                key = tuple(state_array.tolist())
            else:
                key = tuple(state) if hasattr(state, '__iter__') else (state,)
            if tl_id is not None:
                return (tl_id, key)
            return key
        except Exception:
            return (tl_id, (0,)) if tl_id is not None else (0,)
        
    def update_q_table(self, state, action, reward, next_state, tl_id=None, extra_info=None, action_size=None):
        if self.mode == "eval":
            return
        if not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return
        if reward is None or np.isnan(reward) or np.isinf(reward):
            return
        
        if action_size is None:
            action_size = self.action_size
        state_key = self._state_to_key(state, tl_id)
        next_state_key = self._state_to_key(next_state, tl_id)
        state_key = self._state_to_key(state, tl_id)
        next_state_key = self._state_to_key(next_state, tl_id)
        if state_key not in self.q_table or len(self.q_table[state_key]) < action_size:
            self.q_table[state_key] = np.zeros(action_size)
        if next_state_key not in self.q_table or len(self.q_table[next_state_key]) < action_size:
            self.q_table[next_state_key] = np.zeros(action_size)
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        if np.isnan(new_q) or np.isinf(new_q):
            new_q = current_q
        self.q_table[state_key][action] = new_q

        # Always ensure reward is present and not overwritten by extra_info
        entry = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'q_value': new_q,
            'timestamp': time.time(),
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'tl_id': tl_id,
            'adaptive_params': self.adaptive_params.copy()
        }
        if extra_info:
            # Prevent extra_info from overwriting a valid reward with None/missing
            entry.update({k: v for k, v in extra_info.items() if k != "reward"})

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
        if reward < -20:
            self.adaptive_params['min_green'] = min(self.adaptive_params['min_green'] + 1, self.adaptive_params['max_green'])
        elif reward > -5:
            self.adaptive_params['min_green'] = max(self.adaptive_params['min_green'] - 1, 5)
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
        print("Attempting to load Q-table from:", filepath)
        print("Absolute path:", os.path.abspath(filepath))
        print("File exists?", os.path.exists(filepath))
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                self.q_table = {}
                loaded_q_table = model_data.get('q_table', {})
                for k, v in loaded_q_table.items():
                    self.q_table[k] = np.array(v)
                self._loaded_training_count = len(model_data.get('training_data', []))
                params = model_data.get('params', {})
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.discount_factor = params.get('discount_factor', self.discount_factor)
                #self.epsilon = params.get('epsilon', self.epsilon)
                adaptive_params = model_data.get('adaptive_params', {})
                print(f"After model load, epsilon={self.epsilon}")  # <--- Add this line here

                print(f"Loaded Q-table with {len(self.q_table)} states from {filepath}")
                if adaptive_params:
                    print(f"📋 Loaded adaptive parameters from previous run")
                return True, adaptive_params
            else:
                print("No existing Q-table, starting fresh")
                return False, {}
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
                'q_table': {k: v.tolist() for k, v in self.q_table.items()},
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
                    'average_reward': np.mean([x.get('reward', 0) for x in self.training_data[-100:]]) if self.training_data else 0,
                    'reward_components': [x.get('reward_components', {}) for x in self.training_data[-100:]]
                }
            }
            if adaptive_params:
                model_data['adaptive_params'] = adaptive_params.copy()
                print(f"Saving adaptive parameters: {adaptive_params}")
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Model saved with {len(self.training_data)} training entries")
            self.training_data = []
        except Exception as e:
            print(f"Error saving model: {e}")

# Universal Smart Traffic Controller using full logic from SmartTrafficController
class UniversalSmartTrafficController:
    DILEMMA_ZONE_THRESHOLD = 12.0  # meters

    def __init__(self, sumocfg_path=None, mode="train"):
        self.mode = mode
        self.tl_action_sizes = {}
        self.step_count = 0

        self.left_turn_lanes = set()
        self.right_turn_lanes = set()
        self.ambulance_active = defaultdict(bool)
        self.ambulance_start_time = defaultdict(float)
        self.lane_id_to_idx = {}
        self.idx_to_lane_id = {}
        self.intersection_data = {}  # Store intersection data
        self.tl_logic_cache = {}     # Cache traffic light logic
        self.lane_to_tl = {}         # Lane to traffic light mapping
        self.phase_utilization = defaultdict(int)  # Phase tracking
        self.lane_id_list = []       # You must initialize this in your code elsewhere
        self.last_green_time = np.zeros(len(self.lane_id_list))  # Lane tracking
        self.previous_states = {}
        self.previous_actions = {}
        self.step_count = 0
          # Ensure reward history ex
        self.last_phase_change = {}
        self.intersection_data = {}  # Add this to store intersection data
        self.tl_logic_cache = {}  # Add this for traffic light logic caching
        self.lane_to_tl = {}  # Add this for lane to traffic light mapping
        self.phase_utilization = defaultdict(int)  # Add this for phase tracking
        self.current_episode = 0
        

        # Adaptive params for speed and fairness
        self.adaptive_params = {
            'min_green': 30,
            'max_green': 80,
            'starvation_threshold': 40,
            'reward_scale': 40,
            'queue_weight': 0.6,
            'wait_weight': 0.3,
            'flow_weight': 0.5,
            'speed_weight': 0.2,
            'left_turn_priority': 1.2
        }

        self.rl_agent = EnhancedQLearningAgent(state_size=10, action_size=8, mode=mode)
        self.rl_agent.load_model()
        
        self.previous_states = {}
        self.previous_actions = {}
        self.left_phase_counter = defaultdict(lambda: 0)
        self.max_consecutive_left = 1

    def initialize(self):
        # 1. Get the lane list from SUMO
        self.lane_id_list = traci.lane.getIDList()
        self.tl_action_sizes = {
            tl_id: len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
            for tl_id in traci.trafficlight.getIDList()
        }

        # 2. Set up lane index mappings
        self.lane_id_to_idx = {lid: i for i, lid in enumerate(self.lane_id_list)}
        self.idx_to_lane_id = {i: lid for i, lid in enumerate(self.lane_id_list)}

        # 3. Initialize per-lane arrays
        self.last_green_time = np.zeros(len(self.lane_id_list))

        # 4. Subscribe to lane data (do this after you have the correct lane list)
        subscribe_lanes(self.lane_id_list)

        # 5. Detect left/right turn lanes if needed
        self.left_turn_lanes, self.right_turn_lanes = detect_turning_lanes_with_traci()
        print(f"Auto-detected left-turn lanes: {sorted(self.left_turn_lanes)}")
        print(f"Auto-detected right-turn lanes: {sorted(self.right_turn_lanes)}")

    # Remove any logic involving target_lane/current_time here!
    def _init_left_turn_lanes(self):
        try:
            self.left_turn_lanes.clear()
            lanes = traci.lane.getIDList()
            for lane_id in lanes:
                connections = traci.lane.getLinks(lane_id)
                for conn in connections:
                    if (len(conn) > 6 and conn[6] == 'l') or (len(conn) > 3 and conn[3] == 'l'):
                        self.left_turn_lanes.add(lane_id)
                        break
            print(f"Auto-detected left-turn lanes: {sorted(self.left_turn_lanes)}")
        except Exception as e:
            print(f"Error initializing left-turn lanes: {e}")
    
    def _is_in_dilemma_zone(self, tl_id, controlled_lanes, lane_data):
        """Check if any vehicle is in the dilemma zone for lanes about to switch from green to red."""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return False
            current_phase = traci.trafficlight.getPhase(tl_id)
            state = logic.phases[current_phase].state
            for lane_idx, lane in enumerate(controlled_lanes):
                # Only check lanes with green signal
                if lane_idx < len(state) and state[lane_idx].upper() == 'G':
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                    for vid in vehicle_ids:
                        lane_pos = traci.vehicle.getLanePosition(vid)
                        lane_length = traci.lane.getLength(lane)
                        distance_to_stop = lane_length - lane_pos
                        if 0 < distance_to_stop <= self.DILEMMA_ZONE_THRESHOLD:
                            return True
            return False
        except Exception as e:
            print(f"Error in _is_in_dilemma_zone: {e}")
            return False

    def _find_starved_lane(self, controlled_lanes, current_time):
        """Find lane that is starved past the starvation threshold."""
        for lane in controlled_lanes:
            idx = self.lane_id_to_idx.get(lane, None)
            if idx is not None and current_time - self.last_green_time[idx] > self.adaptive_params['starvation_threshold']:
                return lane
        return None
    
    def _handle_protected_left_turn(self, tl_id, controlled_lanes, lane_data, current_time):
        """
        Scan all left-turn lanes at the intersection. 
        If any left-turn lane has high queue/wait, serve it with a protected left phase and log both needed and triggered events.
        """
        try:
            # Identify left-turn lanes needing service
            left_turn_lanes = [
                lane for lane in controlled_lanes
                if lane_data.get(lane, {}).get('left_turn', False)
                and (
                    lane_data[lane]['queue_length'] > 3 or
                    lane_data[lane]['waiting_time'] > 10
                )
            ]
            if not left_turn_lanes:
                return False  # No protected left needed

            # Pick lane with highest queue/wait time
            left_turn_lanes.sort(
                key=lambda x: (lane_data[x]['queue_length'], lane_data[x]['waiting_time']),
                reverse=True
            )
            target_lane = left_turn_lanes[0]
            current_phase = traci.trafficlight.getPhase(tl_id)
            logic = self._get_traffic_light_logic(tl_id)
            left_turn_phase = self._find_best_left_turn_phase(tl_id, target_lane, lane_data)

            # Log that protected left was needed
            self.rl_agent.training_data.append({
                'event': 'protected_left_needed',
                'lane_id': target_lane,
                'tl_id': tl_id,
                'phase': left_turn_phase,
                'simulation_time': current_time,
                'queue_length': lane_data[target_lane]['queue_length'],
                'waiting_time': lane_data[target_lane]['waiting_time'],
            })

            if left_turn_phase is not None and current_phase != left_turn_phase:
                # Compute adaptive green time
                queue_length = lane_data[target_lane]['queue_length']
                waiting_time = lane_data[target_lane]['waiting_time']
                base_duration = 5 + min(queue_length * 0.5, 10) + min(waiting_time * 0.1, 5)
                duration = min(
                    max(base_duration, self.adaptive_params['min_green']),
                    self.adaptive_params['max_green']
                )
                traci.trafficlight.setPhase(tl_id, left_turn_phase)
                traci.trafficlight.setPhaseDuration(tl_id, duration)
                # Log that protected left was triggered
                self.rl_agent.training_data.append({
                    'event': 'protected_left_triggered',
                    'lane_id': target_lane,
                    'tl_id': tl_id,
                    'phase': left_turn_phase,
                    'simulation_time': current_time,
                    'queue_length': lane_data[target_lane]['queue_length'],
                    'waiting_time': lane_data[target_lane]['waiting_time'],
                    'duration': duration,
                })
                # Update tracking
                lane_idx = self.lane_id_to_idx[target_lane]
                self.last_green_time[lane_idx] = current_time
                self.last_phase_change[tl_id] = current_time
                self.phase_utilization[(tl_id, left_turn_phase)] = self.phase_utilization.get((tl_id, left_turn_phase), 0) + 1
                return True

            return False

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
            self.intersection_data = {}

            # 1. Collect lane data once for the whole network
            lane_data = self._collect_enhanced_lane_data()

            for tl_id in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)

                # 2. Emergency priority (ambulance) check
                if self._handle_ambulance_priority(tl_id, controlled_lanes, lane_data, current_time):
                    continue

                # 3. Protected left turn check
                if self._handle_protected_left_turn(tl_id, controlled_lanes, lane_data, current_time):
                    continue

                # 4. Gather intersection-level state
                queues = [lane_data[l]['queue_length'] for l in controlled_lanes if l in lane_data]
                waits = [lane_data[l]['waiting_time'] for l in controlled_lanes if l in lane_data]
                speeds = [lane_data[l]['mean_speed'] for l in controlled_lanes if l in lane_data]

                left_q = sum(lane_data[l]['queue_length'] for l in controlled_lanes if l in self.left_turn_lanes and l in lane_data)
                right_q = sum(lane_data[l]['queue_length'] for l in controlled_lanes if l in self.right_turn_lanes and l in lane_data)

                n_phases = self.tl_action_sizes[tl_id]
                current_phase = traci.trafficlight.getPhase(tl_id)
                self.intersection_data[tl_id] = {
                    'queues': queues,
                    'waits': waits,
                    'speeds': speeds,
                    'left_q': left_q,
                    'right_q': right_q,
                    'n_phases': n_phases,
                    'current_phase': current_phase
                }

                # 5. Hard starvation prevention: force green if any lane is starved
                starved_lane = None
                max_starve = 0
                for lane in controlled_lanes:
                    idx = self.lane_id_to_idx.get(lane, None)
                    if idx is not None:
                        starve = current_time - self.last_green_time[idx]
                        if starve > self.adaptive_params['starvation_threshold'] and starve > max_starve:
                            starved_lane = lane
                            max_starve = starve
                if starved_lane:
                    starved_phase = self._find_phase_for_lane(tl_id, starved_lane)
                    if current_phase != starved_phase:
                        # Only switch if not in dilemma zone
                        if not self._is_in_dilemma_zone(tl_id, controlled_lanes, lane_data):
                            traci.trafficlight.setPhase(tl_id, starved_phase)
                            traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                            self.last_phase_change[tl_id] = current_time
                    continue  # skip RL for this intersection this step

                # 6. RL per-intersection phase selection
                state = self._create_intersection_state_vector(tl_id, self.intersection_data)
                action = self.rl_agent.get_action(state, tl_id, action_size=n_phases)
                last_change = self.last_phase_change.get(tl_id, -9999)
                time_since_last_change = current_time - last_change

                # 7. Dilemma zone and yellow phase enforcement
                if time_since_last_change >= self.adaptive_params['min_green'] and action != current_phase:
                    if not self._is_in_dilemma_zone(tl_id, controlled_lanes, lane_data):
                        logic = self._get_traffic_light_logic(tl_id)
                        yellow_inserted = False
                        if logic:
                            next_phase_state = logic.phases[action].state
                            for i, (cur_sig, next_sig) in enumerate(zip(logic.phases[current_phase].state, next_phase_state)):
                                if cur_sig.upper() == 'G' and next_sig.upper() == 'R':
                                    # Find yellow phase (if present) for this transition
                                    yellow_phase = None
                                    for idx, phase in enumerate(logic.phases):
                                        if (phase.state[i].upper() == 'Y' and
                                            logic.phases[current_phase].state[i].upper() == 'G' and
                                            next_phase_state[i].upper() == 'R'):
                                            yellow_phase = idx
                                            break
                                    if yellow_phase is not None and yellow_phase != current_phase:
                                        traci.trafficlight.setPhase(tl_id, yellow_phase)
                                        traci.trafficlight.setPhaseDuration(tl_id, 3)  # 3s yellow
                                        self.last_phase_change[tl_id] = current_time
                                        yellow_inserted = True
                                        break
                        if not yellow_inserted:
                            traci.trafficlight.setPhase(tl_id, action)
                            traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                            self.last_phase_change[tl_id] = current_time

            # 8. RL learning step
            self._process_rl_learning(self.intersection_data, current_time)

        except Exception as e:
            print(f"Error in run_step: {e}")
    def _collect_enhanced_lane_data(self):
        """Collect comprehensive lane data for complex networks (per lane, not per intersection).
        Returns a dictionary: lane_id -> lane_info_dict
        """
        lane_data = {}
        try:
            for lane_id in self.lane_id_list:
                try:
                    results = traci.lane.getSubscriptionResults(lane_id)
                    if results is None:
                        continue

                    vehicle_count = results.get(traci.constants.LAST_STEP_VEHICLE_NUMBER, 0)
                    mean_speed = results.get(traci.constants.LAST_STEP_MEAN_SPEED, 0)
                    queue_length = results.get(traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                    waiting_time = results.get(traci.constants.VAR_ACCUMULATED_WAITING_TIME, 0)
                    lane_length = traci.lane.getLength(lane_id) or 1.0  # avoid div by zero
                    density = vehicle_count / lane_length
                    speed = max(mean_speed, 0.0)

                    edge_id = traci.lane.getEdgeID(lane_id)
                    route_id = self._get_route_for_lane(lane_id)

                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    class_counts = self._count_vehicle_classes(vehicle_ids)
                    ambulance_detected = class_counts.get('emergency', 0) > 0

                    is_left_turn = lane_id in self.left_turn_lanes
                    is_right_turn = lane_id in self.right_turn_lanes

                    lane_data[lane_id] = {
                        'queue_length': queue_length,
                        'waiting_time': waiting_time,
                        'density': density,
                        'mean_speed': speed,
                        'flow': vehicle_count,
                        'lane_id': lane_id,
                        'edge_id': edge_id,
                        'route_id': route_id,
                        'ambulance': ambulance_detected,
                        'vehicle_classes': class_counts,
                        'left_turn': is_left_turn,
                        'right_turn': is_right_turn,
                        'tl_id': self.lane_to_tl.get(lane_id, '')
                    }

                except Exception as e:
                    print(f"⚠️ Error collecting data for lane {lane_id}: {e}")
                    continue

        except Exception as e:
            print(f"❌ Error in _collect_enhanced_lane_data: {e}")

        return lane_data

    def _count_vehicle_classes(self, vehicle_ids):
        """Count different types of vehicles (passenger, truck, emergency, etc.)"""
        counts = defaultdict(int)
        try:
            for vid in vehicle_ids:
                try:
                    vclass = traci.vehicle.getVehicleClass(vid)
                    counts[vclass] += 1
                except:
                    continue
        except:
            pass
        return counts

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
        status = {}
        try:
            for lane_id, data in lane_data.items():
                idx = self.lane_id_to_idx[lane_id]
                queue_norm = data['queue_length'] / self.norm_bounds['queue']
                wait_norm = data['waiting_time'] / self.norm_bounds['wait']
                speed_norm = data['mean_speed'] / self.norm_bounds['speed']
                flow_norm = data['flow'] / self.norm_bounds['flow']
                arrival_rate = self._calculate_arrival_rate(lane_id)
                arrival_norm = arrival_rate / self.norm_bounds['arrival_rate']
                composite_score = (
                    self.adaptive_params['queue_weight'] * queue_norm +
                    self.adaptive_params['wait_weight'] * wait_norm +
                    (1 - min(speed_norm, 1.0)) +
                    (1 - min(flow_norm, 1.0)) +
                    arrival_norm * 0.5
                )
                if composite_score > 0.8:
                    status[lane_id] = "BAD"
                    delta = -max(2, min(8, int(composite_score * 8)))
                elif composite_score < 0.3:
                    status[lane_id] = "GOOD"
                    delta = max(2, min(8, int((1 - composite_score) * 8)))
                else:
                    status[lane_id] = "NORMAL"
                    delta = 0
                if self.lane_states[idx] == status[lane_id]:
                    self.consecutive_states[idx] += 1
                    momentum_factor = min(3.0, 1.0 + self.consecutive_states[idx] * 0.1)
                    delta *= momentum_factor
                else:
                    self.lane_states[idx] = status[lane_id]
                    self.consecutive_states[idx] = 1
                self.lane_scores[idx] += delta
                if status[lane_id] == "NORMAL":
                    decay_factor = 1.5 if composite_score < 0.4 else 1.0
                    if self.lane_scores[idx] > 0:
                        self.lane_scores[idx] = max(0, self.lane_scores[idx] - decay_factor)
                    elif self.lane_scores[idx] < 0:
                        self.lane_scores[idx] = min(0, self.lane_scores[idx] + decay_factor)
                self.lane_scores[idx] = max(-50, min(50, self.lane_scores[idx]))
        except Exception as e:
            print(f"Error in _update_lane_status_and_score: {e}")
        return status
    
    def _calculate_arrival_rate(self, lane_id):
        try:
            idx = self.lane_id_to_idx[lane_id]
            now = traci.simulation.getTime()
            current_vehicles = set(traci.lane.getLastStepVehicleIDs(lane_id))
            prev_vehicles = self.last_lane_vehicles[idx]
            arrivals = current_vehicles - prev_vehicles
            last_time = self.last_arrival_time[idx]
            delta_time = max(1e-3, now - last_time)
            arrival_rate = len(arrivals) / delta_time
            self.last_lane_vehicles[idx] = current_vehicles
            self.last_arrival_time[idx] = now
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
                if 'arrival_rate' not in data:
                    data['arrival_rate'] = self._calculate_arrival_rate(lane)
                max_arrival = max(max_arrival, data['arrival_rate'])

        max_queue = max(max_queue, 1)
        max_wait = max(max_wait, 1)
        max_arrival = max(max_arrival, 0.1)

        for lane in controlled_lanes:
            if lane in lane_data and lane in self.lane_id_to_idx:
                data = lane_data[lane]
                lane_idx = self.lane_id_to_idx[lane]
                score = self.lane_scores[lane_idx]

                queue_factor = (data['queue_length'] / max_queue) * 10
                wait_factor = (data['waiting_time'] / max_wait) * 5
                arrival_factor = (data.get('arrival_rate', 0) / max_arrival) * 8

                last_green = self.last_green_time[lane_idx]
                starvation_factor = 0
                if current_time - last_green > self.adaptive_params['starvation_threshold']:
                    starvation_factor = min(15, (current_time - last_green -
                                                self.adaptive_params['starvation_threshold']) * 0.3)

                emergency_boost = 20 if data.get('ambulance', False) else 0

                current_phase = traci.trafficlight.getPhase(tl_id)
                phase_efficiency = self._get_phase_efficiency(tl_id, current_phase)

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

    def _handle_ambulance_priority(self, tl_id, controlled_lanes, lane_data, current_time):
        """
        Scan all controlled lanes at the intersection for an emergency (ambulance/authority).
        If found, immediately serve that lane with a priority phase and log the event.
        """
        try:
            ambulance_lanes = [lane for lane in controlled_lanes if lane_data.get(lane, {}).get('ambulance', False)]
            if not ambulance_lanes:
                return False  # No ambulance to serve

            # Find the closest emergency vehicle to the intersection
            min_distance = float('inf')
            target_lane = None
            for lane_id in ambulance_lanes:
                try:
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    for vid in vehicles:
                        if traci.vehicle.getVehicleClass(vid) in ['emergency', 'authority']:
                            lane_length = traci.lane.getLength(lane_id)
                            vehicle_pos = traci.vehicle.getLanePosition(vid)
                            distance = lane_length - vehicle_pos
                            if distance < min_distance:
                                min_distance = distance
                                target_lane = lane_id
                except Exception as e:
                    print(f"Error processing ambulance lane {lane_id}: {e}")

            if target_lane is None:
                return False

            # Find phase for target lane and set green
            phase_index = self._find_phase_for_lane(tl_id, target_lane)
            if phase_index is not None:
                duration = 30 if min_distance < 30 else 20
                traci.trafficlight.setPhase(tl_id, phase_index)
                traci.trafficlight.setPhaseDuration(tl_id, duration)
                self.ambulance_active[tl_id] = True
                self.ambulance_start_time[tl_id] = current_time
                # Log event
                self.rl_agent.training_data.append({
                    'event': 'ambulance_priority',
                    'lane_id': target_lane,
                    'tl_id': tl_id,
                    'phase': phase_index,
                    'simulation_time': current_time,
                    'distance_to_stopline': min_distance,
                    'duration': duration,
                })
                return True

            return False

        except Exception as e:
            print(f"Error in _handle_ambulance_priority: {e}")
            return False

    def _perform_normal_control(self, tl_id, controlled_lanes, lane_data, current_time):
        """Perform normal RL-based traffic control"""
        try:
            # Select target lane based on multiple factors
            if not isinstance(lane_data, dict):
                print(f"⚠️ Error: lane_data is {type(lane_data)}, expected dict")
                return

            target_lane = self._select_target_lane(tl_id, controlled_lanes, lane_data, current_time)
            
            if not target_lane:
                return
                
            # Get state and RL action
            state = self._create_state_vector(target_lane, lane_data)
            if not self.rl_agent.is_valid_state(state):
                return
                
            action = self.rl_agent.get_action(state, lane_id=target_lane)
            
            # Execute action with minimum phase duration enforcement
            # Fix: Handle case where last_phase_change might not be a dict
            if isinstance(self.last_phase_change, dict):
                last_change_time = self.last_phase_change.get(tl_id, 0)
            else:
                # If it's a numpy array or other type, convert or initialize
                print(f"⚠️ Warning: last_phase_change is {type(self.last_phase_change)}, converting to dict")
                self.last_phase_change = {}
                last_change_time = 0
            
            if current_time - last_change_time >= 5:  # 5s minimum
                self._execute_control_action(tl_id, target_lane, action, lane_data, current_time)
                
        except Exception as e:
            print(f"Error in _perform_normal_control: {e}")
            import traceback
            traceback.print_exc()  # This will help debug the actual error
    def _execute_control_action(self, tl_id, target_lane, action, lane_data, current_time):
        """Execute the selected control action"""
        try:
            # Ensure traffic light exists
            if not isinstance(lane_data, dict) or target_lane not in lane_data:
                print("⚠️ Invalid lane_data in _execute_control_action")
                return
                            
            # Get current phase and find target phase for the lane
            current_phase = traci.trafficlight.getPhase(tl_id)
            target_phase = self._find_phase_for_lane(tl_id, target_lane)
            
            if target_phase is None:
                target_phase = current_phase  # Fallback to current phase
                
            if action == 0:  # Set green for target lane
                if current_phase != target_phase:
                    # Ensure we have data for this lane
                    if not isinstance(lane_data, dict) or target_lane not in lane_data:
                        green_time = self.adaptive_params['min_green']
                    else:
                        green_time = self._calculate_dynamic_green(lane_data[target_lane])
                        
                    traci.trafficlight.setPhase(tl_id, target_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, green_time)
                    self.last_phase_change[tl_id] = current_time

                    lane_idx = self.lane_id_to_idx[target_lane]
                    self.last_green_time[lane_idx] = current_time
                    #print(f"🟢 RL ACTION: Green for {target_lane} (duration={green_time}s)")
                            
            elif action == 1:  # Next phase
                phase_count = self._get_phase_count(tl_id)
                next_phase = (current_phase + 1) % phase_count
                traci.trafficlight.setPhase(tl_id, next_phase)
                traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                self.last_phase_change[tl_id] = current_time
                
            elif action == 2:  # Extend current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    extension = min(15, self.adaptive_params['max_green'] - remaining)
                    if extension > 0:
                        traci.trafficlight.setPhaseDuration(tl_id, remaining + extension)
                except Exception as e:
                    print(f"Could not extend phase: {e}")
                    
            elif action == 3:  # Shorten current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    if remaining > self.adaptive_params['min_green'] + 5:
                        reduction = min(5, remaining - self.adaptive_params['min_green'])
                        traci.trafficlight.setPhaseDuration(tl_id, remaining - reduction)
                        #print(f"⏳ RL ACTION: Shortened phase by {reduction}s")
                except Exception as e:
                    print(f"Could not shorten phase: {e}")
                    
            elif action == 4:  # Balanced phase switch
                balanced_phase = self._get_balanced_phase(tl_id, lane_data)
                if balanced_phase != current_phase:
                    traci.trafficlight.setPhase(tl_id, balanced_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                    self.last_phase_change[tl_id] = current_time
                   # print(f"⚖️ RL ACTION: Balanced phase ({balanced_phase})")
                    
            # Update phase utilization stats
            new_phase = traci.trafficlight.getPhase(tl_id)
            key = (tl_id, new_phase)
            self.phase_utilization[key] = self.phase_utilization.get(key, 0) + 1
            
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
    
    def _create_intersection_state_vector(self, tl_id, intersection_data):
        """Create a state vector for the intersection tl_id"""
        data = intersection_data[tl_id]
        queues = data.get('queues', [])
        waits = data.get('waits', [])
        speeds = data.get('speeds', [])
        left_q = data.get('left_q', 0)
        right_q = data.get('right_q', 0)
        n_phases = data.get('n_phases', 4)
        current_phase = data.get('current_phase', 0)
        state = np.array([
            np.max(queues) if queues else 0,
            np.mean(queues) if queues else 0,
            np.min(speeds) if speeds else 0,
            np.mean(speeds) if speeds else 0,
            np.max(waits) if waits else 0,
            np.mean(waits) if waits else 0,
            float(current_phase) / max(n_phases-1, 1),
            float(n_phases),
            left_q,
            right_q
        ])
        return state

    def _process_rl_learning(self, intersection_data, current_time):
        """Process RL learning for each intersection (per-intersection RL)"""
        try:
            for tl_id in traci.trafficlight.getIDList():
                if tl_id not in intersection_data:
                    continue

                # Extract intersection-level data
                data = intersection_data[tl_id]
                state = self._create_intersection_state_vector(tl_id, intersection_data)
                if not self.rl_agent.is_valid_state(state):
                    continue

                queues = data.get('queues', [])
                waits = data.get('waits', [])
                speeds = data.get('speeds', [])
                left_q = data.get('left_q', 0)
                right_q = data.get('right_q', 0)
                n_phases = data.get('n_phases', 0)
                current_phase = data.get('current_phase', -1)

                reward = -0.7 * sum(queues) - 0.3 * sum(waits)  # Simplified reward
                raw_reward = reward
                reward_components = {
                    "queue_penalty": -0.7 * sum(queues),
                    "wait_penalty": -0.3 * sum(waits),
                    "total_raw": reward,
                    "normalized": reward,
                }

                # Update Q-table if previous state/action exist
                if tl_id in self.previous_states and tl_id in self.previous_actions:
                    prev_state = self.previous_states[tl_id]
                    prev_action = self.previous_actions[tl_id]
                    state_key = self.rl_agent._state_to_key(prev_state, tl_id)
                    q_value = (
                        float(self.rl_agent.q_table[state_key][prev_action])
                        if state_key in self.rl_agent.q_table else None
                    )
                    log_entry = {
                        'episode': getattr(self, 'current_episode', 0),
                        'simulation_time': current_time,
                        'lane_id': "",
                        'edge_id': "",
                        'route_id': "",
                        'action': prev_action,
                        'action_name': self.rl_agent._get_action_name(prev_action),
                        'reward': reward,
                        'raw_reward': raw_reward,
                        'q_value': q_value,
                        'queue_length': max(queues) if queues else 0,
                        'waiting_time': max(waits) if waits else 0,
                        'density': "",
                        'mean_speed': np.mean(speeds) if speeds else 0,
                        'flow': "",
                        'queue_route': "",
                        'flow_route': "",
                        'ambulance': "",
                        'left_turn': left_q,
                        'right_turn': right_q,
                        'tl_id': tl_id,
                        'phase_id': current_phase,
                        'epsilon': self.rl_agent.epsilon,
                        'learning_rate': self.rl_agent.learning_rate,
                        'reward_components': reward_components,
                        'adaptive_params': self.adaptive_params.copy()
                    }
                    self.rl_agent.update_q_table(
                        prev_state,
                        prev_action,
                        reward,
                        state,
                        tl_id=tl_id,
                        extra_info=log_entry,
                        action_size=n_phases
                    )

                # Store current state/action for next update
                action = self.rl_agent.get_action(state, tl_id=tl_id)
                self.previous_states[tl_id] = state
                self.previous_actions[tl_id] = action
        except Exception as e:
            print(f"Error in _process_rl_learning: {e}")
            traceback.print_exc()
    def _create_state_vector(self, lane_id, lane_data):
        """Create comprehensive state vector"""
        try:
            if not isinstance(lane_data, dict) or lane_id not in lane_data:
                return np.zeros(self.rl_agent.state_size)
            
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
            route_queue_norm = min(data.get('queue_route', 0) / (self.norm_bounds['queue'] * 3), 1.0)
            route_flow_norm = min(data.get('flow_route', 0) / (self.norm_bounds['flow'] * 3), 1.0)    
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
            last_green = self.last_green_time[self.lane_id_to_idx[lane_id]]
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
                self.lane_scores[self.lane_id_to_idx[lane_id]] / 100,
                phase_efficiency  # Normalized lane score
            ])
            
            # Ensure no invalid values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            
            return state
            
        except Exception as e:
            print(f"Error creating state vector for {lane_id}: {e}")
            return np.zeros(self.rl_agent.state_size)  # Reduced state size

    def _calculate_reward(self, lane_id, lane_data, action_taken, current_time):
        """Calculate comprehensive reward signal with detailed components"""
        try:
            # Get data for this specific lane
            if not isinstance(lane_data, dict) or lane_id not in lane_data:
                return 0.0, {}, 0.0
                
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
            last_green = self.last_green_time[self.lane_id_to_idx[lane_id]]
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
        # Update adaptive parameters based on recent reward history
        if self.rl_agent.reward_history:
            avg_reward = np.mean(self.rl_agent.reward_history)
            print(f"Average reward this episode: {avg_reward:.2f}")
            # Example adaptive parameter update logic:
            if avg_reward < -10:  # High queues/waits (bad)
                self.adaptive_params['min_green'] = min(self.adaptive_params['min_green'] + 1, self.adaptive_params['max_green'])
                self.adaptive_params['max_green'] = min(self.adaptive_params['max_green'] + 5, 120)
            elif avg_reward > -2:  # Low queues/waits (good)
                self.adaptive_params['min_green'] = max(self.adaptive_params['min_green'] - 1, 5)
                self.adaptive_params['max_green'] = max(self.adaptive_params['max_green'] - 5, 30)
        else:
            print("No reward history for adaptive param update.")
        self.rl_agent.adaptive_params = self.adaptive_params.copy()

        print("🔄 Updated adaptive parameters:", self.adaptive_params)
        self.rl_agent.save_model(adaptive_params=self.adaptive_params)
        print(f"🚦 Epsilon after training/episode: {self.rl_agent.epsilon}")  # <--- Add here

        # Clear for next episode
        self.previous_states.clear()
        self.previous_actions.clear()
        self.rl_agent.reward_history.clear()

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

# Main simulation loop using universal controller
def start_universal_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1, num_retries=1, retry_delay=1, mode="train"):
    controller = None
    try:
        for episode in range(episodes):
            print(f"\n{'='*50}")
            print(f"🚦 STARTING UNIVERSAL EPISODE {episode + 1}/{episodes}")
            print(f"{'='*50}")
            sumo_binary = "sumo-gui" if use_gui else "sumo"
            sumo_cmd = [
                os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary),
                '-c', sumocfg_path,
                '--start', '--quit-on-end'
            ]
            traci.start(sumo_cmd)
            controller = UniversalSmartTrafficController(sumocfg_path=sumocfg_path, mode=mode)
            controller.initialize()
            controller.current_episode = episode + 1
            step = 0
            tstart = time.time()
            while traci.simulation.getMinExpectedNumber() > 0:
                if max_steps and step >= max_steps:
                    print(f"Reached max steps ({max_steps}), ending episode.")
                    break
                controller.run_step()
                traci.simulationStep()
                step += 1
                if step % 200 == 0:
                    print(f"Episode {episode + 1}: Step {step} completed, elapsed: {time.time()-tstart:.2f}s")
            print(f"Episode {episode + 1} completed after {step} steps")
            controller.end_episode()
            traci.close()
            if episode < episodes - 1:
                time.sleep(2)
        print(f"\n🎉 All {episodes} episodes completed!")
    except Exception as e:
        print(f"Error in universal simulation: {e}")
    finally:
        try:
            traci.close()
        except:
            pass
        print("Simulation resources cleaned up")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run universal SUMO RL traffic simulation")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of retries if connection fails')
    parser.add_argument('--retry-delay', type=int, default=1, help='Delay in seconds between retries')
    parser.add_argument('--mode', choices=['train', 'eval', 'adaptive'], default='train',
                        help='Controller mode: train (explore+learn), eval (exploit only), adaptive (exploit+learn)')
    args = parser.parse_args()
    start_universal_simulation(
        args.sumo, 
        args.gui, 
        args.max_steps, 
        args.episodes, 
        args.num_retries, 
        args.retry_delay,
        mode=args.mode
    )