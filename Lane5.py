from collections import defaultdict
import os, sys, traci, warnings,time,argparse,traceback,pickle,datetime
import numpy as np
from pyinstrument import Profiler
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify

# === Flask app defined at module/global level ===
app = Flask(__name__)
controller = None  # global reference for the API

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/api/agent_params', methods=['GET'])
def get_agent_params():
    global controller
    try:
        if controller is None:
            return jsonify({'error': 'Controller not initialized'}), 503
        params = controller.rl_agent.adaptive_params
        return jsonify(params)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/epsilon', methods=['GET'])
def get_epsilon():
    global controller
    try:
        if controller is None:
            return jsonify({'error': 'Controller not initialized'}), 503
        return jsonify({'epsilon': controller.rl_agent.epsilon})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_episode', methods=['POST'])
def run_episode():
    # Placeholder: You'd want to trigger an episode run here
    return jsonify({'message': 'Episode run (not implemented in placeholder)'})

# SUMO_HOME setup
os.environ.setdefault('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path: sys.path.append(tools)

def subscribe_vehicles(vehicle_ids):
    """Subscribe to vehicle class for all vehicles once per step."""
    for vid in vehicle_ids:
        try:
            traci.vehicle.subscribe(vid, [traci.constants.VAR_VEHICLECLASS])
        except traci.TraCIException:
            pass  # Vehicle might have left network

def get_vehicle_classes(vehicle_ids):
    """Efficiently get vehicle classes for all vehicles this step."""
    classes = {}
    for vid in vehicle_ids:
        results = traci.vehicle.getSubscriptionResults(vid)
        if results and traci.constants.VAR_VEHICLECLASS in results:
            classes[vid] = results[traci.constants.VAR_VEHICLECLASS]
        else:
            # Fallback in case subscription result missing
            try:
                classes[vid] = traci.vehicle.getVehicleClass(vid)
            except traci.TraCIException:
                classes[vid] = None
    return classes
def subscribe_lanes(lane_ids):
    for lid in lane_ids:
        traci.lane.subscribe(lid, [
            traci.constants.LAST_STEP_VEHICLE_NUMBER,
            traci.constants.LAST_STEP_MEAN_SPEED,
            traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
            traci.constants.LAST_STEP_VEHICLE_ID_LIST
        ])

def detect_turning_lanes_with_traci():
    left, right = set(), set()
    for lid in traci.lane.getIDList():
        for c in traci.lane.getLinks(lid):
            idx = 6 if len(c)>6 else 3 if len(c)>3 else None
            if idx and c[idx] == 'l': left.add(lid)
            if idx and c[idx] == 'r': right.add(lid)
    return left, right
class EnhancedQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 q_table_file="enhanced_q_table.pkl", mode="train", adaptive_params=None):
        self.state_size, self.action_size = state_size, action_size
        self.learning_rate, self.discount_factor = learning_rate, discount_factor
        self.epsilon, self.epsilon_decay, self.min_epsilon = epsilon, epsilon_decay, min_epsilon
        self.q_table, self.training_data = {}, []
        self.q_table_file, self._loaded_training_count = q_table_file, 0
        self.reward_components, self.reward_history = [], []
        self.learning_rate_decay, self.min_learning_rate = 0.999, 0.01
        self.consecutive_no_improvement, self.max_no_improvement = 0, 100
        self.mode = mode
        self.adaptive_params = adaptive_params or {
            'min_green': 30, 'max_green': 80, 'starvation_threshold': 40, 'reward_scale': 40,
            'queue_weight': 0.6, 'wait_weight': 0.3, 'flow_weight': 0.5, 'speed_weight': 0.2, 
            'left_turn_priority': 1.2, 'empty_green_penalty': 15, 'congestion_bonus': 10
        }
        if mode == "eval": self.epsilon = 0.0
        elif mode == "adaptive": self.epsilon = 0.01
        print(f"AGENT INIT: mode={self.mode}, epsilon={self.epsilon}")

    def is_valid_state(self, state):
        arr = np.array(state)
        return (
            isinstance(state, (list, np.ndarray))
            and arr.size == self.state_size
            and not (np.isnan(arr).any() or np.isinf(arr).any())
            and (np.abs(arr) <= 100).all()
            and not np.all(arr == 0)
        )
    def get_action(self, state, tl_id=None, action_size=None):
        action_size = action_size or self.action_size
        key = self._state_to_key(state, tl_id)
        if key not in self.q_table or len(self.q_table[key]) < action_size:
            self.q_table[key] = np.zeros(action_size)
        qs = self.q_table[key][:action_size]
        
        if self.mode == "train" and np.random.rand() < self.epsilon:
            # Congestion-biased exploration
            if hasattr(state, 'queues'):
                congestion_scores = np.array([q + w*0.5 for q, w in zip(state.queues, state.waits)])
                if congestion_scores.sum() > 0:
                    congestion_scores = congestion_scores / congestion_scores.sum()
                    return np.random.choice(range(action_size), p=congestion_scores)
            return np.random.randint(action_size)
        return np.argmax(qs)

    def _state_to_key(self, state, tl_id=None):
        try:
            if isinstance(state, (np.ndarray, list)):
                arr = np.round(np.array(state), 2)
                key = tuple(arr.tolist())
            else:
                key = tuple(state) if hasattr(state, '__iter__') else (state,)
            return (tl_id, key) if tl_id is not None else key
        except Exception:
            return (tl_id, (0,)) if tl_id is not None else (0,)

    def update_q_table(self, state, action, reward, next_state, tl_id=None, extra_info=None, action_size=None):
        if self.mode == "eval" or not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return
        if reward is None or np.isnan(reward) or np.isinf(reward):
            return
        action_size = action_size or self.action_size
        sk, nsk = self._state_to_key(state, tl_id), self._state_to_key(next_state, tl_id)
        for k in [sk, nsk]:
            if k not in self.q_table or len(self.q_table[k]) < action_size:
                self.q_table[k] = np.zeros(action_size)
        q, nq = self.q_table[sk][action], np.max(self.q_table[nsk])
        new_q = q + self.learning_rate * (reward + self.discount_factor * nq - q)
        self.q_table[sk][action] = new_q if not (np.isnan(new_q) or np.isinf(new_q)) else q

        entry = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action, 'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'q_value': self.q_table[sk][action], 'timestamp': time.time(),
            'learning_rate': self.learning_rate, 'epsilon': self.epsilon,
            'tl_id': tl_id, 'adaptive_params': self.adaptive_params.copy()
        }
        if extra_info:
            entry.update({k: v for k, v in extra_info.items() if k != "reward"})
        self.training_data.append(entry)
        self._update_adaptive_parameters(reward)

    def _get_action_name(self, action):
        return {
            0: "Set Green", 1: "Next Phase", 2: "Extend Phase",
            3: "Shorten Phase", 4: "Balanced Phase"
        }.get(action, f"Unknown Action {action}")

    def _state_to_key(self, state, tl_id=None):
        try:
            if isinstance(state, (np.ndarray, list)):
                arr = np.round(np.array(state), 2)
                key = tuple(arr.tolist())
            else:
                key = tuple(state) if hasattr(state, '__iter__') else (state,)
            return (tl_id, key) if tl_id is not None else key
        except Exception:
            return (tl_id, (0,)) if tl_id is not None else (0,)

    def update_q_table(self, state, action, reward, next_state, tl_id=None, extra_info=None, action_size=None):
        if self.mode == "eval" or not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return
        if reward is None or np.isnan(reward) or np.isinf(reward):
            return
        action_size = action_size or self.action_size
        sk, nsk = self._state_to_key(state, tl_id), self._state_to_key(next_state, tl_id)
        for k in [sk, nsk]:
            if k not in self.q_table or len(self.q_table[k]) < action_size:
                self.q_table[k] = np.zeros(action_size)
        q, nq = self.q_table[sk][action], np.max(self.q_table[nsk])
        new_q = q + self.learning_rate * (reward + self.discount_factor * nq - q)
        self.q_table[sk][action] = new_q if not (np.isnan(new_q) or np.isinf(new_q)) else q

        entry = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action, 'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'q_value': self.q_table[sk][action], 'timestamp': time.time(),
            'learning_rate': self.learning_rate, 'epsilon': self.epsilon,
            'tl_id': tl_id, 'adaptive_params': self.adaptive_params.copy()
        }
        if extra_info:
            entry.update({k: v for k, v in extra_info.items() if k != "reward"})
        self.training_data.append(entry)
        self._update_adaptive_parameters(reward)

    def load_model(self, filepath=None):
        filepath = filepath or self.q_table_file
        print(f"Attempting to load Q-table from: {filepath}\nAbsolute path: {os.path.abspath(filepath)}\nFile exists? {os.path.exists(filepath)}")
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.q_table = {k: np.array(v) for k, v in data.get('q_table', {}).items()}
                self._loaded_training_count = len(data.get('training_data', []))
                params = data.get('params', {})
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.discount_factor = params.get('discount_factor', self.discount_factor)
                adaptive_params = data.get('adaptive_params', {})
                print(f"After model load, epsilon={self.epsilon}")
                print(f"Loaded Q-table with {len(self.q_table)} states from {filepath}")
                if adaptive_params: print("📋 Loaded adaptive parameters from previous run")
                return True, adaptive_params
            print("No existing Q-table, starting fresh")
            return False, {}
        except Exception as e:
            print(f"Error loading model: {e}\nNo existing Q-table, starting fresh")
            return False, {}

    def save_model(self, filepath=None, adaptive_params=None):
        filepath = filepath or self.q_table_file
        try:
            if os.path.exists(filepath):
                backup = f"{filepath}.bak_{datetime.datetime.now():%Y%m%d_%H%M%S}"
                for _ in range(3):
                    try:
                        os.rename(filepath, backup)
                        break
                    except Exception as e:
                        print(f"Retrying backup: {e}")
                        time.sleep(0.5)
            meta = {
                'last_updated': datetime.datetime.now().isoformat(),
                'training_count': len(self.training_data),
                'average_reward': np.mean([x.get('reward', 0) for x in self.training_data[-100:]]) if self.training_data else 0,
                'reward_components': [x.get('reward_components', {}) for x in self.training_data[-100:]]
            }
            params = {k: getattr(self, k) for k in ['state_size','action_size','learning_rate','discount_factor','epsilon','epsilon_decay','min_epsilon']}
            model_data = {
                'q_table': {k: v.tolist() for k, v in self.q_table.items()},
                'training_data': self.training_data,
                'params': params,
                'metadata': meta
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
from collections import defaultdict
import numpy as np

class UniversalSmartTrafficController:
    DILEMMA_ZONE_THRESHOLD = 12.0  # meters

    def __init__(self, sumocfg_path=None, mode="train"):
        self.mode = mode
        self.step_count = 0
        self.current_episode = 0
        self.max_consecutive_left = 1
        self.subscribed_vehicles = set()
        self.left_turn_lanes = set()
        self.right_turn_lanes = set()
        self.lane_id_list = []
        self.lane_id_to_idx = {}
        self.idx_to_lane_id = {}
        self.lane_to_tl = {}
        self.tl_action_sizes = {}
        self.last_green_time = None
        self.pending_next_phase = {}
        self.lane_lengths = {}
        self.lane_edge_ids = {}
        self.intersection_data = {}
        self.tl_logic_cache = {}
        self.phase_utilization = defaultdict(int)
        self.last_phase_change = {}
        self.ambulance_active = defaultdict(bool)
        self.ambulance_start_time = defaultdict(float)
        self.left_phase_counter = defaultdict(int)
        self.previous_states = {}
        self.previous_actions = {}
        
        # Normalization bounds
        self.norm_bounds = {
            'queue': 20, 'wait': 60, 'speed': 13.89,  # ~50 km/h
            'flow': 30, 'density': 0.2, 'arrival_rate': 5,
            'time_since_green': 120
        }
        
        self.rl_agent = EnhancedQLearningAgent(state_size=12, action_size=8, mode=mode)
        self.rl_agent.load_model()

        self.adaptive_params = {
            'min_green': 30, 'max_green': 80, 'starvation_threshold': 40,
            'reward_scale': 40, 'queue_weight': 0.6, 'wait_weight': 0.3,
            'flow_weight': 0.5, 'speed_weight': 0.2, 'left_turn_priority': 1.2,
            'empty_green_penalty': 15, 'congestion_bonus': 10
        }

    def initialize(self):
        for tl_id in traci.trafficlight.getIDList():
            phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            for i, phase in enumerate(phases):
                print(f"  Phase {i}: {phase.state} (duration {getattr(phase, 'duration', '?')})")
        
        self.lane_id_list = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        self.tl_action_sizes = {tl_id: len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                              for tl_id in traci.trafficlight.getIDList()}
        self.lane_id_to_idx = {lid: i for i, lid in enumerate(self.lane_id_list)}
        self.idx_to_lane_id = dict(enumerate(self.lane_id_list))
        self.last_green_time = np.zeros(len(self.lane_id_list))
        subscribe_lanes(self.lane_id_list)
        self.left_turn_lanes, self.right_turn_lanes = detect_turning_lanes_with_traci()
        self.lane_lengths = {lid: traci.lane.getLength(lid) for lid in self.lane_id_list}
        self.lane_edge_ids = {lid: traci.lane.getEdgeID(lid) for lid in self.lane_id_list}
    def _init_left_turn_lanes(self):
        try:
            self.left_turn_lanes.clear()
            for lane_id in traci.lane.getIDList():
                if any((len(conn) > 6 and conn[6] == 'l') or (len(conn) > 3 and conn[3] == 'l')
                    for conn in traci.lane.getLinks(lane_id)):
                    self.left_turn_lanes.add(lane_id)
            print(f"Auto-detected left-turn lanes: {sorted(self.left_turn_lanes)}")
        except Exception as e:
            print(f"Error initializing left-turn lanes: {e}")

    def _is_in_dilemma_zone(self, tl_id, controlled_lanes, lane_data):
        """Return True if any vehicle is in the dilemma zone for lanes about to switch from green to red."""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic: return False
            state = logic.phases[traci.trafficlight.getPhase(tl_id)].state
            for lane_idx, lane in enumerate(controlled_lanes):
                if lane_idx < len(state) and state[lane_idx].upper() == 'G':
                    for vid in traci.lane.getLastStepVehicleIDs(lane):
                        dist = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(vid)
                        if 0 < dist <= self.DILEMMA_ZONE_THRESHOLD:
                            return True
            return False
        except Exception as e:
            print(f"Error in _is_in_dilemma_zone: {e}")
            return False

    def _find_starved_lane(self, controlled_lanes, current_time):
        """Returns the lane starved past the threshold, or None."""
        for lane in controlled_lanes:
            idx = self.lane_id_to_idx.get(lane)
            if idx is not None and current_time - self.last_green_time[idx] > self.adaptive_params['starvation_threshold']:
                return lane
        return None
    
    def _handle_protected_left_turn(self, tl_id, controlled_lanes, lane_data, current_time):
        """
        Scan all left-turn lanes at the intersection. 
        If any left-turn lane has high queue/wait, serve it with a protected left phase and log events.
        """
        try:
            # Find left-turn lanes needing service
            left = [
                l for l in controlled_lanes
                if lane_data.get(l, {}).get('left_turn') and (
                    lane_data[l]['queue_length'] > 3 or lane_data[l]['waiting_time'] > 10
                )
            ]
            if not left: return False
            # Pick lane with highest queue/wait
            target = max(left, key=lambda x: (lane_data[x]['queue_length'], lane_data[x]['waiting_time']))
            phase = self._find_best_left_turn_phase(tl_id, target, lane_data)
            self.rl_agent.training_data.append({
                'event': 'protected_left_needed', 'lane_id': target, 'tl_id': tl_id,
                'phase': phase, 'simulation_time': current_time,
                'queue_length': lane_data[target]['queue_length'],
                'waiting_time': lane_data[target]['waiting_time'],
            })
            if phase is not None and traci.trafficlight.getPhase(tl_id) != phase:
                q, w = lane_data[target]['queue_length'], lane_data[target]['waiting_time']
                d = min(max(5 + min(q * 0.5, 10) + min(w * 0.1, 5), self.adaptive_params['min_green']), self.adaptive_params['max_green'])
                traci.trafficlight.setPhase(tl_id, phase)
                traci.trafficlight.setPhaseDuration(tl_id, d)
                self.rl_agent.training_data.append({
                    'event': 'protected_left_triggered', 'lane_id': target, 'tl_id': tl_id,
                    'phase': phase, 'simulation_time': current_time,
                    'queue_length': q, 'waiting_time': w, 'duration': d,
                })
                idx = self.lane_id_to_idx[target]
                self.last_green_time[idx] = self.last_phase_change[tl_id] = current_time
                self.phase_utilization[(tl_id, phase)] = self.phase_utilization.get((tl_id, phase), 0) + 1
                return True
            return False
        except Exception as e:
            print(f"Error in _handle_protected_left_turn: {e}")
            return False

    def _find_best_left_turn_phase(self, tl_id, left_turn_lane, lane_data):
        """Find best phase for protected left turns (prefer exclusive left phases)."""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic: return None
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            idx = lanes.index(left_turn_lane)
            best, score = None, -1
            for i, ph in enumerate(logic.phases):
                state = ph.state.upper()
                if idx >= len(state) or state[idx] != 'G': continue
                s = 20 if all(state[j] in 'rR' or j == idx for j in range(len(state))) else 0
                s -= sum(lane_data.get(l, {}).get('queue_length', 0) * 0.5
                        for j, l in enumerate(lanes) if j != idx and j < len(state) and state[j] == 'G' and lane_data.get(l, {}).get('queue_length', 0) > 5)
                if s > score: best, score = i, s
            return best
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
                self.tl_logic_cache[tl_id] = traci.trafficlight.getAllProgramLogics(tl_id)[0]
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
            return len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id))
        except Exception as e:
            print(f"Error getting phase count for {tl_id}: {e}")
            return 4

    def _get_phase_name(self, tl_id, phase_idx):
        """Get phase name"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if logic and phase_idx < len(logic.phases):
                return getattr(logic.phases[phase_idx], 'name', f'phase_{phase_idx}')
        except Exception as e:
            print(f"Error getting phase name for {tl_id}[{phase_idx}]: {e}")
        return f'phase_{phase_idx}'

    def _switch_phase_with_yellow_if_needed(self, tl_id, current_phase, target_phase, logic, controlled_lanes, lane_data, current_time, min_green=None):
        yellow_phase = self._get_yellow_phase(logic, current_phase, target_phase)
        if yellow_phase is not None and yellow_phase != current_phase:
            yellow_duration = self._calculate_adaptive_yellow(tl_id, controlled_lanes, lane_data)
            traci.trafficlight.setPhase(tl_id, yellow_phase)
            traci.trafficlight.setPhaseDuration(tl_id, yellow_duration)
            self.last_phase_change[tl_id] = current_time
            self.pending_next_phase[tl_id] = (target_phase, current_time)
            return True
        traci.trafficlight.setPhase(tl_id, target_phase)
        traci.trafficlight.setPhaseDuration(tl_id, min_green or self.adaptive_params['min_green'])
        self.last_phase_change[tl_id] = current_time
        return False

    def _calculate_adaptive_yellow(self, tl_id, controlled_lanes, lane_data):
        try:
            max_speed = 0
            for lane in controlled_lanes:
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    res = traci.vehicle.getSubscriptionResults(vid)
                    if res and traci.constants.VAR_SPEED in res:
                        s = res[traci.constants.VAR_SPEED]
                        if s > max_speed:
                            max_speed = s
            # Standard formula: t_yellow = t_reaction + v / (2*a)
            # With t_reaction=1s, a=3m/s^2
            yellow_time = max(3.0, min(6.0, 1.0 + max_speed / (2 * 3.0)))
            return yellow_time
        except Exception as e:
            print(f"Error in _calculate_adaptive_yellow: {e}")
            return 3.0
    def run_step(self):
        try:
            self.step_count += 1
            current_time = traci.simulation.getTime()
            self.intersection_data = {}

            # Vehicle management
            all_vehicles = set(traci.vehicle.getIDList())
            vehicle_classes = get_vehicle_classes(all_vehicles)
            lane_data = self._collect_enhanced_lane_data(vehicle_classes, all_vehicles)
            self.subscribed_vehicles.intersection_update(all_vehicles)
            new_vehicles = all_vehicles - self.subscribed_vehicles
            subscribe_vehicles(new_vehicles)
            self.subscribed_vehicles.update(new_vehicles)

            for tl_id in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                logic = self._get_traffic_light_logic(tl_id)
                current_phase = traci.trafficlight.getPhase(tl_id)
                
                # Handle pending phase transitions
                if tl_id in self.pending_next_phase:
                    pending_phase, set_time = self.pending_next_phase[tl_id]
                    phase_duration = logic.phases[current_phase].duration if logic else 3
                    if current_time - set_time >= phase_duration - 0.1:
                        traci.trafficlight.setPhase(tl_id, pending_phase)
                        traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                        self.last_phase_change[tl_id] = current_time
                        del self.pending_next_phase[tl_id]
                    continue
                
                # Priority handling
                if self._handle_ambulance_priority(tl_id, controlled_lanes, lane_data, current_time): 
                    continue
                if self._handle_protected_left_turn(tl_id, controlled_lanes, lane_data, current_time): 
                    continue
                
                # Starvation detection - only consider lanes with vehicles
                starved_lanes = []
                for lane in controlled_lanes:
                    idx = self.lane_id_to_idx.get(lane)
                    if idx is not None and lane in lane_data and lane_data[lane]['queue_length'] > 0:
                        time_since_green = current_time - self.last_green_time[idx]
                        if time_since_green > self.adaptive_params['starvation_threshold']:
                            starved_lanes.append((lane, time_since_green))
                
                if starved_lanes:
                    most_starved_lane = max(starved_lanes, key=lambda x: x[1])[0]
                    starved_phase = self._find_phase_for_lane(tl_id, most_starved_lane)
                    if current_phase != starved_phase:
                        switched = self._switch_phase_with_yellow_if_needed(
                            tl_id, current_phase, starved_phase, logic, controlled_lanes, lane_data, current_time)
                        if not switched:
                            traci.trafficlight.setPhase(tl_id, starved_phase)
                            traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                            self.last_phase_change[tl_id] = current_time
                    self.last_green_time[self.lane_id_to_idx[most_starved_lane]] = current_time
                    # Debug print after potential phase switch
                    self.debug_green_lanes(tl_id, lane_data)
                    continue
                
                # Normal RL control
                queues = np.array([lane_data[l]['queue_length'] for l in controlled_lanes if l in lane_data])
                waits = [lane_data[l]['waiting_time'] for l in controlled_lanes if l in lane_data]
                speeds = [lane_data[l]['mean_speed'] for l in controlled_lanes if l in lane_data]
                left_q = sum(lane_data[l]['queue_length'] for l in controlled_lanes 
                            if l in self.left_turn_lanes and l in lane_data)
                right_q = sum(lane_data[l]['queue_length'] for l in controlled_lanes 
                            if l in self.right_turn_lanes and l in lane_data)
                
                self.intersection_data[tl_id] = {
                    'queues': queues, 'waits': waits, 'speeds': speeds,
                    'left_q': left_q, 'right_q': right_q,
                    'n_phases': self.tl_action_sizes[tl_id],
                    'current_phase': current_phase
                }
                
                state = self._create_intersection_state_vector(tl_id, self.intersection_data)
                action = self.rl_agent.get_action(state, tl_id, action_size=self.tl_action_sizes[tl_id])
                
                # Only switch if minimum green time has passed and no dilemma zone vehicles
                last_change = self.last_phase_change.get(tl_id, -9999)
                if (current_time - last_change >= self.adaptive_params['min_green'] and 
                    action != current_phase and
                    not self._is_in_dilemma_zone(tl_id, controlled_lanes, lane_data)):
                    if not self._phase_has_traffic(logic, action, controlled_lanes, lane_data):
                        #print(f"[DEBUG] Skipping phase {action} at TL {tl_id}: all green lanes empty.")
                        continue
                    switched = self._switch_phase_with_yellow_if_needed(
                        tl_id, current_phase, action, logic, controlled_lanes, lane_data, current_time)
                    if not switched:
                        traci.trafficlight.setPhase(tl_id, action)
                        traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                        self.last_phase_change[tl_id] = current_time
                        self._process_rl_learning(self.intersection_data, lane_data, current_time)
                # Debug print after any RL action/phase switch or NOT (always show current green lanes)
                self.debug_green_lanes(tl_id, lane_data)

        except Exception as e:
            print(f"Error in run_step: {e}")
            traceback.print_exc()

    def _phase_has_traffic(self, logic, action, controlled_lanes, lane_data):
        """Returns True if at least one green lane in the selected phase has traffic."""
        phase_state = logic.phases[action].state
        for lane_idx, lane in enumerate(controlled_lanes):
            if lane_idx < len(phase_state) and phase_state[lane_idx].upper() == 'G':
                if lane_data.get(lane, {}).get("queue_length", 0) > 0:
                    return True
        return False

    def debug_green_lanes(self, tl_id, lane_data):
        logic = self._get_traffic_light_logic(tl_id)
        current_phase = traci.trafficlight.getPhase(tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        phase_state = logic.phases[current_phase].state
        for lane_idx, lane in enumerate(controlled_lanes):
            if lane_idx < len(phase_state) and phase_state[lane_idx].upper() == 'G':
                qlen = lane_data.get(lane, {}).get("queue_length", None)
                #print(f"[DEBUG] GREEN: Lane {lane} | queue_length={qlen} | {'EMPTY' if qlen == 0 else 'TRAFFIC'}")
    def _get_yellow_phase(self, logic, from_idx, to_idx):
        """
        Find a yellow (amber) phase between two phases.
        Returns the yellow phase index if found, else None.
        """
        if not logic or from_idx == to_idx:
            return None
        current = logic.phases[from_idx].state.upper()
        target = logic.phases[to_idx].state.upper()
        n = len(current)
        for phase_idx, phase in enumerate(logic.phases):
            state = phase.state.upper()
            if len(state) != n:
                continue
            matches = True
            for i in range(n):
                if current[i] == 'G' and target[i] == 'R':
                    if state[i] != 'Y':
                        matches = False
                        break
                elif state[i] != current[i]:
                    matches = False
                    break
            if matches:
                return phase_idx
        return None

    def _collect_enhanced_lane_data(self, vehicle_classes, all_vehicles):
        lane_data = {}
        lane_vehicle_ids = {}
        lane_ids = self.lane_id_list

        # Batch subscription results for all lanes
        results_dict = {lid: traci.lane.getSubscriptionResults(lid) for lid in lane_ids}

        # Preallocate numpy arrays for bulk stats
        vehicle_count = np.zeros(len(lane_ids), dtype=np.float32)
        mean_speed = np.zeros(len(lane_ids), dtype=np.float32)
        queue_length = np.zeros(len(lane_ids), dtype=np.float32)
        waiting_time = np.zeros(len(lane_ids), dtype=np.float32)
        lane_length = np.array([self.lane_lengths.get(lid, 1.0) for lid in lane_ids], dtype=np.float32)
        ambulance_mask = np.zeros(len(lane_ids), dtype=bool)

        # Prepare vehicle ids array for all lanes
        vehicle_ids_arr = []

        # Gather all lane stats at once, vectorized where possible
        for idx, lane_id in enumerate(lane_ids):
            results = results_dict[lane_id]
            if not results:
                vehicle_ids_arr.append([])
                continue

            vids = results.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, [])
            vids = [vid for vid in vids if vid in all_vehicles]
            vehicle_ids_arr.append(vids)
            vehicle_count[idx] = results.get(traci.constants.LAST_STEP_VEHICLE_NUMBER, 0)
            mean_speed[idx] = max(results.get(traci.constants.LAST_STEP_MEAN_SPEED, 0), 0.0)
            queue_length[idx] = results.get(traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            waiting_time[idx] = results.get(traci.constants.VAR_ACCUMULATED_WAITING_TIME, 0)
            ambulance_mask[idx] = any(vehicle_classes.get(vid) == 'emergency' for vid in vids)

        densities = np.divide(vehicle_count, lane_length, out=np.zeros_like(vehicle_count), where=lane_length > 0)
        left_turn_mask = np.array([lid in self.left_turn_lanes for lid in lane_ids])
        right_turn_mask = np.array([lid in self.right_turn_lanes for lid in lane_ids])

        # Batch fill lane_data dicts
        for idx, lane_id in enumerate(lane_ids):
            vids = vehicle_ids_arr[idx]
            def safe_count_classes(vids):
                from collections import defaultdict
                counts = defaultdict(int)
                for vid in vids:
                    vclass = vehicle_classes.get(vid)
                    if vclass: counts[vclass] += 1
                return counts

            lane_data[lane_id] = {
                'queue_length': float(queue_length[idx]),
                'waiting_time': float(waiting_time[idx]),
                'density': float(densities[idx]),
                'mean_speed': float(mean_speed[idx]),
                'vehicle_ids': vids,
                'flow': float(vehicle_count[idx]),
                'lane_id': lane_id,
                'edge_id': self.lane_edge_ids.get(lane_id, ""),
                'route_id': self._get_route_for_lane(lane_id, all_vehicles),
                'ambulance': bool(ambulance_mask[idx]),
                'vehicle_classes': safe_count_classes(vids),
                'left_turn': bool(left_turn_mask[idx]),
                'right_turn': bool(right_turn_mask[idx]),
                'tl_id': self.lane_to_tl.get(lane_id, '')
            }

        self.lane_vehicle_ids = lane_vehicle_ids  # Store for later if needed
        return lane_data
    def _get_route_for_lane(self, lane_id, all_vehicles):
        try:
            vehicles = [vid for vid in self.lane_vehicle_ids.get(lane_id, []) if vid in all_vehicles]
            return traci.vehicle.getRouteID(vehicles[0]) if vehicles else ""
        except:
            return ""
    
    def _count_vehicle_classes(self, vehicle_ids):
        counts = defaultdict(int)
        for vid in vehicle_ids:
            try:
                counts[traci.vehicle.getVehicleClass(vid)] += 1
            except: pass
        return counts


    def _detect_priority_vehicles(self, lane_id):
        try:
            return any(traci.vehicle.getVehicleClass(vid) in ['emergency', 'authority']
                    for vid in traci.lane.getLastStepVehicleIDs(lane_id))
        except: return False

    def _update_lane_status_and_score(self, lane_data):
        status = {}
        try:
            for lane_id, data in lane_data.items():
                idx = self.lane_id_to_idx[lane_id]
                norm = lambda x: data[x] / self.norm_bounds[x]
                queue_norm, wait_norm = norm('queue_length'), norm('waiting_time')
                speed_norm, flow_norm = norm('mean_speed'), norm('flow')
                arrival_norm = self._calculate_arrival_rate(lane_id) / self.norm_bounds['arrival_rate']
                composite = (self.adaptive_params['queue_weight'] * queue_norm +
                            self.adaptive_params['wait_weight'] * wait_norm +
                            1 - min(speed_norm, 1.0) + 1 - min(flow_norm, 1.0) +
                            arrival_norm * 0.5)
                if composite > 0.8:
                    stat, delta = "BAD", -max(2, min(8, int(composite * 8)))
                elif composite < 0.3:
                    stat, delta = "GOOD", max(2, min(8, int((1 - composite) * 8)))
                else:
                    stat, delta = "NORMAL", 0
                if self.lane_states[idx] == stat:
                    self.consecutive_states[idx] += 1
                    delta *= min(3.0, 1.0 + self.consecutive_states[idx] * 0.1)
                else:
                    self.lane_states[idx] = stat
                    self.consecutive_states[idx] = 1
                self.lane_scores[idx] += delta
                if stat == "NORMAL":
                    decay = 1.5 if composite < 0.4 else 1.0
                    self.lane_scores[idx] = (max(0, self.lane_scores[idx] - decay) if self.lane_scores[idx] > 0
                                            else min(0, self.lane_scores[idx] + decay) if self.lane_scores[idx] < 0
                                            else 0)
                self.lane_scores[idx] = max(-50, min(50, self.lane_scores[idx]))
                status[lane_id] = stat
        except Exception as e:
            print(f"Error in _update_lane_status_and_score: {e}")
        return status

    def _calculate_arrival_rate(self, lane_id):
        try:
            idx = self.lane_id_to_idx[lane_id]
            now = traci.simulation.getTime()
            curr = set(traci.lane.getLastStepVehicleIDs(lane_id))
            arrivals = curr - self.last_lane_vehicles[idx]
            delta_time = max(1e-3, now - self.last_arrival_time[idx])
            rate = len(arrivals) / delta_time
            self.last_lane_vehicles[idx], self.last_arrival_time[idx] = curr, now
            return rate
        except Exception as e:
            print(f"Error calculating arrival rate for {lane_id}: {e}")
            return 0.0
        
        
    def _select_target_lane(self, tl_id, controlled_lanes, lane_data, current_time):
        """Prefer lanes with vehicles; ignore empty unless all are empty.
        Score: queue*0.5 + wait*0.3 + arrival*0.2 + starvation*0.3 + emergency."""
        # Only consider lanes with vehicles, unless all are empty
        nonempty_lanes = [l for l in controlled_lanes if l in lane_data and lane_data[l]['queue_length'] > 0]
        lanes_to_consider = nonempty_lanes if nonempty_lanes else [l for l in controlled_lanes if l in lane_data]
        if not lanes_to_consider:
            return None

        max_queue = max((lane_data[l]['queue_length'] for l in lanes_to_consider), default=1)
        max_wait = max((lane_data[l]['waiting_time'] for l in lanes_to_consider), default=1)
        max_arr = max((lane_data[l].get('arrival_rate', 0) for l in lanes_to_consider), default=0.1)

        candidates = []
        for lane in lanes_to_consider:
            d = lane_data[lane]
            idx = self.lane_id_to_idx[lane]
            q_score = d['queue_length'] / max_queue if max_queue > 0 else 0
            w_score = d['waiting_time'] / max_wait if max_wait > 0 else 0
            a_score = d.get('arrival_rate', 0) / max_arr if max_arr > 0 else 0
            last_green = self.last_green_time[idx]
            starve = max(0, (current_time - last_green - self.adaptive_params['starvation_threshold']) / 10)
            emerg = 2 if d.get('ambulance') else 0
            total = (0.5 * q_score + 0.3 * w_score + 0.2 * a_score + 0.3 * starve + emerg)
            candidates.append((lane, total))
        # Pick the lane with the highest score
        return max(candidates, key=lambda x: x[1])[0]

    def _get_phase_efficiency(self, tl_id, phase_index):
        """Phase efficiency based on utilization"""
        try:
            total = sum(c for (tl, _), c in self.phase_utilization.items() if tl == tl_id)
            if not total: return 1.0
            count = self.phase_utilization.get((tl_id, phase_index), 0)
            return min(1.0, max(0.1, count/total))
        except: return 1.0

    def _adjust_traffic_lights(self, lane_data, lane_status, current_time):
        """Adjust with priorities"""
        try:
            for tl_id in traci.trafficlight.getIDList():
                try:
                    cl = traci.trafficlight.getControlledLanes(tl_id)
                    for lane in cl: self.lane_to_tl[lane] = tl_id
                    if not self._handle_priority_conditions(tl_id, cl, lane_data, current_time):
                        self._perform_normal_control(tl_id, cl, lane_data, current_time)
                except Exception as e:
                    print(f"Error adjusting traffic light {tl_id}: {e}")
        except Exception as e:
            print(f"Error in _adjust_traffic_lights: {e}")

    def _handle_priority_conditions(self, tl_id, controlled_lanes, lane_data, current_time):
        """Priority: Ambulance then left-turn"""
        amb = [l for l in controlled_lanes if lane_data.get(l, {}).get('ambulance')]
        if amb: return self._handle_ambulance_priority(tl_id, amb, lane_data, current_time)
        left = [l for l in controlled_lanes if lane_data.get(l, {}).get('left_turn') and
                (lane_data[l]['queue_length'] > 3 or lane_data[l]['waiting_time'] > 10)]
        if left: return self._handle_protected_left_turn(tl_id, left, lane_data, current_time)
        return False

    def _handle_ambulance_priority(self, tl_id, controlled_lanes, lane_data, current_time):
        """Serve emergency vehicle lanes with priority"""
        try:
            amb_lanes = [l for l in controlled_lanes if lane_data.get(l, {}).get('ambulance')]
            if not amb_lanes: return False
            min_dist, target = float('inf'), None
            for lane in amb_lanes:
                try:
                    for vid in traci.lane.getLastStepVehicleIDs(lane):
                        if traci.vehicle.getVehicleClass(vid) in ['emergency', 'authority']:
                            d = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(vid)
                            if d < min_dist: min_dist, target = d, lane
                except Exception as e:
                    print(f"Error ambulance lane {lane}: {e}")
            if target is None: return False
            phase = self._find_phase_for_lane(tl_id, target)
            if phase is not None:
                dur = 30 if min_dist < 30 else 20
                traci.trafficlight.setPhase(tl_id, phase)
                traci.trafficlight.setPhaseDuration(tl_id, dur)
                self.ambulance_active[tl_id] = True
                self.ambulance_start_time[tl_id] = current_time
                self.rl_agent.training_data.append({'event':'ambulance_priority','lane_id':target,'tl_id':tl_id,
                    'phase':phase,'simulation_time':current_time,'distance_to_stopline':min_dist,'duration':dur})
                return True
            return False
        except Exception as e:
            print(f"Error in _handle_ambulance_priority: {e}")
            return False

    def _perform_normal_control(self, tl_id, controlled_lanes, lane_data, current_time):
        """Normal RL-based control"""
        try:
            if not isinstance(lane_data, dict):
                print(f"⚠️ lane_data is {type(lane_data)}, expected dict")
                return
            target = self._select_target_lane(tl_id, controlled_lanes, lane_data, current_time)
            if not target: return
            state = self._create_state_vector(target, lane_data)
            if not self.rl_agent.is_valid_state(state): return
            action = self.rl_agent.get_action(state, lane_id=target)
            last_time = self.last_phase_change[tl_id] if isinstance(self.last_phase_change, dict) else 0
            if current_time - last_time >= 5:
                self._execute_control_action(tl_id, target, action, lane_data, current_time)
        except Exception as e:
            print(f"Error in _perform_normal_control: {e}")
            import traceback; traceback.print_exc()
    def _execute_control_action(self, tl_id, target_lane, action, lane_data, current_time):
        """Execute the selected control action (shortened version)"""
        try:
            if not isinstance(lane_data, dict) or target_lane not in lane_data:
                print("⚠️ Invalid lane_data in _execute_control_action")
                return

            current_phase = traci.trafficlight.getPhase(tl_id)
            target_phase = self._find_phase_for_lane(tl_id, target_lane) or current_phase

            if action == 0:  # Set green for target lane
                if current_phase != target_phase:
                    green_time = self._calculate_dynamic_green(lane_data[target_lane]) \
                        if target_lane in lane_data else self.adaptive_params['min_green']
                    traci.trafficlight.setPhase(tl_id, target_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, green_time)
                    self.last_phase_change[tl_id] = current_time
                    self.last_green_time[self.lane_id_to_idx[target_lane]] = current_time
            elif action == 1:  # Next phase
                next_phase = (current_phase + 1) % self._get_phase_count(tl_id)
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
                except Exception as e:
                    print(f"Could not shorten phase: {e}")
            elif action == 4:  # Balanced phase switch
                balanced_phase = self._get_balanced_phase(tl_id, lane_data)
                if balanced_phase != current_phase:
                    traci.trafficlight.setPhase(tl_id, balanced_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                    self.last_phase_change[tl_id] = current_time

            # Update phase utilization stats
            key = (tl_id, traci.trafficlight.getPhase(tl_id))
            self.phase_utilization[key] = self.phase_utilization.get(key, 0) + 1

        except Exception as e:
            print(f"Error in _execute_control_action: {e}")

    def _get_balanced_phase(self, tl_id, lane_data):
        """
        Get phase that maximizes sum(queue*0.8 + wait*0.5) for lanes with vehicles,
        and penalize phases that are green for ONLY empty lanes.
        """
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return 0
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            best_phase, best_score = 0, -float("inf")
            for phase_idx, phase in enumerate(logic.phases):
                phase_score = 0
                green_lanes = [
                    lane for lane_idx, lane in enumerate(controlled_lanes)
                    if lane_idx < len(phase.state) and phase.state[lane_idx].upper() == 'G'
                ]
                has_vehicle = False
                for lane in green_lanes:
                    if lane in lane_data:
                        q = lane_data[lane]['queue_length']
                        w = lane_data[lane]['waiting_time']
                        if q > 0:
                            has_vehicle = True
                        phase_score += q * 0.8 + w * 0.5

                # Heavy penalty if all green lanes are empty
                if green_lanes and not has_vehicle:
                    phase_score -= 100  # <-- make this large to avoid
                if phase_score > best_score:
                    best_score, best_phase = phase_score, phase_idx
            return best_phase
        except Exception as e:
            print(f"Error in _get_balanced_phase: {e}")
            return 0

    def _calculate_dynamic_green(self, lane_data):
        """Calculate dynamic green time based on lane conditions (shortened)"""
        base = self.adaptive_params['min_green']
        queue = min(lane_data['queue_length'] * 0.7, 15)
        density = min(lane_data['density'] * 5, 10)
        bonus = 10 if lane_data.get('ambulance') else 0
        total = base + queue + density + bonus
        return min(max(total, base), self.adaptive_params['max_green'])

    def _find_phase_for_lane(self, tl_id, target_lane):
        """Find phase that gives green to target lane (shortened)"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic: return 0
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
        d = intersection_data[tl_id]
        queues = np.array(d.get('queues', []), dtype=float)
        waits = np.array(d.get('waits', []), dtype=float)
        speeds = np.array(d.get('speeds', []), dtype=float)
        n_phases = float(d.get('n_phases', 4))
        current_phase = float(d.get('current_phase', 0))
        state = np.array([
            queues.max() if queues.size else 0,
            queues.mean() if queues.size else 0,
            speeds.min() if speeds.size else 0,
            speeds.mean() if speeds.size else 0,
            waits.max() if waits.size else 0,
            waits.mean() if waits.size else 0,
            current_phase / max(n_phases - 1, 1), n_phases,
            float(d.get('left_q', 0)),
            float(d.get('right_q', 0))
        ])
        return state

    def _process_rl_learning(self, intersection_data, lane_data, current_time):
        try:
            for tl_id in intersection_data:
                if tl_id not in intersection_data: 
                    continue
                    
                d = intersection_data[tl_id]
                state = self._create_intersection_state_vector(tl_id, intersection_data)
                if not self.rl_agent.is_valid_state(state): 
                    continue
                    
                queues, waits = d['queues'], d['waits']
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                logic = self._get_traffic_light_logic(tl_id)
                current_phase = d['current_phase']
                
                # Calculate empty green penalty
                empty_green_count = 0
                has_vehicle_on_green = False
                if logic:
                    phase_state = logic.phases[current_phase].state
                    for lane_idx, lane in enumerate(controlled_lanes):
                        if (lane_idx < len(phase_state) and phase_state[lane_idx].upper() == 'G'):
                            if lane in self.lane_id_to_idx and d['queues'][lane_idx] == 0:
                                empty_green_count += 1
                            if lane in self.lane_id_to_idx and d['queues'][lane_idx] > 0:
                                has_vehicle_on_green = True
                
                # Calculate congestion bonus
                congestion_bonus = sum(
                    min(self.adaptive_params['congestion_bonus'], q) 
                    for q in queues if q > 5
                )
                
                # Composite reward
                empty_green_penalty = self.adaptive_params['empty_green_penalty'] * empty_green_count
                only_empty_green_penalty = 0
                if not has_vehicle_on_green:
                    only_empty_green_penalty = 100  # make this large

                reward = (
                    -self.adaptive_params['queue_weight'] * sum(queues) 
                    - self.adaptive_params['wait_weight'] * sum(waits) 
                    - empty_green_penalty
                    - only_empty_green_penalty  # strong penalty!
                    + congestion_bonus
                )
                self.rl_agent.reward_history.append(reward)
                reward_components = {
                    "queue_penalty": -self.adaptive_params['queue_weight'] * sum(queues),
                    "wait_penalty": -self.adaptive_params['wait_weight'] * sum(waits),
                    "empty_green_penalty": -self.adaptive_params['empty_green_penalty'] * empty_green_count,
                    "congestion_bonus": congestion_bonus,
                    "total_raw": reward
                }
                
                # Update Q-table if we have previous state
                if tl_id in self.previous_states and tl_id in self.previous_actions:
                    prev_state, prev_action = self.previous_states[tl_id], self.previous_actions[tl_id]
                    self.rl_agent.update_q_table(
                        prev_state, prev_action, reward, state, 
                        tl_id=tl_id, 
                        extra_info={
                            **reward_components,
                            'episode': self.current_episode,
                            'simulation_time': current_time,
                            'action_name': self.rl_agent._get_action_name(prev_action),
                            'queue_length': max(queues) if queues else 0,
                            'waiting_time': max(waits) if waits else 0,
                            'mean_speed': np.mean(d['speeds']) if d['speeds'] else 0,
                            'left_turn': d['left_q'], 'right_turn': d['right_q'],
                            'phase_id': current_phase
                        },
                        action_size=d['n_phases']
                    )
                
                # Store current state/action for next step
                action = self.rl_agent.get_action(state, tl_id=tl_id)
                self.previous_states[tl_id], self.previous_actions[tl_id] = state, action
                
        except Exception as e:
            print(f"Error in _process_rl_learning: {e}")
            traceback.print_exc()

    def _create_state_vector(self, lane_id, lane_data):
        try:
            if not (isinstance(lane_data, dict) and lane_id in lane_data):
                return np.zeros(self.rl_agent.state_size)
            d = lane_data[lane_id]
            tl_id = self.lane_to_tl.get(lane_id, "")
            norm = lambda x, b: min(x / b, 1.0)
            qn, wn, dn, sn, fn = norm(d['queue_length'], self.norm_bounds['queue']), norm(d['waiting_time'], self.norm_bounds['wait']), norm(d['density'], self.norm_bounds['density']), norm(d['mean_speed'], self.norm_bounds['speed']), norm(d['flow'], self.norm_bounds['flow'])
            d['arrival_rate'] = d.get('arrival_rate', self._calculate_arrival_rate(lane_id))
            an = norm(d['arrival_rate'], self.norm_bounds['arrival_rate'])
            rqn = norm(d.get('queue_route', 0), self.norm_bounds['queue'] * 3)
            rfn = norm(d.get('flow_route', 0), self.norm_bounds['flow'] * 3)
            current_phase, phase_norm, phase_eff = 0, 0.0, 0.0
            if tl_id:
                try:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    num_phases = self._get_phase_count(tl_id)
                    phase_norm = current_phase / max(num_phases-1, 1)
                    phase_eff = self._get_phase_efficiency(tl_id, current_phase)
                except: pass
            last_green = self.last_green_time[self.lane_id_to_idx[lane_id]]
            tsg = norm(traci.simulation.getTime() - last_green, self.norm_bounds['time_since_green'])
            state = np.array([
                qn, wn, dn, sn, fn, rqn, rfn, phase_norm, tsg,
                float(d['ambulance']), self.lane_scores[self.lane_id_to_idx[lane_id]] / 100, phase_eff
            ])
            return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"Error creating state vector for {lane_id}: {e}")
            return np.zeros(self.rl_agent.state_size)

    def _calculate_reward(self, lane_id, lane_data, action_taken, current_time):
        try:
            if not (isinstance(lane_data, dict) and lane_id in lane_data):
                return 0.0, {}, 0.0
            d = lane_data[lane_id]
            lt_factor = 1.5 if d['left_turn'] else 1.0
            qp = -min(d['queue_length'] * self.adaptive_params['queue_weight'] * lt_factor, 30)
            wp = -min(d['waiting_time'] * self.adaptive_params['wait_weight'] * lt_factor, 20)
            tr = min(d['flow'] * self.adaptive_params['flow_weight'], 25)
            sr = min(d['mean_speed'] * self.adaptive_params['speed_weight'], 15)
            ltb = 15 if d['left_turn'] and action_taken == 0 and d['queue_length'] < 2 else 0
            ab = min(d['queue_length'] * 0.7, 20) if action_taken == 0 and d['queue_length'] > 5 else 0
            lg = self.last_green_time[self.lane_id_to_idx[lane_id]]
            sp = -min(30, (current_time-lg-self.adaptive_params['starvation_threshold'])*0.5) if current_time-lg > self.adaptive_params['starvation_threshold'] else 0
            ambb = 25 if d['ambulance'] else 0
            eb = 15 if d['queue_length'] < 3 and d['mean_speed'] > 5 else 0
            total = qp + wp + tr + sr + ab + sp + ambb + eb
            norm_reward = np.clip(total / self.adaptive_params['reward_scale'], -1.0, 1.0)
            if np.isnan(norm_reward) or np.isinf(norm_reward): norm_reward = 0.0
            comps = {'queue_penalty': qp, 'wait_penalty': wp, 'throughput_reward': tr, 'speed_reward': sr, 'action_bonus': ab, 'starvation_penalty': sp, 'ambulance_bonus': ambb, 'total_raw': total, 'normalized': norm_reward}
            return norm_reward, comps, total
        except Exception as e:
            print(f"Error calculating reward for {lane_id}: {e}")
            return 0.0, {}, 0.0

    def end_episode(self):
    # Update adaptive parameters based on average reward
        if self.rl_agent.reward_history:
            avg_reward = np.mean(self.rl_agent.reward_history)
            print(f"Average reward this episode: {avg_reward:.2f}")
            if avg_reward < -10:
                self.adaptive_params['min_green'] = min(self.adaptive_params['min_green'] + 1, self.adaptive_params['max_green'])
                self.adaptive_params['max_green'] = min(self.adaptive_params['max_green'] + 5, 120)
            elif avg_reward > -2:
                self.adaptive_params['min_green'] = max(self.adaptive_params['min_green'] - 1, 5)
                self.adaptive_params['max_green'] = max(self.adaptive_params['max_green'] - 5, 30)
        else:
            print("No reward history for adaptive param update.")
        self.rl_agent.adaptive_params = self.adaptive_params.copy()
        print("🔄 Updated adaptive parameters:", self.adaptive_params)
        self.rl_agent.save_model(adaptive_params=self.adaptive_params)

        # --- Epsilon decay: add this line ---
        old_epsilon = self.rl_agent.epsilon
        self.rl_agent.epsilon = max(self.rl_agent.epsilon * self.rl_agent.epsilon_decay, self.rl_agent.min_epsilon)
        print(f"🚦 Epsilon after training/episode: {old_epsilon} -> {self.rl_agent.epsilon}")

        self.previous_states.clear()
        self.previous_actions.clear()
        self.rl_agent.reward_history.clear()
        

    def _update_adaptive_parameters(self, performance_stats):
        try:
            avg_reward = performance_stats.get('avg_reward', 0)
            if avg_reward > 0.6:
                self.adaptive_params['min_green'] = min(15, self.adaptive_params['min_green'] + 1)
                self.adaptive_params['max_green'] = min(90, self.adaptive_params['max_green'] + 5)
            elif avg_reward < 0.3:
                self.adaptive_params['min_green'] = max(5, self.adaptive_params['min_green'] - 1)
                self.adaptive_params['max_green'] = max(30, self.adaptive_params['max_green'] - 5)
            print("🔄 Updated adaptive parameters:", self.adaptive_params)
        except Exception as e:
            print(f"Error updating adaptive parameters: {e}")


def start_universal_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1, num_retries=1, retry_delay=1, mode="train"):
    global controller  # Only if you want to set the top-level controller for API
    try:
        for episode in range(episodes):
            print(f"\n{'='*50}\n🚦 STARTING UNIVERSAL EPISODE {episode + 1}/{episodes}\n{'='*50}")
            sumo_binary = "sumo-gui" if use_gui else "sumo"
            sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary), '-c', sumocfg_path, '--start', '--quit-on-end']
            traci.start(sumo_cmd)
            controller = UniversalSmartTrafficController(sumocfg_path=sumocfg_path, mode=mode)
            controller.initialize()
            controller.current_episode = episode + 1
            step, tstart = 0, time.time()
            while traci.simulation.getMinExpectedNumber() > 0:
                if max_steps and step >= max_steps:
                    print(f"Reached max steps ({max_steps}), ending episode."); break
                controller.run_step(); traci.simulationStep(); step += 1
                if step % 200 == 0:
                    print(f"Episode {episode + 1}: Step {step} completed, elapsed: {time.time()-tstart:.2f}s")
            print(f"Episode {episode + 1} completed after {step} steps")
            controller.end_episode(); traci.close()
            if episode < episodes - 1: time.sleep(2)
        print(f"\n🎉 All {episodes} episodes completed!")
        return controller  # Return the last controller
    except Exception as e:
        print(f"Error in universal simulation: {e}")
    finally:
        try: traci.close()
        except: pass
        print("Simulation resources cleaned up")

# ... [Flask API placeholder code unchanged, do not use global at top-level] ...

def main():
    global controller  # <-- only inside main
    parser = argparse.ArgumentParser(description="Run universal SUMO RL traffic simulation")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of retries if connection fails')
    parser.add_argument('--retry-delay', type=int, default=1, help='Delay in seconds between retries')
    parser.add_argument('--mode', choices=['train', 'eval', 'adaptive'], default='train',
                        help='Controller mode: train (explore+learn), eval (exploit only), adaptive (exploit+learn)')
    parser.add_argument('--api', action='store_true', help='Start API server instead of simulation')
    args = parser.parse_args()

    if args.api:
        print("Starting API server...")
        controller = None  # Or load/init as needed for API
        app.run(port=5000, debug=True)
        return

    profiler = Profiler()
    profiler.start()

    # Start your simulation and set global controller for API use
    controller = start_universal_simulation(args.sumo, args.gui, args.max_steps, args.episodes, args.num_retries, args.retry_delay, mode=args.mode)

    profiler.stop()
    profiler.open_in_browser() 

if __name__ == "__main__":
    main()