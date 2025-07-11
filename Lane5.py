from collections import defaultdict
import os, sys, traci, warnings, time,traceback,pickle,datetime
import pandas as np
warnings.filterwarnings('ignore')

# SUMO_HOME setup
os.environ.setdefault('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path: sys.path.append(tools)

def subscribe_lanes(lane_ids):
    for lid in lane_ids:
        traci.lane.subscribe(lid, [
            traci.constants.LAST_STEP_VEHICLE_NUMBER,
            traci.constants.LAST_STEP_MEAN_SPEED,
            traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
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
            'queue_weight': 0.6, 'wait_weight': 0.3, 'flow_weight': 0.5, 'speed_weight': 0.2, 'left_turn_priority': 1.2
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
            self.consecutive_no_improvement = (
                self.consecutive_no_improvement + 1 if recent_avg <= older_avg else 0
            )
            if self.consecutive_no_improvement > self.max_no_improvement:
                self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
                self.consecutive_no_improvement = 0

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

        # Lane and intersection management
        self.left_turn_lanes = set()
        self.right_turn_lanes = set()
        self.lane_id_list = []  # Initialize appropriately elsewhere
        self.lane_id_to_idx = {}
        self.idx_to_lane_id = {}
        self.lane_to_tl = {}
        self.tl_action_sizes = {}
        self.last_green_time = np.zeros(len(self.lane_id_list))
        self.pending_next_phase = {}  # Dict: tl_id -> (next_phase, set_time)

        # Traffic light and intersection data
        self.intersection_data = {}
        self.tl_logic_cache = {}
        self.phase_utilization = defaultdict(int)
        self.last_phase_change = {}

        # Ambulance and phase tracking
        self.ambulance_active = defaultdict(bool)
        self.ambulance_start_time = defaultdict(float)
        self.left_phase_counter = defaultdict(int)

        # RL agent & memory
        self.previous_states = {}
        self.previous_actions = {}
        self.rl_agent = EnhancedQLearningAgent(state_size=10, action_size=8, mode=mode)
        self.rl_agent.load_model()

        # Adaptive parameters
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
    def initialize(self):
    # Log traffic light phases
        for tl_id in traci.trafficlight.getIDList():
            phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            print(f"Traffic light {tl_id} phases:")
            for i, phase in enumerate(phases):
                print(f"  Phase {i}: {phase.state} (duration {getattr(phase, 'duration', '?')})")
        self.lane_id_list = traci.lane.getIDList()
        self.tl_action_sizes = {tl_id: len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                                for tl_id in traci.trafficlight.getIDList()}
        self.lane_id_to_idx = {lid: i for i, lid in enumerate(self.lane_id_list)}
        self.idx_to_lane_id = dict(enumerate(self.lane_id_list))
        self.last_green_time = np.zeros(len(self.lane_id_list))
        subscribe_lanes(self.lane_id_list)
        self.left_turn_lanes, self.right_turn_lanes = detect_turning_lanes_with_traci()
        print(f"Auto-detected left-turn lanes: {sorted(self.left_turn_lanes)}")
        print(f"Auto-detected right-turn lanes: {sorted(self.right_turn_lanes)}")

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
        """Calculate adaptive yellow duration based on max speed on controlled lanes."""
        try:
            max_speed = max(
                (traci.vehicle.getSpeed(vid) for lane in controlled_lanes
                for vid in traci.lane.getLastStepVehicleIDs(lane)),
                default=max((traci.lane.getMaxSpeed(lane) for lane in controlled_lanes), default=0)
            )
            yellow_time = max_speed / 6.0 + 1.0  # v/(2*3.0) + 1.0, decel=3.0
            return max(2.5, min(6.0, yellow_time))
        except Exception as e:
            print(f"Error in _calculate_adaptive_yellow: {e}")
            return 3.0

    def run_step(self):
        try:
            self.step_count += 1
            current_time = traci.simulation.getTime()
            self.intersection_data = {}
            lane_data = self._collect_enhanced_lane_data()
            for tl_id in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                logic = self._get_traffic_light_logic(tl_id)
                current_phase = traci.trafficlight.getPhase(tl_id)
                if tl_id in self.pending_next_phase:
                    pending_phase, set_time = self.pending_next_phase[tl_id]
                    phase_duration = logic.phases[current_phase].duration if logic else 3
                    if current_time - set_time >= phase_duration - 0.1:
                        traci.trafficlight.setPhase(tl_id, pending_phase)
                        traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                        self.last_phase_change[tl_id] = current_time
                        del self.pending_next_phase[tl_id]
                    continue
                if self._handle_ambulance_priority(tl_id, controlled_lanes, lane_data, current_time): continue
                if self._handle_protected_left_turn(tl_id, controlled_lanes, lane_data, current_time): continue
                queues = [lane_data[l]['queue_length'] for l in controlled_lanes if l in lane_data]
                waits = [lane_data[l]['waiting_time'] for l in controlled_lanes if l in lane_data]
                speeds = [lane_data[l]['mean_speed'] for l in controlled_lanes if l in lane_data]
                left_q = sum(lane_data[l]['queue_length'] for l in controlled_lanes if l in self.left_turn_lanes and l in lane_data)
                right_q = sum(lane_data[l]['queue_length'] for l in controlled_lanes if l in self.right_turn_lanes and l in lane_data)
                n_phases = self.tl_action_sizes[tl_id]
                self.intersection_data[tl_id] = dict(queues=queues, waits=waits, speeds=speeds, left_q=left_q, right_q=right_q, n_phases=n_phases, current_phase=current_phase)
                most_starved_lane, max_starve_time = None, 0
                for lane in controlled_lanes:
                    idx = self.lane_id_to_idx.get(lane)
                    if idx is not None:
                        time_since_green = current_time - self.last_green_time[idx]
                        if time_since_green > self.adaptive_params['starvation_threshold'] and time_since_green > max_starve_time:
                            most_starved_lane, max_starve_time = lane, time_since_green
                if most_starved_lane:
                    starved_phase = self._find_phase_for_lane(tl_id, most_starved_lane)
                    if current_phase != starved_phase:
                        switched = self._switch_phase_with_yellow_if_needed(
                            tl_id, current_phase, starved_phase, logic, controlled_lanes, lane_data, current_time)
                        if not switched:
                            traci.trafficlight.setPhase(tl_id, starved_phase)
                            traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                            self.last_phase_change[tl_id] = current_time
                    self.last_green_time[self.lane_id_to_idx[most_starved_lane]] = current_time
                    continue
                state = self._create_intersection_state_vector(tl_id, self.intersection_data)
                action = self.rl_agent.get_action(state, tl_id, action_size=n_phases)
                last_change = self.last_phase_change.get(tl_id, -9999)
                if current_time - last_change >= self.adaptive_params['min_green'] and action != current_phase:
                    if not self._is_in_dilemma_zone(tl_id, controlled_lanes, lane_data):
                        switched = self._switch_phase_with_yellow_if_needed(
                            tl_id, current_phase, action, logic, controlled_lanes, lane_data, current_time)
                        if not switched:
                            traci.trafficlight.setPhase(tl_id, action)
                            traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                            self.last_phase_change[tl_id] = current_time
            self._process_rl_learning(self.intersection_data, current_time)
        except Exception as e:
            print(f"Error in run_step: {e}")
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

    def _collect_enhanced_lane_data(self):
        lane_data = {}
        for lane_id in self.lane_id_list:
            try:
                results = traci.lane.getSubscriptionResults(lane_id)
                if not results:
                    continue
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                def safe_get_class(vid):
                    try:
                        return traci.vehicle.getVehicleClass(vid)
                    except Exception:
                        return None
                vehicle_count = results.get(traci.constants.LAST_STEP_VEHICLE_NUMBER, 0)
                mean_speed = max(results.get(traci.constants.LAST_STEP_MEAN_SPEED, 0), 0.0)
                queue_length = results.get(traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                waiting_time = results.get(traci.constants.VAR_ACCUMULATED_WAITING_TIME, 0)
                lane_length = traci.lane.getLength(lane_id) or 1.0

                # Protect vehicle_classes too
                def safe_count_classes(vehicle_ids):
                    from collections import defaultdict
                    counts = defaultdict(int)
                    for vid in vehicle_ids:
                        vclass = safe_get_class(vid)
                        if vclass: counts[vclass] += 1
                    return counts

                lane_data[lane_id] = {
                    'queue_length': queue_length,
                    'waiting_time': waiting_time,
                    'density': vehicle_count / lane_length,
                    'mean_speed': mean_speed,
                    'flow': vehicle_count,
                    'lane_id': lane_id,
                    'edge_id': traci.lane.getEdgeID(lane_id),
                    'route_id': self._get_route_for_lane(lane_id),
                    'ambulance': any(safe_get_class(vid) == 'emergency' for vid in vehicle_ids),
                    'vehicle_classes': safe_count_classes(vehicle_ids),
                    'left_turn': lane_id in self.left_turn_lanes,
                    'right_turn': lane_id in self.right_turn_lanes,
                    'tl_id': self.lane_to_tl.get(lane_id, '')
                }
            except Exception as e:
                print(f"⚠️ Error collecting data for lane {lane_id}: {e}")
        return lane_data
    def _count_vehicle_classes(self, vehicle_ids):
        counts = defaultdict(int)
        for vid in vehicle_ids:
            try:
                counts[traci.vehicle.getVehicleClass(vid)] += 1
            except: pass
        return counts

    def _get_route_for_lane(self, lane_id):
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            return traci.vehicle.getRouteID(vehicles[0]) if vehicles else ""
        except:
            return ""

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
        """Lane selection with multi-factor scoring"""
        maxes = {'queue': 1, 'wait': 1, 'arr': 0.1}
        for lane in controlled_lanes:
            if lane in lane_data:
                d = lane_data[lane]
                maxes['queue'] = max(maxes['queue'], d['queue_length'])
                maxes['wait'] = max(maxes['wait'], d['waiting_time'])
                d['arrival_rate'] = d.get('arrival_rate', self._calculate_arrival_rate(lane))
                maxes['arr'] = max(maxes['arr'], d['arrival_rate'])

        candidates = []
        for lane in controlled_lanes:
            if lane in lane_data and lane in self.lane_id_to_idx:
                d = lane_data[lane]
                idx = self.lane_id_to_idx[lane]
                score = self.lane_scores[idx]
                q = d['queue_length'] / maxes['queue'] * 10
                w = d['waiting_time'] / maxes['wait'] * 5
                a = d.get('arrival_rate', 0) / maxes['arr'] * 8
                last_green = self.last_green_time[idx]
                starve = max(0, (current_time-last_green-self.adaptive_params['starvation_threshold']) * 0.3) \
                    if current_time-last_green > self.adaptive_params['starvation_threshold'] else 0
                emerg = 20 if d.get('ambulance') else 0
                phase = traci.trafficlight.getPhase(tl_id)
                eff = self._get_phase_efficiency(tl_id, phase) * 5
                total = score + q + w + a + min(15, starve) + emerg + eff
                candidates.append((lane, total))
        if not candidates: return None
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
        """Get phase that balances traffic flow (shortened)"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic: return 0

            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            best_phase, best_score = 0, -1
            for phase_idx, phase in enumerate(logic.phases):
                score = sum(
                    (lane_data[lane]['queue_length'] * 0.5 +
                    lane_data[lane]['waiting_time'] * 0.1 +
                    (1 - min(lane_data[lane]['mean_speed'] / self.norm_bounds['speed'], 1)) * 10)
                    for lane_idx, lane in enumerate(controlled_lanes)
                    if lane in lane_data and lane_idx < len(phase.state) and phase.state[lane_idx].upper() == 'G'
                )
                if score > best_score:
                    best_score, best_phase = score, phase_idx
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
        queues, waits, speeds = d.get('queues', []), d.get('waits', []), d.get('speeds', [])
        n_phases, current_phase = d.get('n_phases', 4), d.get('current_phase', 0)
        state = np.array([
            max(queues) if queues else 0, np.mean(queues) if queues else 0,
            min(speeds) if speeds else 0, np.mean(speeds) if speeds else 0,
            max(waits) if waits else 0, np.mean(waits) if waits else 0,
            float(current_phase) / max(n_phases-1, 1), float(n_phases),
            d.get('left_q', 0), d.get('right_q', 0)
        ])
        return state

    def _process_rl_learning(self, intersection_data, current_time):
        try:
            for tl_id in traci.trafficlight.getIDList():
                if tl_id not in intersection_data: continue
                d = intersection_data[tl_id]
                state = self._create_intersection_state_vector(tl_id, intersection_data)
                if not self.rl_agent.is_valid_state(state): continue
                queues, waits = d.get('queues', []), d.get('waits', [])
                reward = -0.7 * sum(queues) - 0.3 * sum(waits)
                reward_components = {"queue_penalty": -0.7 * sum(queues), "wait_penalty": -0.3 * sum(waits), "total_raw": reward, "normalized": reward}
                if tl_id in self.previous_states and tl_id in self.previous_actions:
                    prev_state, prev_action = self.previous_states[tl_id], self.previous_actions[tl_id]
                    state_key = self.rl_agent._state_to_key(prev_state, tl_id)
                    q_value = float(self.rl_agent.q_table[state_key][prev_action]) if state_key in self.rl_agent.q_table else None
                    log_entry = {
                        'episode': getattr(self, 'current_episode', 0), 'simulation_time': current_time,
                        'action': prev_action, 'action_name': self.rl_agent._get_action_name(prev_action),
                        'reward': reward, 'raw_reward': reward, 'q_value': q_value,
                        'queue_length': max(queues) if queues else 0,
                        'waiting_time': max(waits) if waits else 0,
                        'mean_speed': np.mean(d.get('speeds', [])) if d.get('speeds', []) else 0,
                        'left_turn': d.get('left_q', 0), 'right_turn': d.get('right_q', 0),
                        'tl_id': tl_id, 'phase_id': d.get('current_phase', 0),
                        'epsilon': self.rl_agent.epsilon, 'learning_rate': self.rl_agent.learning_rate,
                        'reward_components': reward_components, 'adaptive_params': self.adaptive_params.copy()
                    }
                    self.rl_agent.update_q_table(prev_state, prev_action, reward, state, tl_id=tl_id, extra_info=log_entry, action_size=d.get('n_phases', 0))
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
        print(f"🚦 Epsilon after training/episode: {self.rl_agent.epsilon}")
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
    except Exception as e:
        print(f"Error in universal simulation: {e}")
    finally:
        try: traci.close()
        except: pass
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
    start_universal_simulation(args.sumo, args.gui, args.max_steps, args.episodes, args.num_retries, args.retry_delay, mode=args.mode)