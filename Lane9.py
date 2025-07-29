import os, sys, traci, threading, warnings, time, argparse, pickle, datetime, logging
import numpy as np
warnings.filterwarnings('ignore')
from traffic_light_display import TrafficLightPhaseDisplay
from collections import defaultdict
from api_phase_client import post_traffic_to_api, get_phases_from_api, create_new_phase_in_api

logging.basicConfig(level=logging.DEBUG)  # Use DEBUG for more verbose output
logger = logging.getLogger("controller")

os.environ.setdefault('SUMO_HOME', r'C:\\Program Files (x86)\\Eclipse\\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

class APIBasedAdaptiveController:
    def __init__(self, tls_id, lane_ids, display=None):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.phases = []
        self.phase_map = {}
        self.current_phase_idx = 0
        self.min_green = 10
        self.max_green = 60
        self.display = display

    def initialize(self):
        self.update_phases()

    def collect_traffic_data(self):
        traffic_data = []
        for lane_id in self.lane_ids:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                wait = traci.lane.getWaitingTime(lane_id)
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                logger.debug(f"[COLLECT] {lane_id}: queue={queue}, wait={wait}, speed={speed}")
                traffic_data.append({
                    "lane_id": lane_id,
                    "queue": queue,
                    "wait": wait,
                    "speed": speed
                })
            except traci.TraCIException as e:
                logger.warning(f"[COLLECT] TraCIException for {lane_id}: {e}")
                traffic_data.append({
                    "lane_id": lane_id,
                    "queue": 0,
                    "wait": 0,
                    "speed": 0
                })
        return traffic_data

    def update_phases(self):
        expected_state_length = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
        traffic_data = self.collect_traffic_data()
        logger.debug(f"[API] Posting traffic data to API for {self.tls_id}, expected_state_length={expected_state_length}")
        self.phases = post_traffic_to_api(
            self.tls_id,
            traffic_data,
            expected_state_length=expected_state_length
        )
        logger.debug(f"[API] Received {len(self.phases)} phases from API: "
                     f"{[p['state'] for p in self.phases]}")
        self._build_phase_map()

    def _build_phase_map(self):
        self.phase_map = {}
        for idx, phase in enumerate(self.phases):
            self.phase_map[phase['state']] = idx

    def get_phase_by_state(self, state_str):
        return self.phase_map.get(state_str)

    def create_new_phase(self, state_str, duration=None):
        if duration is None:
            duration = self.min_green
        new_phase = create_new_phase_in_api(self.tls_id, state_str, duration)
        if new_phase:
            self.phases.append(new_phase)
            self.phase_map[state_str] = len(self.phases) - 1
            logger.info(f"[API] Created new phase: idx={len(self.phases)-1}, state={state_str}, duration={duration}")
            return len(self.phases) - 1
        return None

    def set_phase(self, phase_idx):
        logger.debug(f"[SET_PHASE] Trying to set phase_idx={phase_idx}, total phases={len(self.phases)}")
        if 0 <= phase_idx < len(self.phases):
            phase = self.phases[phase_idx]
            try:
                traci.trafficlight.setPhase(self.tls_id, phase_idx)
                traci.trafficlight.setPhaseDuration(self.tls_id, phase['duration'])
                self.current_phase_idx = phase_idx
                logger.info(f"[SET_PHASE] Set phase {phase_idx} ({phase['state']}) duration={phase['duration']}")
                if self.display:
                    now = traci.simulation.getTime()
                    next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
                    self.display.update_phase_duration(
                        phase_idx,
                        duration=phase['duration'],
                        current_time=now,
                        next_switch_time=next_switch
                    )
                return True
            except traci.TraCIException as e:
                logger.error(f"Error setting phase: {e}")
        else:
            logger.error(f"[SET_PHASE] Phase index {phase_idx} out of range [0,{len(self.phases)-1}]")
        return False

    def create_phase_state(self, green_lanes=None, yellow_lanes=None, red_lanes=None):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        state = ['r'] * len(controlled_lanes)
        def set_lanes(lanes, state_char):
            if not lanes: return
            for lane in lanes:
                if lane in controlled_lanes:
                    idx = controlled_lanes.index(lane)
                    state[idx] = state_char
        set_lanes(green_lanes, 'G')
        set_lanes(yellow_lanes, 'y')
        set_lanes(red_lanes, 'r')
        s = "".join(state)
        logger.debug(f"[PHASE_STATE] green={green_lanes} yellow={yellow_lanes} red={red_lanes} -> {s}")
        return s

class EnhancedQLearningAgent:
    def __init__(self, state_size, action_size, adaptive_controller, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 q_table_file="enhanced_q_table.pkl", mode="train", adaptive_params=None):
        self.state_size, self.action_size = state_size, action_size
        self.learning_rate, self.discount_factor = learning_rate, discount_factor
        self.epsilon, self.epsilon_decay, self.min_epsilon = epsilon, epsilon_decay, min_epsilon
        self.q_table, self.training_data = {}, []
        self.q_table_file, self._loaded_training_count = q_table_file, 0
        self.reward_history = []
        self.mode = mode
        
        self.adaptive_params = adaptive_params or {
            'min_green': 30, 'max_green': 80, 'starvation_threshold': 40, 'reward_scale': 40,
            'queue_weight': 0.6, 'wait_weight': 0.3, 'flow_weight': 0.5, 'speed_weight': 0.2, 
            'left_turn_priority': 1.2, 'empty_green_penalty': 15, 'congestion_bonus': 10
        }
        self.adaptive_controller = adaptive_controller
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
        pass


    def switch_or_extend_phase(self, state, green_lanes, force_protected_left=False):
        print(f"[DEBUG][RL Agent] switch_or_extend_phase called with state={state}, green_lanes={green_lanes}, force_protected_left={force_protected_left}")
        R = self.adaptive_controller.calculate_reward()
        raw_delta_t, delta_t, penalty = self.adaptive_controller.calculate_delta_t_and_penalty(R)
        print(f"[DEBUG][RL Agent] R={R}, delta_t={delta_t}, penalty={penalty}")
        if force_protected_left and len(green_lanes) == 1:
            phase_idx = self.adaptive_controller.create_or_extend_protected_left_phase(green_lanes[0], delta_t)
            rl_phase_type = "protected_left"
        else:
            phase_idx = self.adaptive_controller.create_or_extend_phase(green_lanes, delta_t)
            rl_phase_type = "general"
        self.adaptive_controller._log_apc_event({
            "action": "rl_phase_request",
            "rl_phase_type": rl_phase_type,
            "requested_green_lanes": green_lanes,
            "phase_idx": phase_idx,
            "delta_t": delta_t,
            "penalty": penalty,
            "state": str(state),
        })
        print(f"[DEBUG][RL Agent] Will now set phase from PKL: phase_idx={phase_idx}")
        self.adaptive_controller.set_phase_from_pkl(phase_idx)
        phase_record = self.adaptive_controller.load_phase_from_supabase(phase_idx)
        if phase_record:
            traci.trafficlight.setPhase(self.adaptive_controller.tls_id, phase_record["phase_idx"])
            traci.trafficlight.setPhaseDuration(self.adaptive_controller.tls_id, phase_record["duration"])
            self.adaptive_controller.update_display(phase_record["phase_idx"], phase_record["duration"])
            print(f"[APC-RL] RL agent set phase {phase_record['phase_idx']} duration={phase_record['duration']} (from PKL)")
        else:
            print(f"[APC-RL] No PKL record for RL agent phase {phase_idx}, fallback to default.")
            traci.trafficlight.setPhase(self.adaptive_controller.tls_id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.adaptive_controller.tls_id, self.adaptive_controller.max_green)
            self.adaptive_controller.update_display(phase_idx, self.adaptive_controller.max_green)
        return phase_idx
    
    def switch_to_best_phase(self, state, lane_data, left_turn_lanes=None):
        """
        New method: RL agent picks best phase (possibly protected left) based on state and lane_data.
        Prioritizes blocked/queued left-turns (as protected left), else chooses the lane with largest queue.
        """
        # 1. Prioritize protected left turn if severely blocked
        if left_turn_lanes:
            for left_lane in left_turn_lanes:
                # Consider "blocked" if queue is high and (optionally) all vehicles stopped
                queue = lane_data[left_lane]['queue_length']
                vehicles = lane_data[left_lane]['vehicle_ids']
                speeds = [traci.vehicle.getSpeed(vid) for vid in vehicles] if vehicles else []
                # Threshold: queue >= 2 and all speeds low
                if queue >= 2 and (not speeds or max(speeds) < 0.2):
                    print(f"[RL] Detected blocked left {left_lane}, requesting protected left phase.")
                    return self.switch_or_extend_phase(state, [left_lane], force_protected_left=True)

        # 2. Otherwise: select the lane with highest queue for green
        max_q = -1
        best_lane = None
        for lane, data in lane_data.items():
            if data.get('queue_length', 0) > max_q:
                max_q = data['queue_length']
                best_lane = lane

        if best_lane is not None:
            print(f"[RL] Selecting lane {best_lane} with queue {max_q} for green phase.")
            return self.switch_or_extend_phase(state, [best_lane])

        # 3. Fallback: use all lanes (should not usually happen)
        print("[RL] No valid lane found, defaulting to all green.")
        all_lanes = list(lane_data.keys())
        return self.switch_or_extend_phase(state, all_lanes)
    
    def update_display(self, phase_idx, new_duration):
        now = traci.simulation.getTime()
        next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
        if hasattr(self, "display"):
            self.display.update_phase_duration(
                phase_idx,
                duration=new_duration,
                current_time=now,
                next_switch_time=next_switch
            )
    
    def create_or_extend_protected_left_phase(self, left_lane, delta_t):
        """
        Create or extend a protected left turn phase for the given lane.
        If such a phase already exists, extend its duration using delta_t.
        If not, create a new protected left phase and set its duration.
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        left_idx = controlled_lanes.index(left_lane)
        protected_state = ''.join(['G' if i == left_idx else 'r' for i in range(len(controlled_lanes))])
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]

        # Check if protected left phase already exists
        for idx, phase in enumerate(logic.getPhases()):
            if phase.state == protected_state:
                # Extend phase duration
                old_duration = phase.duration
                new_duration = np.clip(old_duration + delta_t, self.min_green, self.max_green)
                phases = list(logic.getPhases())
                phases[idx] = traci.trafficlight.Phase(new_duration, protected_state)
                new_logic = traci.trafficlight.Logic(
                    logic.programID, logic.type, logic.currentPhaseIndex, phases
                )
                traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
                traci.trafficlight.setPhase(self.tls_id, idx)
                self.save_phase_to_supabase(idx, new_duration, protected_state, delta_t, delta_t, penalty=0)
                self._log_apc_event({
                    "action": "extend_protected_left_phase",
                    "tls_id": self.tls_id,
                    "lane_id": left_lane,
                    "phase_idx": idx,
                    "new_duration": new_duration,
                    "delta_t": delta_t,
                    "protected_state": protected_state,
                })
                return idx

        # If not found, create new protected left phase
        yellow_state = ''.join(['y' if i == left_idx else 'r' for i in range(len(controlled_lanes))])
        green_phase = traci.trafficlight.Phase(self.max_green, protected_state)
        yellow_phase = traci.trafficlight.Phase(3, yellow_state)
        phases = list(logic.getPhases()) + [green_phase, yellow_phase]
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type, len(phases) - 2, phases
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
        traci.trafficlight.setPhase(self.tls_id, len(phases) - 2)
        self.save_phase_to_supabase(len(phases) - 2, self.max_green, protected_state, delta_t, delta_t, penalty=0)
        self._log_apc_event({
            "action": "create_protected_left_phase",
            "tls_id": self.tls_id,
            "lane_id": left_lane,
            "phase_idx": len(phases) - 2,
            "duration": self.max_green,
            "delta_t": delta_t,
            "protected_state": protected_state,
        })
        return len(phases) - 2


    def calculate_total_reward(self, adaptive_R, rl_reward):
        """Sum of Adaptive reward R (no penalty) and RL-specific reward."""
        return adaptive_R + rl_reward

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

class UniversalSmartTrafficController:
    def __init__(self, sumocfg_path=None, mode="train"):
        self.mode = mode
        self.lane_id_list = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        self.tls_id = traci.trafficlight.getIDList()[0]
        self.last_phase_change = 0.0
        self.phase_event_log_file = f"phase_event_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.phase_events = []
        self.rl_agent = None

        # --- TrafficLightPhaseDisplay integration ---
        self.display = TrafficLightPhaseDisplay(self.phase_events, poll_interval=500)
        self.display.start()
        self.apc = APIBasedAdaptiveController(self.tls_id, self.lane_id_list, display=self.display)

    def log_phase_event(self, event: dict):
        event["timestamp"] = datetime.datetime.now().isoformat()
        self.phase_events.append(event)
        try:
            with open(self.phase_event_log_file, "wb") as f:
                pickle.dump(self.phase_events, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"[WARN] Could not write phase_events to file: {e}")

    def initialize(self):
        self.apc.initialize()
        state_size = 6
        action_size = len(self.apc.phases) or 1
        self.rl_agent = EnhancedQLearningAgent(
            state_size=state_size,
            action_size=action_size,
            adaptive_controller=self.apc,
            mode=self.mode
        )

    def run_step(self):
        logger.debug("[RUN_STEP] Updating phases")
        self.apc.update_phases()
        state = self._create_state_vector()
        logger.debug(f"[RUN_STEP] RL state: {state}")
        phase_idx = self.rl_agent.get_action(
            state,
            tl_id=self.tls_id,
            action_size=len(self.apc.phases)
        )
        logger.debug(f"[RUN_STEP] RL agent chose phase_idx={phase_idx}")
        if self.apc.set_phase(phase_idx):
            self.log_phase_event({
                "action": "phase_set",
                "phase_idx": phase_idx,
                "duration": self.apc.phases[phase_idx]['duration'],
                "timestamp": datetime.datetime.now().isoformat()
            })
        self.handle_special_cases()

    def _create_state_vector(self):
        traffic = self.apc.collect_traffic_data()
        queues = np.array([d['queue'] for d in traffic])
        waits = np.array([d['wait'] for d in traffic])
        speeds = np.array([d['speed'] for d in traffic])
        return np.array([
            queues.max() if queues.size else 0,
            queues.mean() if queues.size else 0,
            speeds.min() if speeds.size else 0,
            speeds.mean() if speeds.size else 0,
            waits.max() if waits.size else 0,
            waits.mean() if waits.size else 0
        ])

    def handle_special_cases(self):
        for lane_id in self.lane_id_list:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                try:
                    if traci.vehicle.getVehicleClass(vid) in ['emergency', 'priority']:
                        self.handle_emergency_vehicle(lane_id)
                        return
                except traci.TraCIException:
                    continue
        for lane_id in self.lane_id_list:
            if self.is_left_turn_lane(lane_id):
                if self.is_lane_blocked(lane_id):
                    self.handle_protected_left_turn(lane_id)

    def is_left_turn_lane(self, lane_id):
        try:
            return any(len(link) > 6 and link[6] == 'l' for link in traci.lane.getLinks(lane_id))
        except traci.TraCIException:
            return False

    def is_lane_blocked(self, lane_id):
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
        if not vehicles:
            return False
        try:
            return traci.vehicle.getSpeed(vehicles[0]) < 0.1
        except traci.TraCIException:
            return False

    def handle_emergency_vehicle(self, lane_id):
        state_str = self.apc.create_phase_state(green_lanes=[lane_id])
        phase_idx = self.apc.get_phase_by_state(state_str)
        if phase_idx is None:
            logger.info(f"[EMERGENCY] Creating new phase for emergency lane {lane_id}")
            phase_idx = self.apc.create_new_phase(state_str, self.apc.max_green)
        else:
            logger.info(f"[EMERGENCY] Found existing phase for emergency lane {lane_id}: idx={phase_idx}")
        if phase_idx is not None:
            self.apc.set_phase(phase_idx)
            logger.info(f"Emergency priority phase set for {lane_id}")

    def handle_protected_left_turn(self, lane_id):
        state_str = self.apc.create_phase_state(green_lanes=[lane_id])
        phase_idx = self.apc.get_phase_by_state(state_str)
        if phase_idx is None:
            logger.info(f"[PROTECTED_LEFT] Creating new protected left phase for lane {lane_id}")
            phase_idx = self.apc.create_new_phase(state_str, self.apc.min_green)
        else:
            logger.info(f"[PROTECTED_LEFT] Found existing protected left phase for lane {lane_id}: idx={phase_idx}")
        if phase_idx is not None:
            self.apc.set_phase(phase_idx)
            logger.info(f"Protected left turn phase set for {lane_id}")

def main():
    parser = argparse.ArgumentParser(description="Run universal SUMO RL traffic simulation (API phases)")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    args = parser.parse_args()

    def simulation_loop():
        for episode in range(args.episodes):
            print(f"\n{'='*50}\n🚦 STARTING UNIVERSAL EPISODE {episode + 1}/{args.episodes}\n{'='*50}")
            sumo_binary = "sumo-gui" if args.gui else "sumo"
            sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary), '-c', args.sumo, '--start', '--quit-on-end']
            traci.start(sumo_cmd)
            controller = UniversalSmartTrafficController(args.sumo, mode="train")
            controller.initialize()
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                if args.max_steps and step >= args.max_steps:
                    print(f"Reached max steps ({args.max_steps}), ending episode."); break
                controller.run_step()
                traci.simulationStep()
                step += 1
            print(f"Episode {episode + 1} completed after {step} steps")
            traci.close()
            if episode < args.episodes - 1: time.sleep(2)
        print(f"\n🎉 All {args.episodes} episodes completed!")

    simulation_loop()

if __name__ == "__main__":
    main()