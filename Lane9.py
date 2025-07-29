import os, sys, traci, threading, warnings, time, argparse, pickle, datetime, logging
import numpy as np
warnings.filterwarnings('ignore')
from traffic_light_display import TrafficLightPhaseDisplay
from collections import defaultdict
from api_phase_client import (
    post_traffic_to_api, 
    get_phases_from_api, 
    create_new_phase_in_api,
    update_phase_duration_in_api,
    get_phase_by_state_from_api
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("controller")

os.environ.setdefault('SUMO_HOME', r'C:\\Program Files (x86)\\Eclipse\\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

class APIBasedAdaptiveController:
    """
    The controller does not store persistent phase/duration locally.
    All phase CRUD operations are performed through the API.
    The controller collects traffic data and pushes it to the API, which computes/adapts phases.
    """
    def __init__(self, tls_id, lane_ids, display=None):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.display = display
        self.phases = []
        self.phase_map = {}
        self.current_phase_idx = 0
        self.min_green = 10
        self.max_green = 60

    def initialize(self):
        self.update_phases()

    def collect_traffic_data(self):
        traffic_data = []
        for lane_id in self.lane_ids:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                wait = traci.lane.getWaitingTime(lane_id)
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
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
        """Push current traffic data to the API and retrieve up-to-date phase definitions."""
        expected_state_length = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
        traffic_data = self.collect_traffic_data()
        self.phases = post_traffic_to_api(
            self.tls_id,
            traffic_data,
            expected_state_length=expected_state_length
        )
        self._build_phase_map()

    def _build_phase_map(self):
        self.phase_map = {}
        for idx, phase in enumerate(self.phases):
            self.phase_map[phase['state']] = idx

    def get_phase_by_state(self, state_str):
        # Try local map first, fallback to API
        idx = self.phase_map.get(state_str)
        if idx is not None:
            return idx
        # Fallback: ask API for phase index by state
        return get_phase_by_state_from_api(self.tls_id, state_str)

    def create_new_phase(self, state_str, duration=None):
        if duration is None:
            duration = self.min_green
        new_phase = create_new_phase_in_api(self.tls_id, state_str, duration)
        if new_phase:
            self.phases.append(new_phase)
            self.phase_map[state_str] = len(self.phases) - 1
            return len(self.phases) - 1
        return None

    def set_phase(self, phase_idx):
        if 0 <= phase_idx < len(self.phases):
            phase = self.phases[phase_idx]
            try:
                traci.trafficlight.setPhase(self.tls_id, phase_idx)
                traci.trafficlight.setPhaseDuration(self.tls_id, phase['duration'])
                self.current_phase_idx = phase_idx
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
        return "".join(state)

    def update_phase_duration(self, phase_idx, new_duration):
        """Update phase duration in API."""
        update_phase_duration_in_api(self.tls_id, phase_idx, new_duration)
        # Also update locally for this run
        if 0 <= phase_idx < len(self.phases):
            self.phases[phase_idx]['duration'] = new_duration

class EnhancedQLearningAgent:
    """
    RL agent that interacts with the API for all phase and state information.
    """
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
        self.adaptive_controller = adaptive_controller
        self.adaptive_params = adaptive_params or {
            'min_green': 30, 'max_green': 80, 'starvation_threshold': 40, 'reward_scale': 40,
            'queue_weight': 0.6, 'wait_weight': 0.3, 'flow_weight': 0.5, 'speed_weight': 0.2, 
            'left_turn_priority': 1.2, 'empty_green_penalty': 15, 'congestion_bonus': 10
        }
        if mode == "eval": self.epsilon = 0.0
        elif mode == "adaptive": self.epsilon = 0.01

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
        # API-based RL decision; similar to Lane7
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
            arr = np.round(np.array(state), 2)
            key = tuple(arr.tolist())
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

class UniversalSmartTrafficController:
    """
    The controller coordinates SUMO, the RL agent, and API-based phase logic.
    It detects special events, pushes traffic data to the API, and controls SUMO accordingly.
    """
    def __init__(self, sumocfg_path=None, mode="train"):
        self.mode = mode
        self.lane_id_list = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        self.tls_id = traci.trafficlight.getIDList()[0]
        self.phase_event_log_file = f"phase_event_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.phase_events = []
        self.display = TrafficLightPhaseDisplay(self.phase_events, poll_interval=500)
        self.apc = APIBasedAdaptiveController(self.tls_id, self.lane_id_list, display=self.display)
        self.rl_agent = None

    def log_phase_event(self, event: dict):
        event["timestamp"] = datetime.datetime.now().isoformat()
        self.phase_events.append(event)
        try:
            with open(self.phase_event_log_file, "wb") as f:
                pickle.dump(self.phase_events, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"[WARN] Could not write phase_events to file: {e}")

    def initialize(self):
        self.lane_id_list = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        self.tls_id = traci.trafficlight.getIDList()[0]
        self.phase_events = []
        self.display = TrafficLightPhaseDisplay(self.phase_events, poll_interval=500)
        self.apc = APIBasedAdaptiveController(self.tls_id, self.lane_id_list, display=self.display)
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
        # 1. Push traffic data and refresh phases from API
        self.apc.update_phases()
        # 2. Compose RL state vector (mirroring Lane7 logic)
        state = self._create_state_vector()
        phase_idx = self.rl_agent.get_action(
            state,
            tl_id=self.tls_id,
            action_size=len(self.apc.phases)
        )
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
            phase_idx = self.apc.create_new_phase(state_str, self.apc.max_green)
        if phase_idx is not None:
            self.apc.set_phase(phase_idx)

    def handle_protected_left_turn(self, lane_id):
        state_str = self.apc.create_phase_state(green_lanes=[lane_id])
        phase_idx = self.apc.get_phase_by_state(state_str)
        if phase_idx is None:
            phase_idx = self.apc.create_new_phase(state_str, self.apc.min_green)
        if phase_idx is not None:
            self.apc.set_phase(phase_idx)

def main():
    parser = argparse.ArgumentParser(description="Run universal SUMO RL traffic simulation (API phases)")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    args = parser.parse_args()

    def simulation_loop(controller):
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

    controller = UniversalSmartTrafficController(args.sumo, mode="train")
    sim_thread = threading.Thread(target=simulation_loop, args=(controller,), daemon=True)
    sim_thread.start()
    controller.display.start()
    sim_thread.join()

if __name__ == "__main__":
    main()