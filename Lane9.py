from collections import defaultdict
import os, sys, traci,threading, warnings, time, argparse, traceback, pickle, datetime, logging
import numpy as np
warnings.filterwarnings('ignore')
from traffic_light_display import TrafficLightPhaseDisplay
from traci._trafficlight import Logic, Phase
from supabase import create_client
import os, json, datetime
import threading
from collections import deque
from api_phase_client import post_traffic_to_api, get_phases_from_api



SUPABASE_URL = "https://zckiwulodojgcfwyjrcx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")

os.environ.setdefault('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)


def override_sumo_program_from_supabase(apc, current_phase_idx=None):
    phase_seq = apc.get_full_phase_sequence()
    if not phase_seq:
        logger.warning(f"No Supabase phase sequence for {apc.tls_id}, skipping override.")
        return
    if current_phase_idx is None:
        current_phase_idx = traci.trafficlight.getPhase(apc.tls_id)
    phases = [Phase(duration, state) for (state, duration) in phase_seq]
    new_logic = Logic(
        programID="SUPABASE-OVERRIDE",
        type=traci.constants.TL_LOGIC_PROGRAM,
        currentPhaseIndex=current_phase_idx if 0 <= current_phase_idx < len(phases) else 0,
        phases=phases,
    )
    traci.trafficlight.setCompleteRedYellowGreenDefinition(apc.tls_id, new_logic)
    logger.info(f"Overrode SUMO program for {apc.tls_id} with {len(phases)} Supabase phases.")

def override_all_tls_with_supabase(controller, current_phase_idx=None):
    logger.info("Overriding all TLS logic from Supabase records...")
    for tl_id, apc in getattr(controller, 'adaptive_phase_controllers', {}).items():
        phase_seq = apc.get_full_phase_sequence()
        if not phase_seq:
            logger.warning(f"No Supabase phase sequence for {tl_id}, skipping override.")
            continue
        traci_phases = [Phase(duration, state) for (state, duration) in phase_seq]
        phase_indices = [i for i, _ in enumerate(traci_phases)]
        logger.info(f"For {tl_id}: Supabase has {len(traci_phases)} phases, indices: {phase_indices}")
        logic = Logic(
            tl_id,
            0,
            current_phase_idx if current_phase_idx is not None else 0,
            traci_phases
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_id, logic)
        logger.info(f"Set Supabase logic for {tl_id} ({len(traci_phases)} phases)")

class AsyncSupabaseWriter(threading.Thread):
    def __init__(self, apc, interval=2):
        super().__init__(daemon=True)
        self.apc = apc
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            self.apc.flush_pending_supabase_writes()
            time.sleep(self.interval)

    def stop(self):
        self.running = False

class AdaptivePhaseController:
    def __init__(self, lane_ids, tls_id, min_green=30, max_green=80):
        self.lane_ids = lane_ids
        self.tls_id = tls_id
        self.min_green = min_green
        self.max_green = max_green
        self.phase_count = 0
        self.last_phase_switch_sim_time = 0
        self.last_phase_idx = None

    def send_current_traffic_and_update_phases(self):
        """
        Gather current traffic data and send to the API for phase calculation.
        """
        traffic = []
        for lane_id in self.lane_ids:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            wait = traci.lane.getWaitingTime(lane_id)
            speed = traci.lane.getLastStepMeanSpeed(lane_id)
            traffic.append({"lane_id": lane_id, "queue": queue, "wait": wait, "speed": speed})
        post_traffic_to_api(self.tls_id, traffic)

    def load_phases_from_api(self):
        """
        Fetch all phases for this intersection from the API.
        """
        phases = get_phases_from_api(self.tls_id)
        return phases

    def apply_phase_from_api(self, phase_idx):
        """
        Set the active phase in SUMO using the data fetched from the API.
        """
        phases = self.load_phases_from_api()
        if 0 <= phase_idx < len(phases):
            phase = phases[phase_idx]
            traci.trafficlight.setPhase(self.tls_id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.tls_id, phase['duration'])

    def control_step(self):
        """
        The main control loop for the intersection.
        - Uploads current traffic to API.
        - Fetches phases from API.
        - Uses RL agent or round-robin to select the next phase.
        - Applies the chosen phase in SUMO.
        """
        self.phase_count += 1
        # 1. Post current traffic to API (always keep server up-to-date)
        self.send_current_traffic_and_update_phases()
        # 2. Fetch latest phases from API
        phases = self.load_phases_from_api()
        if not phases:
            print(f"[WARN] No API phases available for {self.tls_id}")
            return
        # 3. RL agent or round-robin logic to choose a phase
        # Example round-robin (replace with RL agent as needed)
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        next_phase = (current_phase + 1) % len(phases)
        # 4. Apply the chosen phase
        self.apply_phase_from_api(next_phase)
        self.last_phase_idx = next_phase
        self.last_phase_switch_sim_time = traci.simulation.getTime()

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
        # Fetch latest phases from API for this intersection
        phases = get_phases_from_api(tl_id)
        if not phases:
            print(f"[RL AGENT] No phases available from API for {tl_id}, fallback to action 0")
            return 0
        action_size = len(phases)
        self.action_size = action_size

        key = self._state_to_key(state, tl_id)
        if key not in self.q_table or len(self.q_table[key]) < action_size:
            self.q_table[key] = np.zeros(action_size)
        qs = self.q_table[key][:action_size]
        
        if self.mode == "train" and np.random.rand() < self.epsilon:
            return np.random.randint(action_size)
        return np.argmax(qs)

    def switch_to_best_phase(self, state, tl_id):
        """
        RL agent picks the best phase index from available API phases for this intersection.
        """
        phases = get_phases_from_api(tl_id)
        if not phases:
            print(f"[RL AGENT] No phases available from API for {tl_id}")
            return 0
        action_size = len(phases)
        self.action_size = action_size
        action = self.get_action(state, tl_id=tl_id, action_size=action_size)
        return action

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
        # Optionally update adaptive parameters here

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
    DILEMMA_ZONE_THRESHOLD = 12.0  # meters

    def __init__(self, sumocfg_path=None, mode="train"):
        self.mode = mode
        self.step_count = 0
        self.current_episode = 0
        self.lane_id_list = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        self.tls_id = traci.trafficlight.getIDList()[0]
        self.last_phase_change = defaultdict(lambda: 0.0)
        self.phase_event_log_file = f"phase_event_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.phase_events = []
        self.rl_agent = None

    def log_phase_event(self, event: dict):
        event["timestamp"] = datetime.datetime.now().isoformat()
        self.phase_events.append(event)
        try:
            with open(self.phase_event_log_file, "wb") as f:
                pickle.dump(self.phase_events, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[WARN] Could not write phase_events to file: {e}")

    def subscribe_vehicles(self, vehicle_ids):
        for vid in vehicle_ids:
            try:
                traci.vehicle.subscribe(vid, [traci.constants.VAR_VEHICLECLASS])
            except traci.TraCIException:
                pass

    def subscribe_lanes(self, lane_ids):
        for lid in lane_ids:
            traci.lane.subscribe(lid, [
                traci.constants.LAST_STEP_VEHICLE_NUMBER,
                traci.constants.LAST_STEP_MEAN_SPEED,
                traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                traci.constants.LAST_STEP_VEHICLE_ID_LIST
            ])

    def initialize(self):
        self.subscribe_lanes(self.lane_id_list)
        # RL agent must be set from outside, e.g.:
        # self.rl_agent = EnhancedQLearningAgent(...)

    def run_step(self):
        # (1) Gather current traffic data
        traffic = []
        for lane_id in self.lane_id_list:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            wait = traci.lane.getWaitingTime(lane_id)
            speed = traci.lane.getLastStepMeanSpeed(lane_id)
            traffic.append({"lane_id": lane_id, "queue": queue, "wait": wait, "speed": speed})

        # (2) Send to API to update/compute phases
        post_traffic_to_api(self.tls_id, traffic)

        # (3) Fetch available phases from API
        phases = get_phases_from_api(self.tls_id)
        if not phases:
            print("[WARN] No phases available from API, skipping step.")
            return

        # (4) Get current state for RL agent (could be queues/waits/speeds etc)
        # Example: simple state based on mean/queue
        queues = np.array([d["queue"] for d in traffic])
        waits = np.array([d["wait"] for d in traffic])
        speeds = np.array([d["speed"] for d in traffic])
        state = np.array([
            queues.max() if queues.size else 0,
            queues.mean() if queues.size else 0,
            speeds.min() if speeds.size else 0,
            speeds.mean() if speeds.size else 0,
            waits.max() if waits.size else 0,
            waits.mean() if waits.size else 0
        ])
        # (5) RL agent picks phase index
        phase_idx = self.rl_agent.get_action(state, tl_id=self.tls_id, action_size=len(phases))
        # (6) Set phase in SUMO
        phase = phases[phase_idx]
        traci.trafficlight.setPhase(self.tls_id, phase_idx)
        traci.trafficlight.setPhaseDuration(self.tls_id, phase['duration'])
        self.last_phase_change[self.tls_id] = traci.simulation.getTime()
        self.log_phase_event({
            "action": "phase_set",
            "phase_idx": phase_idx,
            "duration": phase['duration'],
            "timestamp": datetime.datetime.now().isoformat()
        })

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
            # RL agent assignment here (must be done before .run_step)
            # controller.rl_agent = EnhancedQLearningAgent(...)
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