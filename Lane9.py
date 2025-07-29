import os, sys, traci, threading, warnings, time, argparse, pickle, datetime, logging
import numpy as np
warnings.filterwarnings('ignore')
from traffic_light_display import TrafficLightPhaseDisplay
from collections import defaultdict
from api_phase_client import post_traffic_to_api, get_phases_from_api

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")

os.environ.setdefault('SUMO_HOME', r'C:\\Program Files (x86)\\Eclipse\\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

class EnhancedQLearningAgent:
    def __init__(self, state_size, action_size, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, mode="train"):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.mode = mode
        self.q_table = {}
        self.reward_history = []

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
        # Congestion-biased exploration
        if self.mode == "train" and np.random.rand() < self.epsilon:
            if hasattr(state, 'queues') and hasattr(state, 'waits'):
                congestion_scores = np.array([q + w * 0.5 for q, w in zip(state.queues, state.waits)])
                if congestion_scores.sum() > 0:
                    congestion_scores = congestion_scores / congestion_scores.sum()
                    return np.random.choice(range(action_size), p=congestion_scores)
            return np.random.randint(action_size)
        return np.argmax(qs)

    def _state_to_key(self, state, tl_id=None):
        try:
            arr = np.round(np.array(state), 2)
            key = tuple(arr.tolist())
            return (tl_id, key) if tl_id is not None else key
        except Exception:
            return (tl_id, (0,)) if tl_id is not None else (0,)

    def update_q_table(self, state, action, reward, next_state, tl_id=None, action_size=None):
        if not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return
        action_size = action_size or self.action_size
        sk, nsk = self._state_to_key(state, tl_id), self._state_to_key(next_state, tl_id)
        for k in [sk, nsk]:
            if k not in self.q_table or len(self.q_table[k]) < action_size:
                self.q_table[k] = np.zeros(action_size)
        q, nq = self.q_table[sk][action], np.max(self.q_table[nsk])
        lr, gamma = 0.1, 0.95
        new_q = q + lr * (reward + gamma * nq - q)
        self.q_table[sk][action] = new_q if not (np.isnan(new_q) or np.isinf(new_q)) else q

class UniversalSmartTrafficController:
    def __init__(self, sumocfg_path=None, mode="train"):
        self.mode = mode
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

    def run_step(self):
        # 1. Gather traffic data
        traffic = []
        for lane_id in self.lane_id_list:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            wait = traci.lane.getWaitingTime(lane_id)
            speed = traci.lane.getLastStepMeanSpeed(lane_id)
            traffic.append({"lane_id": lane_id, "queue": queue, "wait": wait, "speed": speed})

        # 2. Send to API (API does all adaptive control/phase logic)
        post_traffic_to_api(self.tls_id, traffic)

        # 3. Fetch all phases for this intersection from API
        phases = get_phases_from_api(self.tls_id)
        if not phases:
            print("[WARN] No phases available from API, skipping step.")
            return

        # 4. Prepare RL state vector
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

        # 5. RL agent picks phase
        phase_idx = self.rl_agent.get_action(state, tl_id=self.tls_id, action_size=len(phases))
        phase = phases[phase_idx]
        # PATCH: Use correct tls_id in getRedYellowGreenState
        expected_state_length = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
        actual_state_length = len(phase.get('state', ""))

        print(f"[DEBUG] About to set phase: idx={phase_idx}, duration={phase['duration']}, state={phase['state']}")
        print(f"[DEBUG] Expected state length: {expected_state_length}, Actual state length: {actual_state_length}")

        if actual_state_length != expected_state_length:
            print(f"[ERROR] Phase state string length ({actual_state_length}) does not match expected ({expected_state_length}). Skipping phase set to avoid crash.")
            return

        try:
            traci.trafficlight.setPhase(self.tls_id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.tls_id, phase['duration'])
        except Exception as e:
            print(f"[ERROR] Exception while setting phase: {e}")
            print(f"[ERROR] Phase data: {phase}")

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
            state_size = 6
            phases = get_phases_from_api(controller.tls_id)
            action_size = len(phases) or 1
            controller.rl_agent = EnhancedQLearningAgent(
                state_size=state_size,
                action_size=action_size,
                mode="train"
            )
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