import os, sys, traci, pickle, threading, warnings, time, argparse, logging
import numpy as np

from api_phase_client import (
    post_traffic_to_api,
    get_phases_from_api,
    create_new_phase_in_api,
    update_phase_duration_in_api,
    get_phase_by_state_from_api
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Lane9Controller")

os.environ.setdefault('SUMO_HOME', r'C:\\Program Files (x86)\\Eclipse\\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

class APIBasedPhaseController:
    """
    Handles all phase/logic via remote API. No local APC logic.
    """
    def __init__(self, tls_id, lane_ids):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.current_phase_idx = 0
        self.phases = []
        self.phase_map = {}

    def push_traffic_and_update_phases(self):
        traffic = []
        for lane_id in self.lane_ids:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                wait = traci.lane.getWaitingTime(lane_id)
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                traffic.append({"lane_id": lane_id, "queue": queue, "wait": wait, "speed": speed})
            except traci.TraCIException:
                traffic.append({"lane_id": lane_id, "queue": 0, "wait": 0, "speed": 0})
        expected_state_length = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
        self.phases = post_traffic_to_api(self.tls_id, traffic, expected_state_length)
        self.phase_map = {p['state']: i for i, p in enumerate(self.phases)}

    def get_phase_by_state(self, state_str):
        idx = self.phase_map.get(state_str)
        if idx is not None:
            return idx
        return get_phase_by_state_from_api(self.tls_id, state_str)

    def create_new_phase(self, state_str, duration):
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
                return True
            except traci.TraCIException as e:
                logger.error(f"Error setting phase: {e}")
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

class RLAgent:
    """
    RL agent as in Lane7, but all APC logic is remote (API).
    """
    def __init__(self, state_size, action_size, controller, q_table_file="lane9_qtable.pkl"):
        self.state_size = state_size
        self.action_size = action_size
        self.controller = controller
        self.q_table = {}
        self.q_table_file = q_table_file
        self.load_q_table()

    def state_to_key(self, state):
        arr = np.round(np.array(state), 2)
        return tuple(arr.tolist())

    def get_action(self, state, epsilon=0.1):
        key = self.state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state, alpha=0.1, gamma=0.95):
        k = self.state_to_key(state)
        nk = self.state_to_key(next_state)
        for x in [k, nk]:
            if x not in self.q_table:
                self.q_table[x] = np.zeros(self.action_size)
        q = self.q_table[k][action]
        nq = np.max(self.q_table[nk])
        self.q_table[k][action] = q + alpha * (reward + gamma * nq - q)

    def save_q_table(self):
        with open(self.q_table_file, "wb") as f:
            pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_q_table(self):
        try:
            with open(self.q_table_file, "rb") as f:
                self.q_table = pickle.load(f)
        except Exception:
            self.q_table = {}

class Lane9Controller:
    def __init__(self, tls_id, lane_ids):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.apc = APIBasedPhaseController(tls_id, lane_ids)
        self.episode_reward = 0

        self.state_size = 6
        self.action_size = len(lane_ids) + 1  # +1 for all-red
        self.agent = RLAgent(self.state_size, self.action_size, self.apc)

    def get_state_vector(self):
        traffic = []
        for lane_id in self.lane_ids:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            wait = traci.lane.getWaitingTime(lane_id)
            speed = traci.lane.getLastStepMeanSpeed(lane_id)
            traffic.append({"queue": queue, "wait": wait, "speed": speed})
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

    def run_step(self):
        # 1. Push traffic, update phases from API (APC logic in API)
        self.apc.push_traffic_and_update_phases()
        # 2. Get RL state
        state = self.get_state_vector()
        # 3. RL agent chooses action (phase)
        action = self.agent.get_action(state)
        # 4. Set phase via API
        self.apc.set_phase(action)
        # 5. Reward: fetch from API (should be available in API or meta endpoint)
        reward = self.get_api_reward()
        # 6. Next state for RL
        next_state = self.get_state_vector()
        # 7. RL Q-update
        self.agent.update(state, action, reward, next_state)
        self.episode_reward += reward

    def get_api_reward(self):
        # Optionally: implement fetching reward from a dedicated meta endpoint in the API
        # For now, return 0 (or parse from API if available)
        return 0

    def handle_special_cases(self):
        for lane_id in self.lane_ids:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                try:
                    if traci.vehicle.getVehicleClass(vid) in ['emergency', 'priority']:
                        self.handle_emergency_vehicle(lane_id)
                        return
                except traci.TraCIException:
                    continue
        for lane_id in self.lane_ids:
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
        if not vehicles: return False
        try:
            return traci.vehicle.getSpeed(vehicles[0]) < 0.1
        except traci.TraCIException:
            return False

    def handle_emergency_vehicle(self, lane_id):
        state_str = self.apc.create_phase_state(green_lanes=[lane_id])
        idx = self.apc.get_phase_by_state(state_str)
        if idx is None:
            idx = self.apc.create_new_phase(state_str, 60)
        if idx is not None:
            self.apc.set_phase(idx)

    def handle_protected_left_turn(self, lane_id):
        state_str = self.apc.create_phase_state(green_lanes=[lane_id])
        idx = self.apc.get_phase_by_state(state_str)
        if idx is None:
            idx = self.apc.create_new_phase(state_str, 30)
        if idx is not None:
            self.apc.set_phase(idx)

    def end_episode(self):
        self.agent.save_q_table()
        self.episode_reward = 0

def main():
    parser = argparse.ArgumentParser(description="Lane9 API-Based RL Controller")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Max simulation steps')
    parser.add_argument('--episodes', type=int, default=1, help='Episodes')
    args = parser.parse_args()
    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary), '-c', args.sumo, '--start', '--quit-on-end']

    for episode in range(args.episodes):
        traci.start(sumo_cmd)
        tls_id = traci.trafficlight.getIDList()[0]
        lane_ids = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        controller = Lane9Controller(tls_id, lane_ids)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            if args.max_steps and step >= args.max_steps:
                break
            controller.run_step()
            controller.handle_special_cases()
            traci.simulationStep()
            step += 1
        controller.end_episode()
        traci.close()
        if episode < args.episodes - 1:
            time.sleep(2)

if __name__ == "__main__":
    main()