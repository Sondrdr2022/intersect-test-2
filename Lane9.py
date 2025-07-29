import os
import sys
import traci
import time
import logging
from collections import defaultdict
from api_phase_client import APIPhaseClient
import numpy as np
import json
from supabase import create_client

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Lane9Controller")

Q_TABLE = {}
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
STATE_HISTORY = []

os.environ.setdefault('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

def tuple_keys_to_str(d):
    # Handles nested Q-table dicts (state: {action: value, ...})
    return {str(k): {str(ak): av for ak, av in v.items()} if isinstance(v, dict) else v for k, v in d.items()}

def str_keys_to_tuple(d):
    import ast
    return {tuple(ast.literal_eval(k)): v for k, v in d.items()}

class Lane9Controller:
    def __init__(self, tls_id, lane_ids, min_green=30, max_green=80):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.min_green = min_green
        self.max_green = max_green
        self.api_client = APIPhaseClient(tls_id)
        self.emergency_cooldown = {}
        self.protected_left_cooldown = defaultdict(float)
        self.last_phase_switch_sim_time = 0
        self.last_phase_idx = None
        self.q_table = Q_TABLE
        self.episode_reward = 0
        self.load_state()

    def load_state(self):
        supabase = create_client("https://zckiwulodojgcfwyjrcx.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxNDQ3NDQsImV4cCI6MjA2ODcyMDc0NH0.glM0KT1bfV_5BgQbOLS5JxhjTjJR5sLNn7nuoNpBtBc")
        response = supabase.table("apc_states")\
            .select("data")\
            .eq("tls_id", self.tls_id)\
            .eq("state_type", "full")\
            .order("updated_at", desc=True)\
            .limit(1)\
            .execute()
        if response.data and len(response.data) > 0:
            state_str = response.data[0]['data']
            try:
                state = json.loads(state_str) if isinstance(state_str, str) else state_str
                # Deserialize q_table
                loaded_q_table = state.get('q_table', {})
                self.q_table.update(str_keys_to_tuple(loaded_q_table))
                STATE_HISTORY.extend(state.get('state_history', []))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse state data: {e}")
        else:
            print(f"[INFO] No state data found for tls_id: {self.tls_id}")

    def save_state(self):
        from supabase import create_client
        supabase = create_client("https://zckiwulodojgcfwyjrcx.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxNDQ3NDQsImV4cCI6MjA2ODcyMDc0NH0.glM0KT1bfV_5BgQbOLS5JxhjTjJR5sLNn7nuoNpBtBc")
        payload = {
            "tls_id": self.tls_id,
            "data": {
                "q_table": tuple_keys_to_str(self.q_table),
                "state_history": STATE_HISTORY
            }
        }
        supabase.table("apc_states").insert(payload).execute()

    def get_state(self, lane_data):
        state = tuple(int(min(data['queue_length'], 50) // 5) for data in lane_data.values())
        STATE_HISTORY.append(state)
        return state

    def get_action(self, state):
        if np.random.random() < EXPLORATION_RATE:
            return np.random.randint(0, len(self.get_all_phases()))
        return max(self.q_table.get(state, {}).items(), key=lambda x: x[1])[0] if state in self.q_table else 0

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, {}).get(action, 0.0)
        max_next_q = max(self.q_table.get(next_state, {}).values(), default=0.0)
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        self.q_table.setdefault(state, {})[action] = new_q

    def get_all_phases(self):
        from supabase import create_client
        supabase = create_client("https://zckiwulodojgcfwyjrcx.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxNDQ3NDQsImV4cCI6MjA2ODcyMDc0NH0.glM0KT1bfV_5BgQbOLS5JxhjTjJR5sLNn7nuoNpBtBc")
        response = supabase.table("phases")\
            .select("phase_idx, state, duration")\
            .eq("tls_id", self.tls_id)\
            .execute()
        return response.data if response.data else []

    def create_dynamic_phase(self, lane_idx, green=True):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        new_state = ['r'] * len(controlled_lanes)
        if green:
            new_state[lane_idx] = 'G'
        # Get the maximum phase_idx among existing phases and add 1
        all_phases = self.get_all_phases()
        if all_phases:
            max_idx = max([p['phase_idx'] for p in all_phases if p.get('phase_idx') is not None] + [0])
            next_idx = max_idx + 1
        else:
            next_idx = 0
        new_phase = {
            "tls_id": self.tls_id,
            "phase_idx": next_idx,
            "state": ''.join(new_state),
            "duration": self.min_green
        }
        from supabase import create_client
        supabase = create_client("https://zckiwulodojgcfwyjrcx.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxNDQ3NDQsImV4cCI6MjA2ODcyMDc0NH0.glM0KT1bfV_5BgQbOLS5JxhjTjJR5sLNn7nuoNpBtBc")
        supabase.table("phases").insert(new_phase).execute()
        return new_phase['phase_idx']

    def collect_lane_data(self):
        data = {}
        for lane_id in self.lane_ids:
            try:
                data[lane_id] = {
                    "id": lane_id,
                    "vehicle_count": traci.lane.getLastStepVehicleNumber(lane_id),
                    "mean_speed": traci.lane.getLastStepMeanSpeed(lane_id),
                    "queue_length": traci.lane.getLastStepHaltingNumber(lane_id),
                    "waiting_time": traci.lane.getWaitingTime(lane_id),
                    "density": traci.lane.getLastStepVehicleNumber(lane_id) / max(1, traci.lane.getLength(lane_id)),
                    "vehicle_ids": traci.lane.getLastStepVehicleIDs(lane_id)
                }
            except traci.TraCIException:
                continue
        return data

    def detect_emergency_vehicle(self):
        now = traci.simulation.getTime()
        for lane_id in self.lane_ids:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    if 'emergency' in v_type or 'priority' in v_type:
                        if now - self.emergency_cooldown.get(lane_id, 0) < self.min_green:
                            continue
                        self.emergency_cooldown[lane_id] = now
                        return lane_id, vid
                except traci.TraCIException:
                    continue
        return None, None

    def detect_blocked_left_turn(self):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        current_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        for lane_idx, lane_id in enumerate(controlled_lanes):
            if current_state[lane_idx] not in 'Gg':
                continue
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            if not vehicles:
                continue
            front_vehicle = vehicles[0]
            speed = traci.vehicle.getSpeed(front_vehicle)
            stopped_time = traci.vehicle.getAccumulatedWaitingTime(front_vehicle)
            if speed < 0.2 and stopped_time > 5:
                return lane_id, lane_idx
        return None, None

    def handle_emergency(self, lane_id, vid):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        lane_idx = controlled_lanes.index(lane_id)
        phase_idx = self.create_dynamic_phase(lane_idx)
        green_duration = max(self.min_green, 20)
        traci.trafficlight.setPhase(self.tls_id, phase_idx)
        traci.trafficlight.setPhaseDuration(self.tls_id, green_duration)
        print(f"[INFO] Emergency priority: {vid} on {lane_id} (Phase {phase_idx}, green for {green_duration}s)")
        actual_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        print(f"[DEBUG] Verified state: {actual_state}")

    def handle_protected_left(self, lane_id, lane_idx):
        phase_idx = self.create_dynamic_phase(lane_idx)
        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        wait = traci.lane.getWaitingTime(lane_id)
        green_duration = min(self.max_green, max(self.min_green, queue * 2 + wait * 0.1))
        traci.trafficlight.setPhase(self.tls_id, phase_idx)
        traci.trafficlight.setPhaseDuration(self.tls_id, green_duration)
        print(f"[INFO] Protected left: {lane_id} (Phase {phase_idx}, {green_duration}s)")
        actual_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        print(f"[DEBUG] Verified state: {actual_state}")

    def compute_reward(self, lane_data):
        metrics = np.array([0.0, 0.0, 0.0, 0.0])  # [density, speed, wait, queue]
        valid_lanes = 0
        for data in lane_data.values():
            metrics += np.array([
                min(data['density'], 0.2) / 0.2,
                min(data['mean_speed'], 13.89) / 13.89,
                min(data['waiting_time'], 300) / 300,
                min(data['queue_length'], 50) / 50
            ])
            valid_lanes += 1
        avg_metrics = metrics / valid_lanes if valid_lanes > 0 else np.zeros(4)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        reward = 100 * (-weights[0] * avg_metrics[0] + weights[1] * avg_metrics[1] -
                       weights[2] * avg_metrics[2] - weights[3] * avg_metrics[3])
        return np.clip(reward, -100, 100)

    def handle_priority_events(self):
        emergency_lane, vid = self.detect_emergency_vehicle()
        if emergency_lane:
            self.handle_emergency(emergency_lane, vid)
            return True
        blocked_lane, lane_idx = self.detect_blocked_left_turn()
        if blocked_lane:
            self.handle_protected_left(blocked_lane, lane_idx)
            return True
        return False

    def apply_phase_recommendation(self, recommendation):
        if recommendation is None:
            print("[WARNING] No phase recommendation available, using default phase")
            phases = self.get_all_phases()
            if phases:
                traci.trafficlight.setPhase(self.tls_id, 0)
                traci.trafficlight.setPhaseDuration(self.tls_id, self.min_green)
            return
        phase_idx = recommendation.get('phase_idx')
        duration = recommendation.get('duration', self.min_green)
        if phase_idx is not None:
            try:
                traci.trafficlight.setPhase(self.tls_id, phase_idx)
                traci.trafficlight.setPhaseDuration(self.tls_id, duration)
                print(f"[INFO] API recommended phase: {phase_idx}, duration: {duration}s")
            except traci.TraCIException as e:
                print(f"[ERROR] Failed to apply phase {phase_idx}: {e}")
                phases = self.get_all_phases()
                if phases:
                    traci.trafficlight.setPhase(self.tls_id, 0)
                    traci.trafficlight.setPhaseDuration(self.tls_id, self.min_green)

    def control_step(self):
        try:
            lane_data = self.collect_lane_data()
            sim_time = traci.simulation.getTime()
            try:
                self.api_client.submit_traffic_data(lane_data, sim_time=sim_time)
            except Exception as e:
                print(f"[ERROR] Failed to submit traffic data: {e}")
            if self.handle_priority_events():
                return
            recommendation = self.api_client.get_phase_recommendation()
            self.apply_phase_recommendation(recommendation)
            current_state = self.get_state(lane_data)
            reward = self.compute_reward(lane_data)
            next_state = self.get_state(self.collect_lane_data()) if STATE_HISTORY else current_state
            self.update_q_table(current_state, traci.trafficlight.getPhase(self.tls_id), reward, next_state)
            self.episode_reward += reward
            print(f"[DEBUG] Reward: {reward}, Episode Reward: {self.episode_reward}")
            self.save_state()
        except traci.TraCIException as e:
            print(f"[ERROR] TraCI error: {e}")
            traci.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lane9 API-Based Traffic Controller")
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
        lane_ids = traci.trafficlight.getControlledLanes(tls_id)
        controller = Lane9Controller(tls_id, lane_ids)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            if args.max_steps and step >= args.max_steps:
                break
            controller.control_step()
            traci.simulationStep()
            step += 1
        print(f"[INFO] Episode {episode} completed, Total Reward: {controller.episode_reward}")
        traci.close()
        if episode < args.episodes - 1:
            time.sleep(2)

if __name__ == "__main__":
    main()