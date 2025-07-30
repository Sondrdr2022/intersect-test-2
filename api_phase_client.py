import os
import json
import datetime
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Optional
from supabase import create_client, Client

SUPABASE_URL = "https://zckiwulodojgcfwyjrcx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxNDQ3NDQsImV4cCI6MjA2ODcyMDc0NH0.glM0KT1bfV_5BgQbOLS5JxhjTjJR5sLNn7nuoNpBtBc"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()
def is_valid_phase_state(state, controlled_lanes):
    return isinstance(state, str) and len(state) == len(controlled_lanes)

class LaneData(BaseModel):
    id: str
    queue: float
    wait: float
    speed: float
    density: float
    vehicle_ids: List[str]
    vehicle_classes: List[str]
    is_left: bool = False
    is_right: bool = False

class PhaseData(BaseModel):
    index: int
    state: str
    duration: float

class IntersectionState(BaseModel):
    tls_id: str
    current_phase: int
    phases: List[PhaseData]
    lanes: List[LaneData]
    sim_time: float

class RLStepRequest(BaseModel):
    intersection: IntersectionState

class ControlCommand(BaseModel):
    action: str
    phase_index: Optional[int] = None
    state: Optional[str] = None
    duration: Optional[float] = None

class RLStepResponse(BaseModel):
    commands: List[ControlCommand]
    reward: float

def print_and_log(msg: str, event_log: list = None, log_dict: dict = None):
    print(msg)
    if event_log is not None:
        log_dict = log_dict or {}
        log_dict = dict(log_dict, msg=msg, timestamp=datetime.datetime.now().isoformat())
        event_log.append(log_dict)
        # Optionally buffer and batch DB writes if needed

class PhasePersistence:
    """Persist phase definitions and durations in Supabase"""
    @staticmethod
    def save_phase(tls_id, phase_idx, state, duration, controlled_lanes=None):
        if controlled_lanes and not is_valid_phase_state(state, controlled_lanes):
            print(f"[PHASE-PERSIST] Not saving phase {phase_idx}: state length {len(state)} != lanes {len(controlled_lanes)}")
            return
        # Patch: convert all to native types
        try:
            tls_id = str(tls_id)
            phase_idx = int(phase_idx)
            state = str(state)
            duration = float(duration)
            supabase.table("apc_phases").upsert({
                "tls_id": tls_id,
                "phase_idx": phase_idx,
                "state": state,
                "duration": duration,
                "updated_at": datetime.datetime.now().isoformat()
            }).execute()
        except Exception as e:
            print(f"[PHASE-PERSIST] Save error: {e}")
            print("Hint: Check SUPABASE_KEY, SUPABASE_URL and make sure the `apc_phases` table exists with correct schema.")

    @staticmethod
    def load_phases(tls_id):
        try:
            tls_id = str(tls_id)
            resp = supabase.table("apc_phases").select("*").eq("tls_id", tls_id).execute()
            if resp.data:
                return {int(row["phase_idx"]): dict(state=row["state"], duration=float(row["duration"]))
                        for row in resp.data if "phase_idx" in row}
        except Exception as e:
            print(f"[PHASE-PERSIST] Load error: {e}")
            print("Hint: Check SUPABASE_KEY, SUPABASE_URL and make sure the `apc_phases` table exists with correct schema.")
        return {}

class Lane7QAgent:
    def __init__(self, tls_id, state_size=8, action_size=8, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, min_epsilon=0.01):
        self.tls_id = tls_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self._load_q_table_from_supabase()

    def _state_to_key(self, state):
        arr = np.round(np.array(state), 2)
        return str(tuple(arr.tolist()))

    def get_action(self, state, n_actions):
        key = self._state_to_key(state)
        if key not in self.q_table or len(self.q_table[key]) < n_actions:
            self.q_table[key] = [0.0] * n_actions
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(n_actions))
        return int(np.argmax(self.q_table[key][:n_actions]))

    def update(self, s, a, r, s2, n_actions):
        if s is None or a is None: return
        key, next_key = self._state_to_key(s), self._state_to_key(s2)
        for k in [key, next_key]:
            if k not in self.q_table or len(self.q_table[k]) < n_actions:
                self.q_table[k] = [0.0] * n_actions
        q, nq = self.q_table[key][a], max(self.q_table[next_key])
        self.q_table[key][a] = float(q + self.learning_rate * (r + self.discount_factor * nq - q))

    def save_q_table_to_supabase(self):
        try:
            q_json = json.dumps(self.q_table)
            supabase.table("q_tables").upsert({
                "tls_id": self.tls_id,
                "q_table": q_json,
                "updated_at": datetime.datetime.now().isoformat()
            }).execute()
        except Exception as e:
            print(f"Supabase Q-table save error: {e}")

    def _load_q_table_from_supabase(self):
        try:
            resp = supabase.table("q_tables").select("q_table").eq("tls_id", self.tls_id).execute()
            if resp.data and resp.data[0].get("q_table"):
                self.q_table = json.loads(resp.data[0]["q_table"])
        except Exception as e:
            print(f"Supabase Q-table load error: {e}")

def get_state_vector(intersection: IntersectionState):
    queues = [l.queue for l in intersection.lanes]
    waits = [l.wait for l in intersection.lanes]
    speeds = [l.speed for l in intersection.lanes]
    current_phase = intersection.current_phase
    n_phases = len(intersection.phases)
    state = np.array([
        np.max(queues) if queues else 0,
        np.mean(queues) if queues else 0,
        np.max(waits) if waits else 0,
        np.mean(waits) if waits else 0,
        np.min(speeds) if speeds else 0,
        np.mean(speeds) if speeds else 0,
        current_phase,
        n_phases
    ])
    return tuple(np.round(state, 2))

# ---- Place your helper here ----
def create_valid_phase_state(green_lanes, controlled_lanes):
    # PATCH: Build state string with correct mapping
    state = []
    for lane in controlled_lanes:
        if lane in green_lanes:
            state.append('G')
        else:
            state.append('r')
    return ''.join(['G' if lane in green_lanes else 'r' for lane in controlled_lanes])
class Lane7Controller:
    def __init__(self, tls_id: str, min_green=10, max_green=60):
        self.tls_id = tls_id
        self.min_green = min_green
        self.max_green = max_green
        self.starvation: Dict[str, float] = {}
        self.protected_left_cooldown: Dict[str, float] = {}
        self.last_phase_switch_time = 0.0
        self.last_phase_idx = None
        self.event_log = []
        self.supabase_event_batch = []
        self.emergency_cooldown = {}
        self.left_cooldowns = {}
        self.q_agent = Lane7QAgent(tls_id)
        self.supabase_qtable_interval = 1000
        self.supabase_qtable_counter = 0
        self.phase_db = PhasePersistence
        # Load persisted phases for this TLS
        self.persisted_phases = self.phase_db.load_phases(tls_id)
        self.phase_stuck_counter = {}  # PATCH: counts consecutive times a phase is selected
        self.max_stuck_iterations = 3  # PATCH: after this, force a change
        self.max_phase_green = 40      # PATCH: max seconds a phase can stay green
    def _supabase_save_qtable(self):
        if self.supabase_qtable_counter % self.supabase_qtable_interval == 0:
            self.q_agent.save_q_table_to_supabase()
        self.supabase_qtable_counter += 1

    def insert_yellow_phase(self, intersection: IntersectionState, from_idx, to_idx, cmds):
        # Defensive: only insert if both indices are in range
        n = len(intersection.phases)
        if from_idx is None or to_idx is None or from_idx == to_idx:
            return
        if from_idx >= n or to_idx >= n:
            print(f"[YELLOW PHASE] Skipped: from_idx {from_idx} or to_idx {to_idx} out of range (n={n})")
            return
        current = intersection.phases[from_idx].state.upper()
        target = intersection.phases[to_idx].state.upper()
        yellow_state = ''.join('y' if current[i] == 'G' and target[i] == 'R' else 'r' for i in range(len(current)))
        for ph in intersection.phases:
            if ph.state == yellow_state:
                cmds.append(ControlCommand(action="switch_phase", phase_index=ph.index))
                cmds.append(ControlCommand(action="set_duration", phase_index=ph.index, duration=self.yellow_duration))
                return
        new_idx = n
        cmds.append(ControlCommand(action="add_phase", state=yellow_state, duration=self.yellow_duration))
        cmds.append(ControlCommand(action="switch_phase", phase_index=new_idx))
        self.persist_phase(new_idx, yellow_state, self.yellow_duration, [l.id for l in intersection.lanes])

    def _log(self, msg, log_dict=None):
        print_and_log(msg, self.event_log, log_dict)

    def _find_lane_idx(self, lanes, lane_id):
        for i, l in enumerate(lanes):
            if l.id == lane_id:
                return i
        return None

    def emergency_detection(self, intersection: IntersectionState):
        sim_time = intersection.sim_time
        for lane in intersection.lanes:
            for vid, vclass in zip(lane.vehicle_ids, lane.vehicle_classes):
                if vclass in ("emergency", "authority"):
                    if lane.id in self.emergency_cooldown and sim_time - self.emergency_cooldown[lane.id] < self.min_green:
                        continue
                    self.emergency_cooldown[lane.id] = sim_time
                    self._log(f"[EMERGENCY] Detected emergency ({vclass}) on {lane.id}", {"event": "emergency", "lane": lane.id})
                    return lane.id
        return None

    def starvation_detection(self, intersection: IntersectionState):
        sim_time = intersection.sim_time
        for lane in intersection.lanes:
            last = self.starvation.get(lane.id, sim_time)
            if lane.queue > 0 and sim_time - last > 90:
                self.starvation[lane.id] = sim_time
                self._log(f"[STARVATION] Starved lane {lane.id}", {"event": "starvation", "lane": lane.id})
                return lane.id
        return None

    def protected_left_detection(self, intersection: IntersectionState):
        sim_time = intersection.sim_time
        for lane in intersection.lanes:
            cool = self.left_cooldowns.get(lane.id, 0)
            if lane.is_left and lane.queue >= 2 and lane.speed < 0.1 and sim_time - cool > 30:
                self.left_cooldowns[lane.id] = sim_time
                self._log(f"[PROTECTED_LEFT] Blocked left-turn detected on {lane.id}", {"event": "protected_left", "lane": lane.id})
                return lane.id
        return None

    def reward(self, intersection: IntersectionState):
        queues = [l.queue for l in intersection.lanes]
        waits = [l.wait for l in intersection.lanes]
        speeds = [l.speed for l in intersection.lanes]
        reward = -2 * np.mean(queues) - 0.5 * np.mean(waits) + 0.3 * np.mean(speeds)
        bonus, penalty = 0, 0
        avg_status = np.mean([l.queue/10 + l.wait/60 for l in intersection.lanes])
        if avg_status >= 5 * 1.25:
            penalty = 2
        elif avg_status <= 2.5:
            bonus = 1
        out = float(np.clip(reward + bonus - penalty, -100, 100))
        self._log(f"[REWARD] R={out:.2f} (bonus={bonus}, penalty={penalty})", {"event": "reward", "val": out})
        return out

    def persist_phase(self, idx, state, duration):
        self.phase_db.save_phase(self.tls_id, idx, state, duration)
        self.persisted_phases[idx] = dict(state=state, duration=duration)

    def restore_phase_duration(self, idx, state, default):
        # Try to get persisted duration, else use default
        ph = self.persisted_phases.get(idx)
        if ph and ph["state"] == state:
            return ph["duration"]
        return default

    def log_phase_switch(self, intersection: IntersectionState, new_phase_idx: int, force_priority=False):
        now = intersection.sim_time
        elapsed = now - self.last_phase_switch_time
        if (self.last_phase_idx == new_phase_idx) and (elapsed < self.min_green) and not force_priority:
            self._log(f"[PHASE SWITCH BLOCKED] Flicker prevention for phase {new_phase_idx}")
            return False
        if (elapsed < self.min_green) and not force_priority:
            self._log(f"[MIN_GREEN BLOCK] Phase switch blocked (elapsed {elapsed:.1f}s < {self.min_green}s)")
            return False
        self._log(f"[PHASE SWITCH] {self.last_phase_idx} → {new_phase_idx} (elapsed {elapsed:.1f}s)", {"event": "phase_switch", "from": self.last_phase_idx, "to": new_phase_idx})
        self.last_phase_idx = new_phase_idx
        self.last_phase_switch_time = now
        return True

    def handle_step(self, intersection: IntersectionState):
        controlled_lanes = [l.id for l in intersection.lanes]
        self._supabase_save_qtable()
        sim_time = intersection.sim_time
        phases = intersection.phases
        cmds = []

        # --- 1. Emergency always overrides everything ---
        emerg_lane = self.emergency_detection(intersection)
        if emerg_lane:
            idx = self._find_lane_idx(intersection.lanes, emerg_lane)
            for ph in phases:
                if idx is not None and idx < len(ph.state) and ph.state[idx].upper() == "G":
                    if is_valid_phase_state(ph.state, controlled_lanes):
                        if self.log_phase_switch(intersection, ph.index, force_priority=True):
                            duration = self.restore_phase_duration(ph.index, ph.state, 30)
                            cmds.append(ControlCommand(action="switch_phase", phase_index=ph.index))
                            cmds.append(ControlCommand(action="set_duration", phase_index=ph.index, duration=duration))
                            self.persist_phase(ph.index, ph.state, duration)
                            self._log(f"[EMERGENCY] Switching to phase {ph.index} for emergency on {emerg_lane}")
                            return cmds, 90

        # --- 2. Protected left turn logic ---
        block_left = self.protected_left_detection(intersection)
        if block_left:
            state = create_valid_phase_state([block_left], controlled_lanes)
            if not is_valid_phase_state(state, controlled_lanes):
                print(f"[COMMAND] Not issuing add_phase: state length mismatch ({len(state)} vs {len(controlled_lanes)})")
            else:
                found = None
                for ph in phases:
                    if ph.state == state:
                        found = ph.index
                        break
                duration = 20
                if found is None:
                    new_idx = len(phases)
                    cmds.append(ControlCommand(action="add_phase", state=state, duration=duration))
                    cmds.append(ControlCommand(action="switch_phase", phase_index=new_idx))
                    cmds.append(ControlCommand(action="set_duration", phase_index=new_idx, duration=duration))
                    self.persist_phase(new_idx, state, duration)
                    self._log(f"[PROTECTED_LEFT] Created and switched to new protected left phase for {block_left}")
                    return cmds, 70
                else:
                    if self.log_phase_switch(intersection, found):
                        duration = self.restore_phase_duration(found, state, duration)
                        cmds.append(ControlCommand(action="switch_phase", phase_index=found))
                        cmds.append(ControlCommand(action="set_duration", phase_index=found, duration=duration))
                        self.persist_phase(found, state, duration)
                        self._log(f"[PROTECTED_LEFT] Switched to existing protected left phase {found} for {block_left}")
                        return cmds, 70

        # --- 3. Starvation: Serve long-starved lanes ---
        starved_lane = self.starvation_detection(intersection)
        if starved_lane:
            idx = self._find_lane_idx(intersection.lanes, starved_lane)
            for ph in phases:
                if idx is not None and idx < len(ph.state) and ph.state[idx].upper() == "G":
                    if is_valid_phase_state(ph.state, controlled_lanes):
                        if self.log_phase_switch(intersection, ph.index, force_priority=True):
                            duration = self.restore_phase_duration(ph.index, ph.state, 20)
                            cmds.append(ControlCommand(action="switch_phase", phase_index=ph.index))
                            cmds.append(ControlCommand(action="set_duration", phase_index=ph.index, duration=duration))
                            self.persist_phase(ph.index, ph.state, duration)
                            self._log(f"[STARVATION] Switching to phase {ph.index} for starved lane {starved_lane}")
                            return cmds, 50

        # --- 4. Heavy congestion fallback: Always serve the most congested lane if stuck ---
        most_congested = max(intersection.lanes, key=lambda l: (l.queue, l.wait), default=None)
        if most_congested and (most_congested.queue > 12 or most_congested.wait > 80):
            idx = self._find_lane_idx(intersection.lanes, most_congested.id)
            for ph in phases:
                if idx is not None and idx < len(ph.state) and ph.state[idx].upper() == "G":
                    if is_valid_phase_state(ph.state, controlled_lanes):
                        if intersection.current_phase != ph.index:
                            if self.log_phase_switch(intersection, ph.index, force_priority=True):
                                duration = self.restore_phase_duration(ph.index, ph.state, self.max_green)
                                cmds.append(ControlCommand(action="switch_phase", phase_index=ph.index))
                                cmds.append(ControlCommand(action="set_duration", phase_index=ph.index, duration=duration))
                                self.persist_phase(ph.index, ph.state, duration)
                                self._log(f"[CONGESTION] Forced switch to phase {ph.index} for congestion on {most_congested.id}")
                                return cmds, 75

        # --- 5. RL agent selection (with stuck-phase detection and max green enforcement) ---
        # Prevent the RL from getting stuck on a phase forever
        if not hasattr(self, "phase_stuck_counter"):
            self.phase_stuck_counter = {}  # phase_idx -> count
        if not hasattr(self, "max_stuck_iterations"):
            self.max_stuck_iterations = 3
        if not hasattr(self, "max_phase_green"):
            self.max_phase_green = 40

        state = get_state_vector(intersection)
        n_phases = len(phases)
        action = self.q_agent.get_action(state, n_phases)
        reward = self.reward(intersection)

        # Update stuck counter
        if intersection.current_phase == action:
            self.phase_stuck_counter[action] = self.phase_stuck_counter.get(action, 0) + 1
        else:
            self.phase_stuck_counter[action] = 0

        # If stuck too long, force a switch to some other phase with a queue
        if self.phase_stuck_counter.get(action, 0) >= self.max_stuck_iterations:
            for ph in phases:
                if ph.index != action:
                    for i, lane in enumerate(intersection.lanes):
                        if i < len(ph.state) and ph.state[i].upper() == "G" and lane.queue > 2:
                            if is_valid_phase_state(ph.state, controlled_lanes):
                                if self.log_phase_switch(intersection, ph.index, force_priority=True):
                                    duration = self.restore_phase_duration(ph.index, ph.state, self.min_green)
                                    cmds.append(ControlCommand(action="switch_phase", phase_index=ph.index))
                                    cmds.append(ControlCommand(action="set_duration", phase_index=ph.index, duration=duration))
                                    self.persist_phase(ph.index, ph.state, duration)
                                    self._log(f"[FALLBACK] Forced fallback switch to phase {ph.index} to avoid deadlock")
                                    self.phase_stuck_counter[action] = 0
                                    return cmds, reward

        # Max green enforcement
        elapsed = sim_time - self.last_phase_switch_time
        if elapsed > self.max_phase_green:
            next_idx = (intersection.current_phase + 1) % n_phases
            ph = phases[next_idx]
            if is_valid_phase_state(ph.state, controlled_lanes):
                if self.log_phase_switch(intersection, next_idx, force_priority=True):
                    duration = self.restore_phase_duration(next_idx, ph.state, self.min_green)
                    cmds.append(ControlCommand(action="switch_phase", phase_index=next_idx))
                    cmds.append(ControlCommand(action="set_duration", phase_index=next_idx, duration=duration))
                    self.persist_phase(next_idx, ph.state, duration)
                    self._log(f"[MAX_GREEN] Forced switch due to max green time in phase {intersection.current_phase}")
                    return cmds, reward

        # Standard RL agent selection
        ph = phases[action]
        if is_valid_phase_state(ph.state, controlled_lanes):
            if self.last_phase_idx is None or self.log_phase_switch(intersection, action):
                duration = self.restore_phase_duration(action, ph.state, self.min_green)
                cmds.append(ControlCommand(action="switch_phase", phase_index=action))
                cmds.append(ControlCommand(action="set_duration", phase_index=action, duration=duration))
                self.persist_phase(action, ph.state, duration)
                self.q_agent.update(self.q_agent.last_state, self.q_agent.last_action, self.q_agent.last_reward, state, n_phases)
                self.q_agent.last_state = state
                self.q_agent.last_action = action
                self.q_agent.last_reward = reward
                self._log(f"[RL] RL agent selected phase {action} (reward={reward:.2f})")
        else:
            print(f"[RL] Skipping RL phase {action}: state length mismatch ({len(ph.state)} vs {len(controlled_lanes)})")
        return cmds, reward
controllers: Dict[str, Lane7Controller] = {}

@app.websocket("/apc/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req_dict = json.loads(data)
            req = RLStepRequest(**req_dict)
            intersection = req.intersection
            tls_id = intersection.tls_id
            if tls_id not in controllers:
                controllers[tls_id] = Lane7Controller(tls_id)
            ctl = controllers[tls_id]
            cmds, reward = ctl.handle_step(intersection)
            await websocket.send_text(RLStepResponse(commands=cmds, reward=reward).json())
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        for ctl in controllers.values():
            ctl.q_agent.save_q_table_to_supabase()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_phase_client:app", host="0.0.0.0", port=8000, reload=True)