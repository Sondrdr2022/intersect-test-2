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
MAX_PHASE_DURATION = 60 


SUPABASE_URL = "https://zckiwulodojgcfwyjrcx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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
    def __init__(self, lane_ids, tls_id, alpha=1.0, min_green=30, max_green=80,
                 r_base=0.5, r_adjust=0.1, severe_congestion_threshold=0.8,
                 large_delta_t=20):
        self.lane_ids = lane_ids
        self.tls_id = tls_id
        self.alpha = alpha
        self.min_green = min_green
        self.max_green = max_green
        self.supabase = supabase
        self.traci = traci
        self.logger = logger or logging.getLogger(__name__)
        self._db_lock = threading.Lock()
        self.apc_state = {"events": [], "phases": []}
        self._pending_db_ops = []
        self._db_writer = AsyncSupabaseWriter(self)
        self._db_writer.start()
        self.scheduled_phase_updates = {}
        self.r_base = r_base
        self.r_adjust = r_adjust
        self.max_green = min(max_green, MAX_PHASE_DURATION)
        self.severe_congestion_threshold = severe_congestion_threshold
        self.large_delta_t = large_delta_t
        self.phase_repeat_counter = defaultdict(int)
        self.last_served_time = defaultdict(lambda: 0)
        self.severe_congestion_global_cooldown_time = 5  # Reduced from 10 for faster response
        self._links_map = {lid: traci.lane.getLinks(lid) for lid in lane_ids}
        self._controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        self._phase_defs = [phase for phase in traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].getPhases()]
        self.weights = np.array([0.4, 0.2, 0.2, 0.2])  # Increased queue weight from 0.25
        self.weight_history = []
        self.metric_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=50)
        self.R_target = r_base
        self.phase_count = 0
        self.rl_agent = None
        self.emergency_cooldown = {}
        self.emergency_global_cooldown = 0
        self.last_extended_time = 0
        self.last_phase_switch_sim_time = 0
        self.protected_left_cooldown = defaultdict(float)  # Reduced cooldown frequency
        self.severe_congestion_cooldown = {}
        self.severe_congestion_global_cooldown = 0
        self.last_phase_idx = None
        self.apc_state = {"events": [], "phases": []}
        self._pending_db_ops = []
        self._db_writer = AsyncSupabaseWriter(self)
        self._db_writer.start()
        self._load_apc_state_supabase()
        self.preload_phases_from_sumo()
        self.last_emergency_event = {}  # new: (lane_id, vehicle_id) -> timestamp

    def flush_pending_supabase_writes(self):
        with self._db_lock:
            if self._pending_db_ops:
                state_json = json.dumps(self._pending_db_ops[-1])
                try:
                    supabase.table("apc_states").upsert({
                        "tls_id": self.tls_id,
                        "state_type": "full",
                        "data": state_json,
                        "updated_at": datetime.datetime.now().isoformat()
                    }).execute()
                    self._pending_db_ops.clear()
                except Exception as e:
                    print(f"[Supabase] Write failed: {e}")

    def log_phase_adjustment(self, action_type, phase, old_duration, new_duration):
        print(f"[LOG] {action_type} phase {phase}: {old_duration} -> {new_duration}")    
    def subscribe_lanes(self):
        for lid in self.lane_id_list:
            traci.lane.subscribe(lid, [
                traci.constants.LAST_STEP_VEHICLE_NUMBER,
                traci.constants.LAST_STEP_MEAN_SPEED,
                traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                traci.constants.LAST_STEP_VEHICLE_ID_LIST
            ])
    def subscribe_vehicles(self, vehicle_ids):
        for vid in vehicle_ids:
            try:
                traci.vehicle.subscribe(vid, [traci.constants.VAR_VEHICLECLASS])
            except traci.TraCIException:
                pass

    def get_vehicle_classes(self, vehicle_ids):
        classes = {}
        for vid in vehicle_ids:
            results = traci.vehicle.getSubscriptionResults(vid)
            if results and traci.constants.VAR_VEHICLECLASS in results:
                classes[vid] = results[traci.constants.VAR_VEHICLECLASS]
            else:
                try:
                    classes[vid] = traci.vehicle.getVehicleClass(vid)
                except traci.TraCIException:
                    classes[vid] = None
        return classes   
    def detect_turning_lanes(self):
        left, right = set(), set()
        for lid in self.lane_id_list:

            for c in traci.lane.getLinks(lid):
                idx = 6 if len(c) > 6 else 3 if len(c) > 3 else None
                if idx and c[idx] == 'l':
                    left.add(lid)
                if idx and c[idx] == 'r':
                    right.add(lid)
        return left, right    
    def _save_apc_state_supabase(self):
        self._pending_db_ops.append(self.apc_state.copy())
    def _load_apc_state_supabase(self):
        response = supabase.table("apc_states").select("data").eq("tls_id", self.tls_id).eq("state_type", "full").execute()
        if response.data and len(response.data) > 0:
            self.apc_state = json.loads(response.data[0]["data"])
            # Fix any oversized durations in loaded state
            for phase in self.apc_state.get("phases", []):
                if phase["duration"] > MAX_PHASE_DURATION:
                    print(f"[CLEANUP] Phase {phase['phase_idx']} had duration {phase['duration']} > cap, setting to {MAX_PHASE_DURATION}")
                    phase["duration"] = MAX_PHASE_DURATION
        else:
            self.apc_state = {"events": [], "phases": []}

    def save_phase_to_supabase(self, phase_idx, duration, state_str, delta_t, raw_delta_t, penalty,
                            reward=None, bonus=None, weights=None, event_type=None, lanes=None):
        duration = min(duration, MAX_PHASE_DURATION)
        entry = {
            "phase_idx": phase_idx,
            "duration": duration,
            "base_duration": duration if not self.load_phase_from_supabase(phase_idx) else
                self.load_phase_from_supabase(phase_idx).get("base_duration", duration),            "state": state_str,
            "delta_t": delta_t,
            "raw_delta_t": raw_delta_t,
            "penalty": penalty,
            "timestamp": datetime.datetime.now().isoformat(),
            "sim_time": traci.simulation.getTime(),
            "reward": reward if reward is not None else getattr(self, "last_R", None),
            "bonus": bonus if bonus is not None else getattr(self, "last_bonus", 0),
            "weights": weights if weights is not None else self.weights.tolist(),
            "event_type": event_type,
            "lanes": lanes if lanes is not None else self.lane_ids[:],
        }
        found = False
        for i, p in enumerate(self.apc_state.setdefault("phases", [])):
            if p["phase_idx"] == phase_idx:
                # Always keep the initial base_duration
                entry["base_duration"] = p.get("base_duration", entry["base_duration"])
                self.apc_state["phases"][i] = entry
                found = True
                break
        if not found:
            self.apc_state["phases"].append(entry)
        self._save_apc_state_supabase()

    def create_or_extend_phase(self, green_lanes, delta_t):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        valid_green_lanes = [lane for lane in green_lanes if lane in controlled_lanes]
        if not valid_green_lanes:
            return None

        for lane in valid_green_lanes:
            try:
                links = traci.lane.getLinks(lane)
            except Exception:
                continue
            is_left = any(len(link) > 6 and link[6] == 'l' for link in links)
            if is_left:
                left_idx = controlled_lanes.index(lane)
                protected_state = ''.join(['G' if i == left_idx else 'r' for i in range(len(controlled_lanes))])
                phase_idx = None
                base_duration = self.min_green
                for idx, phase in enumerate(logic.getPhases()):
                    if phase.state == protected_state:
                        phase_record = self.load_phase_from_supabase(idx)
                        base_duration = phase_record["duration"] if phase_record and "duration" in phase_record else phase.duration
                        phase_idx = idx
                        break
                # Always calculate duration using capped logic
                duration_cap = min(self.max_green * 1.5 if delta_t > 10 else self.max_green, MAX_PHASE_DURATION)
                duration = np.clip(base_duration + delta_t, self.min_green, duration_cap)
                if phase_idx is not None:
                    self.save_phase_to_supabase(phase_idx, duration, protected_state, delta_t, delta_t, penalty=0)
                    if hasattr(self, "update_display"):
                        self.update_display(phase_idx, duration)
                    return phase_idx
                new_phase = traci.trafficlight.Phase(duration, protected_state)
                phases = list(logic.getPhases()) + [new_phase]
                new_logic = traci.trafficlight.Logic(
                    logic.programID, logic.type, len(phases) - 1, [traci.trafficlight.Phase(duration=ph.duration, state=ph.state) for ph in phases]
                )
                traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
                traci.trafficlight.setPhase(self.tls_id, len(phases) - 1)
                self.save_phase_to_supabase(len(phases) - 1, duration, protected_state, delta_t, delta_t, penalty=0)
                if hasattr(self, "update_display"):
                    self.update_display(len(phases) - 1, duration)
                return len(phases) - 1

        new_state = self.create_phase_state(green_lanes=valid_green_lanes)
        phase_idx = None
        base_duration = self.min_green
        for idx, phase in enumerate(logic.getPhases()):
            if phase.state == new_state:
                phase_record = self.load_phase_from_supabase(idx)
                base_duration = phase_record["duration"] if phase_record and "duration" in phase_record else phase.duration
                phase_idx = idx
                break
        duration_cap = min(self.max_green * 1.5 if delta_t > 10 else self.max_green, MAX_PHASE_DURATION)
        duration = np.clip(base_duration + delta_t, self.min_green, duration_cap)
        if phase_idx is not None:
            self.save_phase_to_supabase(phase_idx, duration, new_state, delta_t, delta_t, penalty=0)
            if hasattr(self, "update_display"):
                self.update_display(phase_idx, duration)
            return phase_idx
        new_phase = traci.trafficlight.Phase(duration, new_state)
        phases = list(logic.getPhases()) + [new_phase]
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type, len(phases) - 1, [traci.trafficlight.Phase(duration=ph.duration, state=ph.state) for ph in phases]
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
        traci.trafficlight.setPhase(self.tls_id, len(phases) - 1)
        self.save_phase_to_supabase(len(phases) - 1, duration, new_state, delta_t, delta_t, penalty=0)
        if hasattr(self, "update_display"):
            self.update_display(len(phases) - 1, duration)
        return len(phases) - 1

    def calculate_delta_t_and_penalty(self, R):
        raw_delta_t = self.alpha * (R - self.R_target)
        penalty = max(0, abs(raw_delta_t) - 10)
        ext_t = 20 * np.tanh(raw_delta_t / 20)  # Increased from 10 to 20 for larger adjustments
        delta_t = np.clip(ext_t, -20, 20)  # Expanded range from ±10 to ±20
        print(f"[DEBUG] [DELTA_T_PENALTY] R={R:.2f}, R_target={self.R_target:.2f}, raw={raw_delta_t:.2f}, Δt={delta_t:.2f}, penalty={penalty:.2f}")
        return raw_delta_t, delta_t, penalty
 
    def get_full_phase_sequence(self):
        phase_records = sorted(self.apc_state.get("phases", []), key=lambda x: x["phase_idx"])
        if not phase_records:
            return [(p.state, p.duration) for p in self._phase_defs]
        return [(rec["state"], rec["duration"]) for rec in phase_records]

    def load_phase_from_supabase(self, phase_idx=None):
        for p in self.apc_state.get("phases", []):
            if p["phase_idx"] == phase_idx:
                return p
        return None

    def _log_apc_event(self, event):
        event["timestamp"] = datetime.datetime.now().isoformat()
        event["sim_time"] = traci.simulation.getTime()
        event["tls_id"] = self.tls_id
        event["weights"] = self.weights.tolist()
        event["bonus"] = getattr(self, "last_bonus", 0)
        event["penalty"] = getattr(self, "last_penalty", 0)
        self.apc_state["events"].append(event)
        self._save_apc_state_supabase()

    def compute_reward_and_bonus(self):
        status_score = 0
        valid_lanes = 0
        metrics = np.zeros(4)
        MAX_VALUES = [0.2, 13.89, 300, 50]

        current_max = [
            max(0.1, max(traci.lane.getLastStepVehicleNumber(lid)/max(1, traci.lane.getLength(lid)) for lid in self.lane_ids)),
            max(5.0, max(traci.lane.getLastStepMeanSpeed(lid) for lid in self.lane_ids)),
            max(30.0, max(traci.lane.getWaitingTime(lid) for lid in self.lane_ids)),
            max(5.0, max(traci.lane.getLastStepHaltingNumber(lid) for lid in self.lane_ids))
        ]
        max_vals = [min(MAX_VALUES[i], current_max[i]) for i in range(4)]

        bonus, penalty = 0, 0
        for lane_id in self.lane_ids:
            queue, wtime, v, dens = self.get_lane_stats(lane_id)
            if queue < 0 or wtime < 0:
                continue

            metrics += [
                min(dens, max_vals[0]) / max_vals[0],
                min(v, max_vals[1]) / max_vals[1],
                min(wtime, max_vals[2]) / max_vals[2],
                min(queue, max_vals[3]) / max_vals[3]
            ]
            valid_lanes += 1
            status_score += min(queue, 50)/10 + min(wtime, 300)/60

            # Add congestion severity bonus
            if queue > 10:
                bonus += min(2.0, queue / 10.0)

        if valid_lanes == 0:
            avg_metrics = np.zeros(4)
            avg_status = 0
        else:
            avg_metrics = metrics / valid_lanes
            avg_status = status_score / valid_lanes

        if avg_status >= 5 * 1.25:
            penalty = 2
        elif avg_status <= 2.5:
            bonus += 1

        self.last_bonus = bonus
        self.last_penalty = penalty

        R = 100 * (
            -self.weights[0] * avg_metrics[0] +
            self.weights[1] * avg_metrics[1] -
            self.weights[2] * avg_metrics[2] -
            self.weights[3] * avg_metrics[3] +
            bonus - penalty
        )

        self.last_R = np.clip(R, -100, 100)
        return self.last_R, bonus, penalty
    def shutdown(self):
        """Gracefully stop the background writer and flush all state."""
        self._db_writer.stop()
        self.flush_pending_supabase_writes()
    def get_lane_stats(self, lane_id):
        try:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            waiting_time = traci.lane.getWaitingTime(lane_id)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            length = max(1.0, traci.lane.getLength(lane_id))
            density = traci.lane.getLastStepVehicleNumber(lane_id) / length
            return queue, waiting_time, mean_speed, density
        except traci.TraCIException:
            return 0, 0, 0, 0
    def adjust_weights(self, window=10):
        available = len(self.metric_history)
        if available == 0:
            self.weights = np.array([0.25] * 4)
            return
        use_win = min(window, available)
        recent = np.mean(list(self.metric_history)[-use_win:], axis=0)
        density, speed, wait, queue = recent

        speed_importance = 1 - min(speed, 1.0)
        values = np.array([
            min(density, 1.0),
            speed_importance,
            min(wait, 1.0),
            min(queue, 1.0)
        ])

        total = np.sum(values)
        if total == 0:
            self.weights = np.array([0.25] * 4)
        else:
            self.weights = values / total

        self.weight_history.append(self.weights.copy())
        print(f"[ADAPTIVE WEIGHTS] {self.tls_id}: {self.weights}")        
    def log_phase_switch(self, new_phase_idx):
        current_time = traci.simulation.getTime()
        elapsed = current_time - self.last_phase_switch_sim_time

        # Block switch if min green not met and no priority
        if elapsed < self.min_green and not self.check_priority_conditions():
            print(f"[MIN_GREEN BLOCK] Phase switch blocked (elapsed: {elapsed:.1f}s < {self.min_green}s)")
            return False

        # Flicker prevention: block if same as last phase
        if self.last_phase_idx == new_phase_idx:
            print(f"[PHASE SWITCH BLOCKED] Flicker prevention triggered for {self.tls_id}")
            return False

        # Insert yellow phase if needed
        self.insert_yellow_phase_if_needed(new_phase_idx)

        try:
            traci.trafficlight.setPhase(self.tls_id, new_phase_idx)
            traci.trafficlight.setPhaseDuration(self.tls_id, max(self.min_green, self.max_green))
            new_phase = traci.trafficlight.getPhase(self.tls_id)
            new_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            self.last_phase_idx = new_phase_idx
            self.last_phase_switch_sim_time = current_time

            # PATCH: Check if this phase was RL-created
            phase_was_rl_created = False
            phase_pkl = self.load_phase_from_supabase(new_phase_idx)
            if phase_pkl and phase_pkl.get("rl_created"):
                phase_was_rl_created = True

            event = {
                "action": "phase_switch",
                "old_phase": self.last_phase_idx,
                "new_phase": new_phase,
                "old_state": "",  # Optional: populate if needed
                "new_state": new_state,
                "reward": getattr(self, "last_R", None),
                "weights": self.weights.tolist(),
                "bonus": getattr(self, "last_bonus", 0),
                "penalty": getattr(self, "last_penalty", 0),
                "rl_created": phase_was_rl_created,
                "phase_idx": new_phase_idx
            }
            self._log_apc_event(event)
            print(f"\n[PHASE SWITCH] {self.tls_id}: {self.last_phase_idx}→{new_phase}")
            print(f"  New state: {new_state}")
            print(f"  Weights: {self.weights}, Bonus: {getattr(self, 'last_bonus', 0)}, Penalty: {getattr(self, 'last_penalty', 0)}")
            if phase_was_rl_created:
                print(f"  [INFO] RL agent's phase is now in use (phase {new_phase_idx})")
            return True
        except Exception as e:
            print(f"[ERROR] Phase switch failed: {e}")
            return False

    def check_priority_conditions(self):
        # Returns True if there is a priority event that allows preemption of min green
        # You may want to expand this as needed (emergency, protected left, etc)
        event_type, event_lane = self.check_special_events()
        if event_type == "emergency_vehicle":
            return True
        if self.serve_true_protected_left_if_needed():
            return True
        return False
    def create_phase_state(self, green_lanes=None, yellow_lanes=None, red_lanes=None):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        state = ['r'] * len(controlled_lanes)
        def set_lanes(lanes, state_char):
            if not lanes:
                return
            for lane in lanes:
                if lane in controlled_lanes:
                    idx = controlled_lanes.index(lane)
                    state[idx] = state_char
        set_lanes(green_lanes, 'G')
        set_lanes(yellow_lanes, 'y')
        set_lanes(red_lanes, 'r')
        if green_lanes:
            for lane in green_lanes:
                if lane in controlled_lanes:
                    idx = controlled_lanes.index(lane)
                    if state[idx] == 'r':
                        state[idx] = 'y'
        return "".join(state)
    def add_new_phase(self, green_lanes, green_duration=20, yellow_duration=3, yellow_lanes=None):
        print(f"[DEBUG] add_new_phase called with green_lanes={green_lanes}, green_duration={green_duration}, yellow_duration={yellow_duration}, yellow_lanes={yellow_lanes}")
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            # Check for existing phase with same green
            green_state = self.create_phase_state(green_lanes=green_lanes)
            for idx, phase in enumerate(logic.getPhases()):
                if phase.state == green_state:
                    print(f"[DEBUG] Existing phase with same green found: {idx}")
                    return idx
            phases = list(logic.getPhases())
            yellow_lanes_final = yellow_lanes if yellow_lanes is not None else green_lanes
            yellow_state = self.create_phase_state(yellow_lanes=yellow_lanes_final)
            new_green = traci.trafficlight.Phase(green_duration, green_state)
            new_yellow = traci.trafficlight.Phase(yellow_duration, yellow_state)
            phases.extend([new_green, new_yellow])
            new_logic = traci.trafficlight.Logic(
                logic.programID,
                logic.type,
                min(logic.currentPhaseIndex, len(phases)-1),
                phases
            )
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
            print(f"[DEBUG] setCompleteRedYellowGreenDefinition called for {self.tls_id} with {len(phases)} phases")
            traci.trafficlight.setPhase(self.tls_id, len(phases)-2)
            print(f"[DEBUG] setPhase called for {self.tls_id} to phase {len(phases)-2}")
            event = {
                "action": "add_new_phase",
                "tls_id": self.tls_id,
                "green_lanes": green_lanes,
                "yellow_lanes": yellow_lanes_final,
                "green_duration": green_duration,
                "yellow_duration": yellow_duration,
                "new_phase_idx": len(phases)-2,
                "green_state": green_state,
                "yellow_state": yellow_state
            }
            self._log_apc_event(event)
            print(f"[DEBUG] add_new_phase event logged: {event}")
            return len(phases)-2
        except Exception as e:
            print(f"[ERROR] Phase creation failed: {e}")
            return None
    def check_special_events(self):
        now = traci.simulation.getTime()
        if hasattr(self, "_last_special_check") and now - self._last_special_check < 1:  # Reduced from 2
            return None, None
        self._last_special_check = now
        next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
        time_left = max(0, next_switch - now)
        for lane_id in self.lane_ids:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    key = (lane_id, vid)
                    last_evt_time = self.last_emergency_event.get(key, -9999)
                    # Only log if new or enough time has passed
                    if ('emergency' in v_type or 'priority' in v_type and
                        now - last_evt_time > self.min_green):
                        self._log_apc_event({
                            "action": "emergency_vehicle",
                            "lane_id": lane_id,
                            "vehicle_id": vid,
                            "vehicle_type": v_type
                        })
                        self.last_emergency_event[key] = now
                        self.emergency_cooldown[lane_id] = now
                        self.emergency_global_cooldown = now
                        return 'emergency_vehicle', lane_id
                except traci.TraCIException:
                    continue

        if now - self.severe_congestion_global_cooldown < self.severe_congestion_global_cooldown_time:
            return None, None

        congested_lanes = []
        for lane_id in self.lane_ids:
            if now - self.severe_congestion_cooldown.get(lane_id, 0) < self.min_green / 2:  # Reduced from full min_green
                continue
            queue, _, _, _ = self.get_lane_stats(lane_id)
            if queue >= self.severe_congestion_threshold * 10:
                congested_lanes.append((lane_id, queue))

        if congested_lanes:
            lane_id, queue = max(congested_lanes, key=lambda x: x[1])
            self.severe_congestion_cooldown[lane_id] = now
            self.severe_congestion_global_cooldown = now
            return 'severe_congestion', lane_id
        return None, None
    def reorganize_or_create_phase(self, lane_id, event_type):

        try:
            is_left_turn = any(link[6] == 'l' for link in traci.lane.getLinks(lane_id))
            now = traci.simulation.getTime()
            if is_left_turn and now - self.protected_left_cooldown[lane_id] < 60:
                return False
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            if lane_id not in controlled_lanes:
                print(f"[ERROR] Lane {lane_id} not controlled")
                return False
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            target_phase = None
            for idx, phase in enumerate(logic.getPhases()):
                lane_idx = controlled_lanes.index(lane_id)
                if lane_idx < len(phase.state) and phase.state[lane_idx] in 'Gg':
                    target_phase = idx
                    break
            if target_phase is None:
                print(f"[INFO] Creating new phase for {lane_id}")
                target_phase = self.add_new_phase_for_lane(lane_id)
            if target_phase is not None:
                switched = self.log_phase_switch(target_phase)
                if switched:
                    self._log_apc_event({
                        "action": "reorganize_phase",
                        "lane_id": lane_id,
                        "event_type": event_type,
                        "phase": target_phase,
                        "weights": self.weights.tolist(),
                        "bonus": getattr(self, "last_bonus", 0),
                        "penalty": getattr(self, "last_penalty", 0)
                    })
                    if is_left_turn:
                        self.protected_left_cooldown[lane_id] = now
                return switched
            return False
        except Exception as e:
            print(f"[ERROR] Phase reorganization failed: {e}")
            return False    
    def compute_status_and_bonus_penalty(self, status_threshold=5):
        status_score = 0
        valid_lanes = 0

        for lane_id in self.lane_ids:
            queue, wtime, _, _ = self.get_lane_stats(lane_id)
            if queue < 0 or wtime < 0:
                continue
            status_score += min(queue, 50)/10 + min(wtime, 300)/60
            valid_lanes += 1

        if valid_lanes == 0:
            self.last_bonus = 0
            self.last_penalty = 0
            return 0, 0

        avg_status = status_score / valid_lanes
        bonus, penalty = 0, 0

        if avg_status >= status_threshold * 1.25:
            penalty = 2
            print(f"\n[PENALTY] Status={avg_status:.2f}")
        elif avg_status <= status_threshold / 2:
            bonus = 1
            print(f"\n[BONUS] Status={avg_status:.2f}")

        print(f"[BONUS/PENALTY] {self.tls_id}: Bonus={bonus}, Penalty={penalty}, AvgStatus={avg_status:.2f}")
        self.last_bonus = bonus
        self.last_penalty = penalty
        return bonus, penalty
    
    def get_state_vector(self):
        # Returns a state vector including phase info, all lane queue lengths, and waiting times
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0].getPhases())
        queues = [self.get_lane_stats(lane_id)[0] for lane_id in self.lane_ids]
        waits = [self.get_lane_stats(lane_id)[1] for lane_id in self.lane_ids]
        # [phase, num_phases, ...queues..., ...waits...]
        return np.array([current_phase, num_phases] + queues + waits)
    def rl_action_space(self):
        # 0: stay (extend current phase), 1..N: switch to phase i-1
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        return ['stay'] + list(range(len(logic.getPhases())))
    
    def calculate_reward(self, bonus=0, penalty=0):
        # Penalizes total queue length and waiting time for all lanes
        total_queue = sum(self.get_lane_stats(lane_id)[0] for lane_id in self.lane_ids)
        total_wait = sum(self.get_lane_stats(lane_id)[1] for lane_id in self.lane_ids)
        reward = -1.0 * total_queue - 0.2 * total_wait
        return reward
    def calculate_delta_t(self, R):
        """Calculate adaptive time adjustment with smoothing"""
        raw_delta_t = self.alpha * (R - self.R_target)
        delta_t = 10 * np.tanh(raw_delta_t / 20)
        print(f"[DELTA_T] R={R:.2f}, R_target={self.R_target:.2f}, Δt={delta_t:.2f}")
        return np.clip(delta_t, -10, 10)

    def update_R_target(self, window=10):
        if len(self.reward_history) < window or self.phase_count % 10 != 0:
            return
        avg_R = np.mean(list(self.reward_history)[-window:])
        self.R_target = self.r_base + self.r_adjust * (avg_R - self.r_base)
        print(f"\n[TARGET UPDATE] R_target={self.R_target:.2f} (avg={avg_R:.2f})")

    def add_new_phase_for_lane(self, lane_id, green_duration=None, yellow_duration=3):
        """Create phase for single lane, log RL-creation intent."""
        idx = self.add_new_phase(
            green_lanes=[lane_id],
            green_duration=green_duration or self.max_green,
            yellow_duration=yellow_duration
        )
        # PATCH: Log explicit RL phase creation for audit trail
        self._log_apc_event({
            "action": "add_new_phase_for_lane",
            "lane_id": lane_id,
            "phase_idx": idx,
            "rl_created": True,
        })
        return idx

    def get_protected_left_lanes(self):
        """Return list of lanes that serve as protected left turns."""
        protected_lefts = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            for link in traci.lane.getLinks(lane_id):
                if link[6] == 'l':
                    protected_lefts.append(lane_id)
                    break
        return protected_lefts
    def add_protected_left_phase(self, green_duration=None, yellow_duration=3):
        """Create a phase where only protected left lanes have green."""
        left_lanes = self.get_protected_left_lanes()
        if not left_lanes:
            print("[INFO] No protected left lanes to add phase for.")
            return None
        return self.add_new_phase(
            green_lanes=left_lanes,
            green_duration=green_duration or self.max_green,
            yellow_duration=yellow_duration
        )
    def create_new_protected_left_phase(self, lane_id=None):
        """Create dedicated left turn phase (for one or all left lanes)."""
        try:
            if lane_id is not None:
                return self.add_new_phase(
                    green_lanes=[lane_id],
                    green_duration=self.max_green,
                    yellow_duration=3
                )
            else:
                return self.add_protected_left_phase()
        except Exception as e:
            print(f"[ERROR] Protected left creation failed: {e}")
            return None
    def trigger_protected_left_if_blocked(self):
        """Detect and handle blocked left turns"""
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            current_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            current_time = self._now()
            if any(current_time - cooldown < 60 for cooldown in self.protected_left_cooldown.values()):
                return False
            for lane_idx, lane_id in enumerate(controlled_lanes):
                if lane_idx >= len(current_state) or current_state[lane_idx] not in 'Gg':
                    continue

                if not any(link[6] == 'l' for link in traci.lane.getLinks(lane_id)):
                    continue

                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                if not vehicles:
                    continue

                front_veh = vehicles[0]
                if traci.vehicle.getSpeed(front_veh) > 0.1:
                    continue

                left_phase = self.create_new_protected_left_phase(lane_id)
                if left_phase is None or left_phase == current_phase:
                    continue

                self.log_phase_switch(left_phase)
                self._log_apc_event({
                    "action": "activate_protected_left",
                    "lane_id": lane_id,
                    "vehicle_id": front_veh,
                    "phase": left_phase,
                    "weights": self.weights.tolist(),
                    "bonus": getattr(self, "last_bonus", 0),
                    "penalty": getattr(self, "last_penalty", 0)
                })
                return True

        except traci.TraCIException as e:
            print(f"[ERROR] Protected left check failed: {e}")
        return False


    def detect_blocked_left_turn(self):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        current_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        for lane_idx, lane_id in enumerate(controlled_lanes):
            # Only check if lane is currently green
            if current_state[lane_idx] not in 'Gg':
                continue
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            if not vehicles:
                continue
            front_vehicle = vehicles[0]
            speed = traci.vehicle.getSpeed(front_vehicle)
            stopped_time = traci.vehicle.getAccumulatedWaitingTime(front_vehicle)
            # If front vehicle stopped for > threshold and green, treat as blocked
            if speed < 0.2 and stopped_time > 5:
                return lane_id
        return None
    def _is_vehicle_on_left_turn(self, lane_id, vehicle_id):
        """Check if the vehicle is intending to turn left from this lane."""
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            edge = traci.lane.getEdgeID(lane_id)
            idx = route.index(edge) if edge in route else -1
            # If next movement is a left, return True
            # (This is an approximation; SUMO might not give explicit turn type)
            # Could be improved with more context
            return True if idx >= 0 else False
        except Exception:
            return False
    def get_conflicting_straight_lanes(self, left_lane):
        """
        Given a left turn lane, return the controlled lanes that are straight in the opposing direction.
        Assumes SUMO lane connections are available.
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        # Identify edge and direction of left_lane
        left_edge = traci.lane.getEdgeID(left_lane)
        left_links = traci.lane.getLinks(left_lane)
        # Find the outgoing edge for left turn link
        left_turn_targets = [link[0] for link in left_links if len(link) > 6 and link[6] == 'l']
        # For each controlled lane, check if it connects to the opposing (straight) edge
        conflicting_straight = []
        for lane in controlled_lanes:
            for link in traci.lane.getLinks(lane):
                # Straight movement: if direction is 's' or 'S' and the link's target edge is in left_turn_targets
                if len(link) > 6 and link[6] in ['s', 'S']:
                    if link[0] in left_turn_targets:
                        conflicting_straight.append(lane)
        return conflicting_straight
    

    def preload_phases_from_sumo(self):
        for idx, phase in enumerate(self._phase_defs):
            if not any(p['phase_idx'] == idx for p in self.apc_state.get('phases', [])):
                self.save_phase_to_supabase(
                    phase_idx=idx,
                    duration=phase.duration,
                    state_str=phase.state,
                    delta_t=0,
                    raw_delta_t=0,
                    penalty=0
                )

    def update_phase_duration_record(self, phase_idx, new_duration, extended_time=0):
        new_duration = min(new_duration, MAX_PHASE_DURATION)
        updated = False
        for p in self.apc_state.get("phases", []):
            if p["phase_idx"] == phase_idx:
                p["duration"] = new_duration
                p["extended_time"] = extended_time
                updated = True
        if updated:
            self._save_apc_state_supabase()
        self._log_apc_event({
            "action": "phase_duration_update",
            "phase_idx": phase_idx,
            "duration": new_duration,
            "extended_time": extended_time,
            "tls_id": self.tls_id
        })

    def set_phase_from_pkl(self, phase_idx, requested_duration=None):
        print(f"[DB]/[Supabase][SET] set_phase_from_pkl({phase_idx}, requested_duration={requested_duration}) called")
        phase_record = self.load_phase_from_supabase(phase_idx)
        if phase_record:
            duration = requested_duration if requested_duration is not None else phase_record["duration"]
            duration = min(duration, MAX_PHASE_DURATION)  # CAP here
            extended_time = duration - phase_record.get("duration", duration) if requested_duration is not None else 0
            traci.trafficlight.setPhase(self.tls_id, phase_record["phase_idx"])
            traci.trafficlight.setPhaseDuration(self.tls_id, duration)
            self.update_phase_duration_record(phase_record["phase_idx"], duration, extended_time)            
            if hasattr(self, "update_display"):
                current_time = traci.simulation.getTime()
                next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
                self.update_display(phase_idx, duration, current_time, next_switch, extended_time)
            return True
        else:
            print(f"[APC-RL] Phase {phase_idx} not found in PKL, attempting to create or substitute.")
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            phases = logic.getPhases()
            if 0 <= phase_idx < len(phases):
                # CAP fallback duration
                phase_duration = min(phases[phase_idx].duration, MAX_PHASE_DURATION)
                self.save_phase_to_supabase(
                    phase_idx=phase_idx,
                    duration=phase_duration,
                    state_str=phases[phase_idx].state,
                    delta_t=0,
                    raw_delta_t=0,
                    penalty=0
                )
                print(f"[DB]/[Supabase][AUTO-SAVE] Saved missing SUMO phase {phase_idx} to Supabase.")
                return self.set_phase_from_pkl(phase_idx, requested_duration=phase_duration)
            if 0 <= phase_idx < len(phases):
                target_state = phases[phase_idx].state
                for p in self.apc_state.get("phases", []):
                    if p["state"] == target_state:
                        print(f"[DB]/[Supabase][SUBSTITUTE] Found Supabase phase with matching state. Using phase_idx={p['phase_idx']}.")
                        traci.trafficlight.setPhase(self.tls_id, p["phase_idx"])
                        traci.trafficlight.setPhaseDuration(self.tls_id, p["duration"])
                        self.update_phase_duration_record(p["phase_idx"], p["duration"])
                        if hasattr(self, "update_display"):
                            self.update_display(p["phase_idx"], p["duration"])
                        return True
            if len(phases) > 0:
                fallback_idx = phase_idx if 0 <= phase_idx < len(phases) else 0
                fallback_state = phases[fallback_idx].state
                fallback_duration = min(self.max_green, MAX_PHASE_DURATION)
                self.save_phase_to_supabase(
                    phase_idx=phase_idx,
                    duration=fallback_duration,
                    state_str=fallback_state,
                    delta_t=0,
                    raw_delta_t=0,
                    penalty=0
                )
                print(f"[DB]/[Supabase][FALLBACK] Created and saved fallback phase {phase_idx} with max_green to Supabase.")
                traci.trafficlight.setPhase(self.tls_id, phase_idx)
                traci.trafficlight.setPhaseDuration(self.tls_id, fallback_duration)
                self.update_phase_duration_record(phase_idx, fallback_duration)
                if hasattr(self, "update_display"):
                    self.update_display(phase_idx, fallback_duration)
                return True
            print(f"[ERROR] Could not find or create a suitable phase for index {phase_idx}.")
            return False

    def ensure_true_protected_left_phase(self, left_lane):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        left_idx = controlled_lanes.index(left_lane)
        conflicting_straight = self.get_conflicting_straight_lanes(left_lane)

        # Compose state: green for left_lane, red for conflicting straight lanes, red for others
        protected_state = []
        for i, lane in enumerate(controlled_lanes):
            if i == left_idx:
                protected_state.append('G')
            elif lane in conflicting_straight:
                protected_state.append('r')
            else:
                # you could allow parallel non-conflicting greens if desired, but safest is red
                protected_state.append('r')
        protected_state_str = ''.join(protected_state)
        yellow_state_str = ''.join('y' if i == left_idx else 'r' for i in range(len(controlled_lanes)))

        # Check if phase exists
        for idx, phase in enumerate(logic.getPhases()):
            if phase.state == protected_state_str:
                return idx

        # Otherwise, create and append
        green_phase = traci.trafficlight.Phase(self.max_green, protected_state_str)
        yellow_phase = traci.trafficlight.Phase(3, yellow_state_str)
        phases = list(logic.getPhases()) + [green_phase, yellow_phase]
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type, len(phases) - 2, phases
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
        print(f"[TRUE PROTECTED LEFT PHASE] Added for lane {left_lane} at {self.tls_id}")
        self._log_apc_event({
            "action": "add_true_protected_left_phase",
            "lane_id": left_lane,
            "green_state": protected_state_str,
            "yellow_state": yellow_state_str,
            "phase_idx": len(phases) - 2
        })
        return len(phases) - 2
    def create_true_protected_left_phase(self, left_lane):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        left_idx = controlled_lanes.index(left_lane)
        protected_state = ''.join(['G' if i == left_idx else 'r' for i in range(len(controlled_lanes))])
        yellow_state = ''.join(['y' if i == left_idx else 'r' for i in range(len(controlled_lanes))])
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        for idx, phase in enumerate(logic.getPhases()):
            if phase.state == protected_state:
                return idx
        green_phase = traci.trafficlight.Phase(self.max_green, protected_state)
        yellow_phase = traci.trafficlight.Phase(5, yellow_state)  # Increased from 3 for safety
        phases = list(logic.getPhases()) + [green_phase, yellow_phase]
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type, len(phases) - 2, phases
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
        self._log_apc_event({
            "action": "add_true_protected_left_phase",
            "lane_id": left_lane,
            "green_state": protected_state,
            "yellow_state": yellow_state,
            "phase_idx": len(phases) - 2
        })
        return len(phases) - 2
    def detect_blocked_left_turn_conflict(self):
        """
        Detect if any left-turn lane is blocked by conflicting straight traffic.
        If found, returns (lane_id, True) for blocked left turn needing protection.
        """
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            for lane_id in controlled_lanes:
                # Check if lane is a left-turn lane
                links = traci.lane.getLinks(lane_id)
                is_left = any(len(link) > 6 and link[6] == 'l' for link in links)
                if not is_left:
                    continue

                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                if not vehicles:
                    continue

                # If all vehicles have speed < 0.2, consider blocked
                speeds = [traci.vehicle.getSpeed(vid) for vid in vehicles]
                if speeds and max(speeds) < 0.2:
                    # Now check if conflicting straight lanes have queue
                    conflicting_straight = self.get_conflicting_straight_lanes(lane_id)
                    if any(traci.lane.getLastStepVehicleNumber(l) > 0 for l in conflicting_straight):
                        return lane_id, True
            return None, False
        except Exception as e:
            print(f"[ERROR] Blocked left turn conflict detection failed: {e}")
            return None, False

    def serve_true_protected_left_if_needed(self):
        lane_id, needs_protection = self.detect_blocked_left_turn_conflict()
        if not needs_protection:
            return False

        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        wait = traci.lane.getWaitingTime(lane_id)
        print(f"[DEBUG] [PROTECTED_LEFT_TURN] Detected blocked left lane {lane_id}. Queue={queue}, Wait={wait}")

        phase_idx = self.create_true_protected_left_phase(lane_id)
        if phase_idx is None:
            print(f"[DEBUG] [PROTECTED_LEFT_TURN] No protected left phase index could be created for lane {lane_id}.")
            return False

        now = traci.simulation.getTime()
        # Use log_phase_switch and only proceed if switch succeeded
        switched = self.log_phase_switch(phase_idx)
        if not switched:
            return False

        # Dynamically set phase duration based on queue/wait
        green_duration = min(self.max_green, max(self.min_green, queue * 2 + wait * 0.1))
        traci.trafficlight.setPhaseDuration(self.tls_id, green_duration)
        print(f"[DEBUG] [PROTECTED_LEFT_TURN] Phase {phase_idx} activated for lane {lane_id} at {self.tls_id} for {green_duration:.1f}s (queue={queue}, wait={wait})")
        self._log_apc_event({
            "action": "serve_true_protected_left",
            "lane_id": lane_id,
            "phase": phase_idx,
            "green_duration": green_duration,
            "queue": queue,
            "wait": wait
        })
        # timers already updated in log_phase_switch
        return True    
    
    def insert_yellow_phase_if_needed(self, next_phase_idx):
        """
        When switching phases, if any lane is going from green to red,
        insert a yellow phase before switching.
        """
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            current_state = logic.getPhases()[current_phase].state
            target_state = logic.getPhases()[next_phase_idx].state
            n = len(current_state)
            yellow_needed = False
            yellow_state = ['r'] * n
            for i in range(n):
                if current_state[i].upper() == 'G' and target_state[i].upper() == 'R':
                    yellow_needed = True
                    yellow_state[i] = 'y'
                else:
                    yellow_state[i] = target_state[i]

            if yellow_needed:
                # Check if this yellow phase exists
                for idx, phase in enumerate(logic.getPhases()):
                    if phase.state == ''.join(yellow_state):
                        # Set yellow phase before switching
                        traci.trafficlight.setPhase(self.tls_id, idx)
                        traci.trafficlight.setPhaseDuration(self.tls_id, 3)
                        print(f"[YELLOW INSERT] Inserted yellow phase before switch at {self.tls_id}")
                        time.sleep(0.2)  # Let yellow phase run (small delay, adjust if needed)
                        break
                else:
                    # Create new yellow phase if not found
                    yellow_phase = traci.trafficlight.Phase(3, ''.join(yellow_state))
                    phases = list(logic.getPhases()) + [yellow_phase]
                    new_logic = traci.trafficlight.Logic(
                        logic.programID, logic.type, len(phases) - 1, phases
                    )
                    traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
                    traci.trafficlight.setPhase(self.tls_id, len(phases) - 1)
                    traci.trafficlight.setPhaseDuration(self.tls_id, 3)
                    print(f"[YELLOW INSERT] Created & inserted new yellow phase before switch at {self.tls_id}")
                    time.sleep(0.2)
        except Exception as e:
            print(f"[ERROR] Yellow phase insertion failed: {e}")


    def enforce_min_green(self):
        """Return True if enough time has passed since last phase switch."""
        current_sim_time = traci.simulation.getTime()
        elapsed = current_sim_time - self.last_phase_switch_sim_time
        if elapsed < self.min_green:
            print(f"[MIN_GREEN ENFORCED] {self.tls_id}: Only {elapsed:.2f}s since last switch, min_green={self.min_green}s")
            return False
        return True


    def adjust_phase_duration(self, phase_idx, delta_t):
        """
        Schedule a phase duration adjustment for the given phase (by index).
        The change will be applied the next time this phase becomes active,
        unless it's an emergency/priority situation.
        """
        try:
            # Only allow immediate adjustment for priority/emergency
            is_priority = self.check_priority_conditions()
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            if not is_priority and phase_idx == current_phase:
                # Not priority: don't allow changing the duration of the currently running phase
                print("[ADJUST BLOCKED] Wait for phase to finish before applying duration change.")
                return traci.trafficlight.getPhaseDuration(self.tls_id)

            # Get base duration from DB/PKL if possible
            phase_record = self.load_phase_from_supabase(phase_idx) if hasattr(self, "load_phase_from_supabase") else None
            if phase_record and "duration" in phase_record:
                base_duration = phase_record["duration"]
            else:
                # fallback: get from SUMO (may not be accurate if not active)
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
                base_duration = logic.getPhases()[phase_idx].duration if phase_idx < len(logic.getPhases()) else self.min_green

            new_duration = np.clip(base_duration + delta_t, self.min_green, min(self.max_green, MAX_PHASE_DURATION))
            extended_time = new_duration - base_duration
            self.last_extended_time = extended_time

            # Schedule the change if not currently running, or apply immediately if allowed (priority)
            if is_priority and (phase_idx == current_phase):
                traci.trafficlight.setPhaseDuration(self.tls_id, new_duration)
                print(f"[ADJUST] (PRIORITY) Set duration for CURRENT phase {phase_idx} to {new_duration}s immediately.")
            else:
                self.schedule_phase_duration_update(phase_idx, new_duration)
                print(f"[ADJUST] Scheduled duration for phase {phase_idx} to be {new_duration}s at next activation.")

            self.update_phase_duration_record(phase_idx, new_duration, extended_time)
            event = {
                "action": "adjust_phase_duration",
                "phase": phase_idx,
                "delta_t": delta_t,
                "old_duration": base_duration,
                "new_duration": new_duration,
                "duration": new_duration,
                "extended_time": extended_time,
                "reward": self.last_R,
                "weights": self.weights.tolist(),
                "bonus": getattr(self, "last_bonus", 0),
                "penalty": getattr(self, "last_penalty", 0)
            }
            self._log_apc_event(event)
            print(f"\n[PHASE ADJUST] Phase {phase_idx}: {base_duration:.1f}s → {new_duration:.1f}s (Δ {extended_time:.1f}s) [scheduled={not (is_priority and (phase_idx == current_phase))}]")
            if hasattr(self, "update_display"):
                current_time = traci.simulation.getTime()
                next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
                self.update_display(phase_idx, new_duration, current_time, next_switch, extended_time)
            return new_duration
        except traci.TraCIException as e:
            print(f"[ERROR] Duration adjustment failed: {e}")
            return traci.trafficlight.getPhaseDuration(self.tls_id)
    def control_step(self):
        """Adaptive RL-based control step: only schedule phase duration changes, never change current phase's duration except for emergencies.
        - Machine learning agent chooses next phase and duration.
        - Phase duration changes are only applied at the start of a phase (never mid-phase), except for emergency/priority vehicles.
        """
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        self.apply_scheduled_duration_if_needed(current_phase)
        self.phase_count += 1
        now = traci.simulation.getTime()
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        num_phases = len(logic.getPhases())
        min_green_elapsed = (now - self.last_phase_switch_sim_time) >= self.min_green

        # 1. Priority: Emergency vehicles/preemption
        event_type, event_lane = self.check_special_events()
        if event_type == 'emergency_vehicle':
            self.reorganize_or_create_phase(event_lane, event_type)
            self.last_served_time[current_phase] = now
            if hasattr(self, 'prev_state') and hasattr(self, 'prev_action'):
                reward = self.calculate_reward() - 100
                self.rl_agent.update_q_table(self.prev_state, self.prev_action, reward, self.prev_state, tl_id=self.tls_id)
            return

        # 2. Handle protected lefts only if min green elapsed or priority
        if (min_green_elapsed or self.check_priority_conditions()):
            if self.serve_true_protected_left_if_needed():
                return

        # 3. If not yet time to switch, just wait
        next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
        time_left = max(0, next_switch - now)
        if time_left > 0.1:
            return

        # 4. RL chooses next phase (and duration) -- MACHINE LEARNING LOGIC
        # State vector can be made richer if desired
        state = self.get_state_vector()
        actions = self.rl_action_space()
        try:
            action_idx = self.rl_agent.get_action(state, tl_id=self.tls_id, action_size=len(actions))
        except Exception as e:
            print(f"[RL] Fallback to round-robin due to error: {e}")
            action_idx = (current_phase + 1) % num_phases

        # 5. Compute reward and adaptive duration for next phase
        bonus, penalty = self.compute_status_and_bonus_penalty()
        R = self.calculate_reward(bonus, penalty)
        self.reward_history.append(R)
        if self.phase_count % 10 == 0:
            self.update_R_target()
        delta_t = self.calculate_delta_t(R)

        # 6. Only schedule duration change for the chosen phase (never current phase unless priority)
        if action_idx == 0 or actions[action_idx] == current_phase:
            # Stay: do nothing, let current phase run out. (No extension.)
            pass
        else:
            target_phase = actions[action_idx]
            # Get base duration for the target phase
            phase_record = self.load_phase_from_supabase(target_phase)
            if phase_record:
                base = phase_record.get("base_duration", phase_record.get("duration", self.min_green))
            else:
                base = self.min_green
            next_duration = np.clip(base + delta_t, self.min_green, self.max_green)
            # SCHEDULE: only update when phase becomes active (not immediately)
            self.schedule_phase_duration_update(target_phase, next_duration)

            # Switch phase using PKL (will apply scheduled duration)
            self.set_phase_from_pkl(target_phase)
            self.last_served_time[target_phase] = now
            self.last_phase_switch_sim_time = now

        # 7. Update RL Q-table
        if hasattr(self, 'prev_state') and hasattr(self, 'prev_action'):
            self.rl_agent.update_q_table(self.prev_state, self.prev_action, R, state, tl_id=self.tls_id)
        self.prev_state = state
        self.prev_action = action_idx

    # --- Needed from above for patch to work ---
    def schedule_phase_duration_update(self, phase_idx, new_duration):
        """Schedule a duration update for the given phase, to be applied the next time that phase becomes active."""
        if not hasattr(self, '_scheduled_phase_updates'):
            self._scheduled_phase_updates = {}
        self._scheduled_phase_updates[phase_idx] = new_duration

    def apply_scheduled_duration_if_needed(self, phase_idx):
        if hasattr(self, '_scheduled_phase_updates') and phase_idx in self._scheduled_phase_updates:
            duration = self._scheduled_phase_updates.pop(phase_idx)
            duration = min(duration, MAX_PHASE_DURATION)
            traci.trafficlight.setPhaseDuration(self.tls_id, duration)
            self._log_apc_event({
                "action": "apply_scheduled_duration",
                "tls_id": self.tls_id,
                "phase_idx": phase_idx,
                "applied_duration": duration,
                "sim_time": traci.simulation.getTime(),
            })
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
        self.logger = logging.getLogger(__name__)

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

    def get_action(self, state, tl_id=None, action_size=None, strategy="epsilon_greedy"):
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
                        self.logger.error(f"Error saving model: {e}")
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

    def __init__(self, sumocfg_path=None, mode="train", config=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.mode = mode
        self.step_count = 0
        self.current_episode = 0
        self.max_consecutive_left = 1
        self.subscribed_vehicles = set()
        self.left_turn_lanes = set()
        self.right_turn_lanes = set()
        self.lane_id_list = []
        self.lane_id_to_idx = {}
        self.lane_to_tl = {}
        self.tl_action_sizes = {}
        self.pending_next_phase = {}
        self.lane_lengths = {}
        self.phase_event_log_file = f"phase_event_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.phase_events = []
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
        self.adaptive_phase_controllers = {}
        self.norm_bounds = {
            'queue': 20, 'wait': 60, 'speed': 13.89,
            'flow': 30, 'density': 0.2, 'arrival_rate': 5,
            'time_since_green': 120
        }
        self.lane_scores = defaultdict(float)
        self.lane_states = defaultdict(str)
        self.consecutive_states = defaultdict(int)
        self.last_arrival_time = defaultdict(lambda: 0.0)
        self.last_lane_vehicles = defaultdict(set)
        self.last_green_time = defaultdict(float)

        # PATCH: create adaptive_phase_controller and RL agent, then link them
        lane_ids = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        tls_id = traci.trafficlight.getIDList()[0]
        self.adaptive_phase_controller = AdaptivePhaseController(
            lane_ids=lane_ids,
            tls_id=tls_id,
            alpha=1.0,
            min_green=10,
            max_green=60
        )
        self.apc = self.adaptive_phase_controller  # <== for convenience
        self.rl_agent = EnhancedQLearningAgent(
            state_size=12,
            action_size=8,
            adaptive_controller=self.adaptive_phase_controller,
            mode=mode
        )
        self.adaptive_phase_controller.rl_agent = self.rl_agent
        self.rl_agent.load_model()
        self.adaptive_params = {
            'min_green': 30, 'max_green': 80, 'starvation_threshold': 40,
            'reward_scale': 40, 'queue_weight': 0.6, 'wait_weight': 0.3,
            'flow_weight': 0.5, 'speed_weight': 0.2, 'left_turn_priority': 1.2,
            'empty_green_penalty': 15, 'congestion_bonus': 10
        }

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

    def get_vehicle_classes(self, vehicle_ids):
        classes = {}
        for vid in vehicle_ids:
            results = traci.vehicle.getSubscriptionResults(vid)
            if results and traci.constants.VAR_VEHICLECLASS in results:
                classes[vid] = results[traci.constants.VAR_VEHICLECLASS]
            else:
                try:
                    classes[vid] = traci.vehicle.getVehicleClass(vid)
                except traci.TraCIException:
                    classes[vid] = None
        return classes

    def subscribe_lanes(self, lane_ids):
        for lid in lane_ids:
            traci.lane.subscribe(lid, [
                traci.constants.LAST_STEP_VEHICLE_NUMBER,
                traci.constants.LAST_STEP_MEAN_SPEED,
                traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                traci.constants.LAST_STEP_VEHICLE_ID_LIST
            ])

    def detect_turning_lanes(self):
        left, right = set(), set()
        for lid in self.lane_id_list:

            for c in traci.lane.getLinks(lid):
                idx = 6 if len(c) > 6 else 3 if len(c) > 3 else None
                if idx and c[idx] == 'l':
                    left.add(lid)
                if idx and c[idx] == 'r':
                    right.add(lid)
        return left, right

    def initialize(self):
        # Print phase info for each traffic light
        for tl_id in traci.trafficlight.getIDList():
            phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            for i, phase in enumerate(phases):
                print(f"  Phase {i}: {phase.state} (duration {getattr(phase, 'duration', '?')})")

        # Main lane and traffic light setup
        self.lane_id_list = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        self.tl_action_sizes = {tl_id: len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                                for tl_id in traci.trafficlight.getIDList()}
        # Setup RL agent and AdaptivePhaseControllers
        tls_list = traci.trafficlight.getIDList()
        if tls_list:
            # Create RL agent using the first intersection
            tls_id = tls_list[0]
            lane_ids = traci.trafficlight.getControlledLanes(tls_id)
            self.adaptive_phase_controller = AdaptivePhaseController(
                lane_ids=lane_ids,
                tls_id=tls_id,
                alpha=1.0,
                min_green=10,
                max_green=60
            )
            self.rl_agent = EnhancedQLearningAgent(
                state_size=12,
                action_size=self.tl_action_sizes[tls_id],
                adaptive_controller=self.adaptive_phase_controller,
                mode=self.mode
            )
            self.adaptive_phase_controller.rl_agent = self.rl_agent
        # Setup AdaptivePhaseController for each intersection
        self.adaptive_phase_controllers = {}
        for tls_id in tls_list:
            lane_ids = traci.trafficlight.getControlledLanes(tls_id)
            apc = AdaptivePhaseController(
                lane_ids=lane_ids,
                tls_id=tls_id,
                alpha=1.0,
                min_green=10,
                max_green=60
            )
            apc.rl_agent = self.rl_agent
            self.adaptive_phase_controllers[tls_id] = apc

        # Lanes and topology setup
        self.lane_id_to_idx = {lid: i for i, lid in enumerate(self.lane_id_list)}
        self.idx_to_lane_id = dict(enumerate(self.lane_id_list))
        for lid in self.lane_id_list:
            self.last_green_time[lid] = 0.0
        self.subscribe_lanes(self.lane_id_list)
        self.left_turn_lanes, self.right_turn_lanes = self.detect_turning_lanes()
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
        """Return True if any vehicle is in the dilemma zone for lanes about to switch from green to red.
        PATCH: Avoid tuple index out of range if phase state length < controlled_lanes."""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return False
            current_phase_index = traci.trafficlight.getPhase(tl_id)
            phases = getattr(logic, "phases", None)
            if phases is None or current_phase_index >= len(phases) or current_phase_index < 0:
                print(f"Error in _is_in_dilemma_zone: current_phase_index {current_phase_index} out of range for {tl_id} (phases: {len(phases) if phases else 'N/A'})")
                return False
            state = phases[current_phase_index].state
            n = min(len(state), len(controlled_lanes))
            for lane_idx in range(n):
                lane = controlled_lanes[lane_idx]
                if state[lane_idx].upper() == 'G':
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
        Improved: Check for blocked left-turn lanes and serve a protected left phase if needed.
        PATCH: Set protected left phase and duration using APC PKL information.
        """
        try:
            apc = self.adaptive_phase_controllers.get(tl_id, self.apc)
            # Find left-turn lanes needing service (blocked: queue > 0, all vehicles stopped)
            left_candidates = []
            for lane in controlled_lanes:
                # Is left-turn?
                links = traci.lane.getLinks(lane)
                is_left = any(len(link) > 6 and link[6] == 'l' for link in links)
                if not is_left:
                    continue
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                if not vehicles:
                    continue
                speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
                if speeds and max(speeds) < 0.2 and lane_data.get(lane, {}).get('queue_length', 0) > 0:
                    left_candidates.append((lane, len(vehicles)))

            if not left_candidates:
                return False

            # Most blocked left
            target, _ = max(left_candidates, key=lambda x: x[1])
            logic = self._get_traffic_light_logic(tl_id)
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            left_idx = controlled_lanes.index(target)

            # Check for existing protected left phase (green for target, red for all others)
            for idx, phase in enumerate(logic.phases):
                if all((ch == 'G' if i == left_idx else ch in 'rR') for i, ch in enumerate(phase.state)):
                    phase_idx = idx
                    break
            else:
                # Create it if not present
                protected_state = ''.join('G' if i == left_idx else 'r' for i in range(len(controlled_lanes)))
                yellow_state = ''.join('y' if i == left_idx else 'r' for i in range(len(controlled_lanes)))
                green_phase = traci.trafficlight.Phase(self.adaptive_params['max_green'], protected_state)
                yellow_phase = traci.trafficlight.Phase(3, yellow_state)
                phases = list(logic.phases) + [green_phase, yellow_phase]
                new_logic = traci.trafficlight.Logic(
                    logic.programID, logic.type, len(phases) - 2, phases
                )
                traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_id, new_logic)
                phase_idx = len(phases) - 2
                print(f"[NEW PROTECTED LEFT] Added at {tl_id} for lane {target}")

            current_phase = traci.trafficlight.getPhase(tl_id)
            # Flicker prevention: only switch if not already in protected left or min green elapsed
            min_green = self.adaptive_params['min_green']
            last_change = self.last_phase_change.get(tl_id, -9999)
            if current_phase == phase_idx and (current_time - last_change < min_green):
                return False

            # PATCH: Use phase record from APC PKL to set SUMO phase/duration
            phase_record = apc.load_phase_from_supabase(phase_idx)
            if phase_record:
                traci.trafficlight.setPhase(apc.tls_id, phase_record["phase_idx"])
                traci.trafficlight.setPhaseDuration(apc.tls_id, phase_record["duration"])
            else:
                traci.trafficlight.setPhase(apc.tls_id, phase_idx)
                traci.trafficlight.setPhaseDuration(apc.tls_id, apc.max_green)
            
            if phase_record:
                traci.trafficlight.setPhase(self.adaptive_controller.tls_id, phase_record["phase_idx"])
                traci.trafficlight.setPhaseDuration(self.adaptive_controller.tls_id, phase_record["duration"])
                # Add display update here
                if hasattr(self.adaptive_controller, 'update_display'):
                    self.adaptive_controller.update_display(phase_record["phase_idx"], phase_record["duration"])

            self.last_phase_change[tl_id] = current_time
            self.phase_utilization[(tl_id, phase_idx)] = self.phase_utilization.get((tl_id, phase_idx), 0) + 1
            print(f"[PROTECTED LEFT] Served at {tl_id} for lane {target} (phase {phase_idx})")
            self.rl_agent.training_data.append({
                'event': 'protected_left_served',
                'lane_id': target,
                'tl_id': tl_id,
                'phase': phase_idx,
                'simulation_time': current_time
            })
            return True
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
            max_queue = 0
            for lane in controlled_lanes:
                queue = lane_data.get(lane, {}).get('queue_length', 0)
                max_queue = max(max_queue, queue)
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    res = traci.vehicle.getSubscriptionResults(vid)
                    if res and traci.constants.VAR_SPEED in res:
                        s = res[traci.constants.VAR_SPEED]
                        if s > max_speed:
                            max_speed = s
            yellow_time = max(3.0, min(8.0, 1.0 + max_speed / (2 * 3.0) + max_queue / 10.0))  # Extended based on congestion
            return yellow_time
        except Exception as e:
            print(f"Error in _calculate_adaptive_yellow: {e}")
            return 5.0
    
    def _add_new_green_phase_for_lane(self, tl_id, lane_id, min_green=None, yellow=3):
        """Creates and appends a new green+yellow phase for a specific lane, pushes to TraCI, updates caches, and returns the new phase index."""
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
        phases = list(logic.getPhases())
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        phase_state = ['r'] * len(controlled_lanes)
        try:
            lane_idx = controlled_lanes.index(lane_id)
        except ValueError:
            print(f"[ERROR] Lane {lane_id} not controlled by {tl_id}")
            return None

        min_green = min_green or self.adaptive_params.get('min_green', 10)

        # Green phase for the target lane
        phase_state[lane_idx] = 'G'
        new_green_phase = traci.trafficlight.Phase(min_green, "".join(phase_state), 0, 0)
        # Yellow phase for the target lane
        phase_state[lane_idx] = 'y'
        new_yellow_phase = traci.trafficlight.Phase(yellow, "".join(phase_state), 0, 0)
        # Add both phases to the end
        phases.append(new_green_phase)
        phases.append(new_yellow_phase)
        # Update logic and push to SUMO
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type, len(phases) - 2, phases
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_id, new_logic)
        print(f"[NEW PHASE] Added green+yellow phase for lane {lane_id} at {tl_id}")

        # Force update of phase logic cache
        self.tl_logic_cache[tl_id] = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        # Update action space for RL agent and controller
        self.tl_action_sizes[tl_id] = len(self.tl_logic_cache[tl_id].phases)
        self.rl_agent.action_size = max(self.tl_action_sizes.values())
        print(f"[ACTION SPACE] tl_id={tl_id} now n_phases={self.tl_action_sizes[tl_id]}")
        return len(phases) - 2  # return index of new green phase

    def run_step(self):
        try:
            self.step_count += 1
            current_time = traci.simulation.getTime()
            self.intersection_data = {}

            # Call control_step for each intersection's AdaptivePhaseController
            for tls_id, apc in self.adaptive_phase_controllers.items():
                apc.control_step()

            # Defensive re-initialization of APCs if needed (for dynamic networks)
            for tls_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(tls_id)
                if tls_id not in self.adaptive_phase_controllers:
                    self.adaptive_phase_controllers[tls_id] = AdaptivePhaseController(
                        lane_ids=lanes,
                        tls_id=tls_id,
                        alpha=1.0,
                        min_green=10,
                        max_green=60
                    )
            all_vehicles = set(traci.vehicle.getIDList())
            vehicle_classes = self.get_vehicle_classes(all_vehicles)
            lane_data = self._collect_enhanced_lane_data(vehicle_classes, all_vehicles)
            self.subscribed_vehicles.intersection_update(all_vehicles)
            new_vehicles = all_vehicles - self.subscribed_vehicles
            self.subscribe_vehicles(new_vehicles)
            self.subscribed_vehicles.update(new_vehicles)

            for tl_id in traci.trafficlight.getIDList():
                apc = self.adaptive_phase_controllers[tl_id]
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                logic = self._get_traffic_light_logic(tl_id)
                current_phase = traci.trafficlight.getPhase(tl_id)

                # Handle pending phase transitions
                if tl_id in self.pending_next_phase:
                    pending_phase, set_time = self.pending_next_phase[tl_id]
                    n_phases = len(logic.phases) if logic else 0
                    if logic and 0 <= current_phase < n_phases:
                        phase_duration = logic.phases[current_phase].duration
                    else:
                        phase_duration = 3
                    if n_phases == 0 or pending_phase >= n_phases or pending_phase < 0:
                        print(f"[WARNING] Pending phase {pending_phase} for {tl_id} is out of bounds (n_phases={n_phases}), setting to 0")
                        pending_phase = 0
                    if current_time - set_time >= phase_duration - 0.1:
                        # PATCH: Use APC PKL for all phase switches
                        apc.set_phase_from_pkl(pending_phase)
                        self.last_phase_change[tl_id] = current_time
                        del self.pending_next_phase[tl_id]
                        logic = self._get_traffic_light_logic(tl_id)
                        n_phases = len(logic.phases) if logic else 0
                        current_phase = traci.trafficlight.getPhase(tl_id)
                        if n_phases == 0 or current_phase >= n_phases or current_phase < 0:
                            apc.set_phase_from_pkl(max(0, n_phases - 1))
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
                    if starved_phase is None:
                        starved_phase = self._add_new_green_phase_for_lane(
                            tl_id, most_starved_lane, min_green=self.adaptive_params['min_green'], yellow=3)
                        logic = self._get_traffic_light_logic(tl_id)
                    if starved_phase is not None and current_phase != starved_phase:
                        switched = self._switch_phase_with_yellow_if_needed(
                            tl_id, current_phase, starved_phase, logic, controlled_lanes, lane_data, current_time)
                        logic = self._get_traffic_light_logic(tl_id)
                        n_phases = len(logic.phases) if logic else 1
                        current_phase = traci.trafficlight.getPhase(tl_id)
                        if current_phase >= n_phases:
                            apc.set_phase_from_pkl(n_phases - 1)
                        # PATCH: Always set via APC PKL, not direct setPhase/setPhaseDuration
                        if not switched:
                            apc.set_phase_from_pkl(starved_phase)
                            self.last_phase_change[tl_id] = current_time
                    self.last_green_time[self.lane_id_to_idx[most_starved_lane]] = current_time
                    self.debug_green_lanes(tl_id, lane_data)
                    continue

                # Normal RL control
                self.tl_action_sizes[tl_id] = len(logic.phases)
                self.rl_agent.action_size = max(self.tl_action_sizes.values())
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
                last_change = self.last_phase_change.get(tl_id, -9999)
                if (current_time - last_change >= self.adaptive_params['min_green'] and
                    action != current_phase and
                    not self._is_in_dilemma_zone(tl_id, controlled_lanes, lane_data)):
                    if not self._phase_has_traffic(logic, action, controlled_lanes, lane_data):
                        continue
                    switched = self._switch_phase_with_yellow_if_needed(
                        tl_id, current_phase, action, logic, controlled_lanes, lane_data, current_time)
                    # PATCH: Always use APC PKL for all RL phase changes!
                    if not switched:
                        apc.set_phase_from_pkl(action)
                        self.last_phase_change[tl_id] = current_time
                        self._process_rl_learning(self.intersection_data, lane_data, current_time)
                self.debug_green_lanes(tl_id, lane_data)

        except Exception as e:
            self.logger.error(f"Error in run_step: {e}", exc_info=True)

    def _phase_has_traffic(self, logic, action, controlled_lanes, lane_data):
        """Returns True if at least one green lane in the selected phase has traffic."""
        phase_state = logic.phases[action].state
        for lane_idx, lane in enumerate(controlled_lanes):
            if lane_idx < len(phase_state) and phase_state[lane_idx].upper() == 'G':
                if lane_data.get(lane, {}).get("queue_length", 0) > 0:
                    return True
        return False

    def debug_green_lanes(self, tl_id, lane_data):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
        current_phase = traci.trafficlight.getPhase(tl_id)
        if not (0 <= current_phase < len(logic.phases)):
            print(f"[ERROR] Current phase {current_phase} is out of range for {tl_id} (phases: {len(logic.phases)})")
            return
        phase_state = logic.phases[current_phase].state
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        for lane_idx, lane in enumerate(controlled_lanes):
            if lane_idx < len(phase_state) and phase_state[lane_idx].upper() == 'G':
                qlen = lane_data.get(lane, {}).get("queue_length", None)
                # print debug info as needed                
    def _get_yellow_phase(self, logic, from_idx, to_idx):
        """
        Find a yellow (amber) phase between two phases.
        Returns the yellow phase index if found, else None.
        """
        # Defensive: check bounds!
        n_phases = len(logic.phases) if logic else 0
        if not logic or from_idx == to_idx or from_idx >= n_phases or to_idx >= n_phases or from_idx < 0 or to_idx < 0:
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
            lane_last_green = self.last_green_time.get(lane_id, 0)
            if now - lane_last_green < self.min_green:
                # This lane was just served, skip special event for now
                return None, None
            # Else, handle congestion and update last green time
            self.last_green_time[lane_id] = now
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
            traceback.print_exc()
    def _execute_control_action(self, tl_id, target_lane, action, lane_data, current_time):
        """Execute the selected control action (shortened version)"""
        try:
            if not isinstance(lane_data, dict) or target_lane not in lane_data:
                print("⚠️ Invalid lane_data in _execute_control_action")
                return

            current_phase = traci.trafficlight.getPhase(tl_id)
            target_phase = self._find_phase_for_lane(tl_id, target_lane) or current_phase

            apc = self.adaptive_phase_controllers[tl_id]
            if current_phase != target_phase:
                # Apply adaptive phase controller's PKL-driven phase record
                phase_record = apc.load_phase_from_supabase(target_phase)
                if phase_record:
                    traci.trafficlight.setPhase(tl_id, phase_record["phase_idx"])
                    traci.trafficlight.setPhaseDuration(tl_id, phase_record["duration"])
                else:
                    # Fallback to default phase timing
                    traci.trafficlight.setPhase(tl_id, target_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, apc.max_green)
                # Update timing records
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
                print(f"\n[RL STATE] TL: {tl_id}, Phase: {current_phase}")
                print(f"  - Queues: {queues}")
                print(f"  - Waits: {waits}")
                print(f"  - Current phase state: {logic.phases[current_phase].state if logic else 'N/A'}")
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
                print(f"\n[RL REWARD COMPONENTS] TL: {tl_id}")
                print(f"  - Queue penalty: {reward_components['queue_penalty']:.2f}")
                print(f"  - Wait penalty: {reward_components['wait_penalty']:.2f}")
                print(f"  - Empty green penalty: {reward_components['empty_green_penalty']:.2f}")
                print(f"  - Congestion bonus: {reward_components['congestion_bonus']:.2f}")
                print(f"  - TOTAL REWARD: {reward:.2f}")
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

def main():
    global controller
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

    start_universal_simulation(args.sumo, args.gui, args.max_steps, args.episodes, args.num_retries, args.retry_delay, mode=args.mode)

def start_universal_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1, num_retries=1, retry_delay=1, mode="train"):
    global controller
    controller = None

    def simulation_loop():
        global controller
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
                    controller.run_step()
                    traci.simulationStep()
                    step += 1
                    if step % 1000 == 0:
                        print(f"Episode {episode + 1}: Step {step} completed, elapsed: {time.time()-tstart:.2f}s")
                print(f"Episode {episode + 1} completed after {step} steps")
                controller.end_episode()
                traci.close()
                if episode < episodes - 1: time.sleep(2)
            print(f"\n🎉 All {episodes} episodes completed!")
        except Exception as e:
            print(f"Error in universal simulation: {e}")
        finally:
            try: traci.close()
            except: pass
            print("Simulation resources cleaned up")

    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.start()

    while controller is None or not hasattr(controller, "adaptive_phase_controllers"):
        time.sleep(0.1)

    while True:
        try:
            tls_list = traci.trafficlight.getIDList()
            if tls_list:
                tls_id = tls_list[0]
                break
        except Exception:
            pass
        time.sleep(0.1)

    display = TrafficLightPhaseDisplay(controller.phase_events, poll_interval=500)
    display.start()

    sim_thread.join()
    return controller

if __name__ == "__main__":
    main()