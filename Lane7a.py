# python Lane7a.py --sumo dataset1.sumocfg --gui --max-steps 1000 --episodes 1

import os, sys, time, json, pickle,traceback, logging, threading,argparse, datetime, warnings
from collections import defaultdict,deque
import numpy as np
import traci
from traci._trafficlight import Logic, Phase

from supabase import create_client
from traffic_light_display import SmartIntersectionTrafficDisplay

# ---- Config and Setup ----

SUPABASE_URL = "https://ffrxpjvnovsoqupzochp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZmcnhwanZub3Zzb3F1cHpvY2hwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDU0MDY3NiwiZXhwIjoyMDcwMTE2Njc2fQ.d28oR_bQAQLgZCz3CLeWfTRwikYJaGYzDrSn23obj-s"

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")
for noisy in ("httpx", "httpcore", "postgrest", "storage3"):
    lg = logging.getLogger(noisy)
    lg.setLevel(logging.WARNING)   # or logging.ERROR to hide even more
    lg.propagate = False  
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

os.environ.setdefault('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

# ---- Utility Functions ----

def verify_supabase_connection():
    try:
        supabase.table("apc_states").select("id").limit(1).execute()
        print("✅ Supabase connection successful")
        return True
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}\nCheck SUPABASE_URL and SUPABASE_KEY")
        return False

def fix_phase_states_and_missing_greens(phases, controlled_lanes, min_green=10):
    n_lanes = len(controlled_lanes)
    for phase in phases:
        if len(phase.state) > n_lanes:
            phase.state = phase.state[:n_lanes]
        elif len(phase.state) < n_lanes:
            phase.state = phase.state.ljust(n_lanes, 'r')
    # Ensure each lane is served by at least one green phase
    for i, lane in enumerate(controlled_lanes):
        if not any(phase.state[i].upper() == 'G' for phase in phases):
            state = ''.join('G' if j == i else 'r' for j in range(n_lanes))
            phases.append(Phase(min_green, state))
    return phases

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

def retry_supabase_operation(operation, max_retries=3):
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = 0.5 * (2 ** attempt)
            print(f"[Supabase] Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
            time.sleep(wait_time)


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
        self.r_base = r_base
        self.r_adjust = r_adjust
        self.severe_congestion_threshold = severe_congestion_threshold
        self.large_delta_t = large_delta_t
        self.phase_repeat_counter = defaultdict(int)
        self.last_served_time = defaultdict(lambda: 0)
        self.severe_congestion_global_cooldown_time = 5
        self._links_map = {lid: traci.lane.getLinks(lid) for lid in lane_ids}
        self._controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        self.supabase_available = verify_supabase_connection()
        self._phase_defs = [phase for phase in traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].getPhases()]
        self.weights = np.array([0.4, 0.2, 0.2, 0.2])
        self.weight_history = []
        self.metric_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=50)
        self.R_target = r_base
        self.phase_count = 0
        self.rl_agent = None
        self.last_phase_switch_sim_time = 0
        self.pending_request_timestamp = 0
        self.emergency_cooldown = {}
        self.emergency_global_cooldown = 0
        self.last_extended_time = 0
        self.protected_left_cooldown = defaultdict(float)
        self.severe_congestion_cooldown = {}
        self.severe_congestion_global_cooldown = 0
        self.last_phase_idx = None
        self.last_emergency_event = {}
        self.pending_phase_request = None
        self.pending_extension_request = None
        self.pending_priority_type = None
        self._last_ext_telemetry = -1.0
        self.left_block_steps = defaultdict(int)      # lane_id -> consecutive blocked steps while green
        self.left_block_min_steps = 3 
        self.activation = {
            "phase_idx": None,
            "start_time": 0.0,
            "base_duration": None,       # The base (nominal) duration used at activation
            "desired_total": None        # The desired total duration for this activation
        }
        self._load_apc_state_supabase()
        self.preload_phases_from_sumo()
        self._initialize_base_durations()
    # =============== ACTIVATION/EXTENSION HELPERS (PATCH) ===============
    def _reset_activation(self, phase_idx, base_duration, desired_total):
        now = traci.simulation.getTime()
        self.activation["phase_idx"] = phase_idx
        self.activation["start_time"] = now
        self.activation["base_duration"] = float(base_duration)
        self.activation["desired_total"] = float(desired_total)
        self.last_phase_switch_sim_time = now  # keep existing behavior

    def _get_phase_elapsed(self):
        try:
            return max(0.0, traci.simulation.getTime() - float(self.activation.get("start_time", 0.0)))
        except Exception:
            return 0.0

    def _get_phase_remaining(self):
        try:
            now = traci.simulation.getTime()
            next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
            return max(0.0, next_switch - now)
        except Exception:
            return 0.0

    def _maybe_update_phase_remaining(self, desired_total, buffer=0.5):

        if self.activation["phase_idx"] is None:
            return False

        elapsed = self._get_phase_elapsed()
        remaining = self._get_phase_remaining()
        desired_remaining = max(0.0, float(desired_total) - elapsed)

        # Only adjust if we're meaningfully different from what's already scheduled
        if abs(remaining - desired_remaining) > float(buffer):
            try:
                traci.trafficlight.setPhaseDuration(self.tls_id, desired_remaining)
                # Update book-keeping to reflect new target
                self.activation["desired_total"] = float(desired_total)
                # Update records for auditability
                current_phase = traci.trafficlight.getPhase(self.tls_id)
                total_after_update = elapsed + desired_remaining
                phase_record = self.load_phase_from_supabase(current_phase)
                base = phase_record.get("base_duration", self.min_green) if phase_record else self.min_green
                extended_time = max(0.0, total_after_update - base)
                self.update_phase_duration_record(current_phase, total_after_update, extended_time)
                if hasattr(self, "log_phase_to_event_log"):
                    self.log_phase_to_event_log(current_phase, total_after_update)

                print(f"[PATCH][EXT] Phase {current_phase}: elapsed={elapsed:.1f}s, remaining {remaining:.1f}s → {desired_remaining:.1f}s (desired_total={desired_total:.1f}s)")
                return True
            except Exception as e:
                print(f"[PATCH][EXT][ERROR] Failed to set remaining time: {e}")
                return False
        return False
    def get_current_extension_seconds(self):
        """
        Returns cumulative extension for the current activation:
        max(0, (elapsed + remaining) - base_duration_of_activation)
        """
        try:
            if self.activation["phase_idx"] is None:
                return 0.0
            elapsed = self._get_phase_elapsed()
            remaining = self._get_phase_remaining()
            base = self.activation.get("base_duration") or self.min_green
            return max(0.0, (elapsed + remaining) - base)
        except Exception:
            return 0.0

    def emit_extension_telemetry(self, threshold=0.5):
        """
        Pushes a telemetry event with base, total, and extended_time so the UI can
        render the correct 'Extended Time' even when no patch event occurs.
        """
        try:
            if self.activation["phase_idx"] is None:
                return 0.0
            elapsed = self._get_phase_elapsed()
            remaining = self._get_phase_remaining()
            base = self.activation.get("base_duration") or self.min_green
            total = elapsed + remaining
            extended = max(0.0, total - base)
            # throttle event spam
            if self._last_ext_telemetry < 0 or abs(extended - self._last_ext_telemetry) >= threshold:
                self._last_ext_telemetry = extended
                if hasattr(self, "controller") and hasattr(self.controller, "phase_events"):
                    self.controller.phase_events.append({
                        "tls_id": self.tls_id,
                        "phase_idx": self.activation.get("phase_idx"),
                        "base_duration": base,
                        "duration": total,
                        "extended_time": extended,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "telemetry": True  # marker so the UI can distinguish from patch events
                    })
            return extended
        except Exception:
            return 0.0
    def apply_extension_delta(self, delta_t, buffer=0.5):

        if self.activation["phase_idx"] is None:
            return None
        base = self.activation["base_duration"] if self.activation["base_duration"] is not None else self.min_green
        desired_total = float(np.clip(base + float(delta_t), self.min_green, self.max_green))
        self._maybe_update_phase_remaining(desired_total, buffer=buffer)
        return desired_total

    # ----- Phase Request/Transition/Starvation/Preemption -----

    def request_phase_change(self, phase_idx, priority_type='normal', extension_duration=None):
        current_time = traci.simulation.getTime()
        priority_order = {
            'critical_starvation': 10,
            'heavy_congestion': 9,   # raise above 'emergency'
            'emergency': 8,
            'starvation': 5,
            'normal': 1
        }        
        current_priority = priority_order.get(self.pending_priority_type, 0)
        new_priority = priority_order.get(priority_type, 1)
        if new_priority >= current_priority:
            self.pending_phase_request = phase_idx
            self.pending_extension_request = extension_duration
            self.pending_priority_type = priority_type
            self.pending_request_timestamp = current_time
            self._log_apc_event({
                "action": "pending_phase_request",
                "requested_phase": phase_idx,
                "priority_type": priority_type,
                "extension_duration": extension_duration,
                "current_phase": traci.trafficlight.getPhase(self.tls_id),
                "timestamp": current_time
            })
            if priority_type in ['critical_starvation', 'emergency'] and self.is_phase_ending():
                return self.process_pending_requests_on_phase_end()
            return True
        else:
            self.logger.info(f"[PENDING REQUEST IGNORED] {self.tls_id}: Lower priority request ignored")
            return False

    def is_phase_ending(self):
        try:
            current_time = traci.simulation.getTime()
            next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
            time_remaining = next_switch - current_time
            phase_duration = traci.trafficlight.getPhaseDuration(self.tls_id)
            return (time_remaining <= 2.0 or time_remaining <= phase_duration * 0.1)
        except Exception:
            return False

    def clear_pending_requests(self):
        if self.pending_phase_request is not None:
            print(f"[PENDING REQUEST CLEARED] {self.tls_id}: Cleared pending phase {self.pending_phase_request}")
        self.pending_phase_request = None
        self.pending_extension_request = None
        self.pending_priority_type = None
        self.pending_request_timestamp = 0

    def process_pending_requests_on_phase_end(self):
        if self.pending_phase_request is None:
            return False
        current_time = traci.simulation.getTime()
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        elapsed = current_time - self.last_phase_switch_sim_time
        if elapsed < self.min_green and self.pending_priority_type != 'emergency':
            print(f"[PENDING REQUEST BLOCKED] {self.tls_id}: Min green not met ({elapsed:.1f}s < {self.min_green}s)")
            return False
        requested_phase = self.pending_phase_request
        extension_duration = self.pending_extension_request
        priority_type = self.pending_priority_type
        print(f"[EXECUTING PENDING REQUEST] {self.tls_id}: Switching to phase {requested_phase} (priority: {priority_type})")
        success = self.set_phase_from_API(requested_phase, requested_duration=extension_duration)
        if success:
            self._log_apc_event({
                "action": "executed_pending_request",
                "old_phase": current_phase,
                "new_phase": requested_phase,
                "priority_type": priority_type,
                "extension_duration": extension_duration,
                "request_age": current_time - self.pending_request_timestamp
            })
            self.clear_pending_requests()
            return True
        else:
            print(f"[PENDING REQUEST FAILED] {self.tls_id}: Failed to execute phase change")
            return False

    # ----- Supabase State Management -----

    def flush_pending_supabase_writes(self, max_retries=6, max_batch=1):
        """
        Attempts to write pending state(s) in self._pending_db_ops to Supabase,
        with exponential backoff and jitter. Keeps failed writes in the queue.
        Args:
            max_retries (int): Number of retry attempts before giving up.
            max_batch (int): Number of states to send in one flush (default: 1, only latest).
        """
        import random

        with self._db_lock:
            if not self._pending_db_ops:
                return

            # Only try the latest N states, discard older if needed
            batch = self._pending_db_ops[-max_batch:] if max_batch > 0 else [self._pending_db_ops[-1]]

            for state in batch:
                state_json = json.dumps(state)
                attempt = 0
                delay = 1.0
                while attempt < max_retries:
                    try:
                        response = supabase.table("apc_states").upsert({
                            "tls_id": self.tls_id,
                            "state_type": "full",
                            "data": state_json,
                            "updated_at": datetime.datetime.now().isoformat()
                        }).execute()
                        if response.data:
                            print(f"[Supabase] Successfully synced state for {self.tls_id}")
                            # Remove this state from the queue
                            if state in self._pending_db_ops:
                                self._pending_db_ops.remove(state)
                            break
                    except Exception as e:
                        print(f"[Supabase] Write attempt {attempt+1}/{max_retries} failed: {e}\nData: {state_json[:300]}...")
                        if attempt == max_retries - 1:
                            print(f"[Supabase] All {max_retries} attempts failed, keeping data in queue for {self.tls_id}")
                        else:
                            # Exponential backoff with jitter
                            sleep_time = delay + random.uniform(0, delay * 0.5)
                            time.sleep(sleep_time)
                            delay = min(delay * 2, 30)
                        attempt += 1
    def _load_apc_state_supabase(self):
        try:
            response = supabase.table("apc_states").select("data, updated_at").eq("tls_id", self.tls_id).eq("state_type", "full").order("updated_at", desc=True).limit(1).execute()
            if response.data and len(response.data) > 0:
                self.apc_state = json.loads(response.data[0]["data"])
                print(f"[Supabase] Loaded state for {self.tls_id} from {response.data[0]['updated_at']}")
            else:
                self.apc_state = {"events": [], "phases": []}
                print(f"[Supabase] No existing state for {self.tls_id}, initializing fresh")
        except Exception as e:
            print(f"[Supabase] Failed to load state for {self.tls_id}: {e}")
            self.apc_state = {"events": [], "phases": []}

    def _save_apc_state_supabase(self):
        if self.supabase_available:
            self._pending_db_ops.append(self.apc_state.copy())
        else:
            print(f"[Supabase] Offline mode - state not saved for {self.tls_id}")

    # ----- Phase/Reward/Traffic Logic -----

    def get_full_phase_sequence(self):
        phase_records = sorted(self.apc_state.get("phases", []), key=lambda x: x["phase_idx"])
        if not phase_records:
            return [(p.state, p.duration) for p in self._phase_defs]
        return [(rec["state"], rec["duration"]) for rec in phase_records]

    def load_phase_from_supabase(self, phase_idx=None):
        # 1) Try cached state first
        for p in self.apc_state.get("phases", []):
            if p.get("phase_idx") == phase_idx:
                return p
        # 2) Fallback to SUMO logic if not cached; also cache it into apc_state
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            phases = logic.getPhases()
            if phase_idx is not None and 0 <= phase_idx < len(phases):
                ph = phases[phase_idx]
                record = {
                    "phase_idx": phase_idx,
                    "duration": float(getattr(ph, "duration", self.min_green)),
                    "base_duration": float(getattr(ph, "duration", self.min_green)),
                    "state": ph.state,
                    "extended_time": 0.0,
                }
                self.apc_state.setdefault("phases", []).append(record.copy())
                self._save_apc_state_supabase()
                return record
        except Exception as e:
            print(f"[WARN] load_phase_from_supabase fallback failed for phase {phase_idx}: {e}")
        return None

    def update_lane_serving_status(self):
        current_phase_idx = traci.trafficlight.getPhase(self.tls_id)
        current_time = traci.simulation.getTime()
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        if current_phase_idx >= len(logic.getPhases()):
            return
        phase_state = logic.getPhases()[current_phase_idx].state
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for i, lane_id in enumerate(controlled_lanes):
            if i < len(phase_state) and phase_state[i].upper() == 'G':
                self.last_served_time[lane_id] = current_time

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
        self.weights = values / total if total != 0 else np.array([0.25] * 4)
        self.weight_history.append(self.weights.copy())
        print(f"[ADAPTIVE WEIGHTS] {self.tls_id}: {self.weights}")

    def calculate_reward(self, bonus=0, penalty=0):
        metrics = np.zeros(4)
        valid_lanes = 0
        MAX_VALUES = [0.2, 13.89, 300, 50]
        current_max = [
            max(0.1, max(traci.lane.getLastStepVehicleNumber(lid) / max(1, traci.lane.getLength(lid)) for lid in self.lane_ids)),
            max(5.0, max(traci.lane.getLastStepMeanSpeed(lid) for lid in self.lane_ids)),
            max(30.0, max(traci.lane.getWaitingTime(lid) for lid in self.lane_ids)),
            max(5.0, max(traci.lane.getLastStepHaltingNumber(lid) for lid in self.lane_ids))
        ]
        max_vals = [min(MAX_VALUES[i], current_max[i]) for i in range(4)]
        for lane_id in self.lane_ids:
            q, w, v, dens = self.get_lane_stats(lane_id)
            if any(val < 0 for val in (q, w, v, dens)):
                continue
            metrics += [
                min(dens, max_vals[0]) / max_vals[0],
                min(v, max_vals[1]) / max_vals[1],
                min(w, max_vals[2]) / max_vals[2],
                min(q, max_vals[3]) / max_vals[3]
            ]
            valid_lanes += 1
        if valid_lanes == 0:
            self.last_R = 0
            return 0
        avg_metrics = metrics / valid_lanes
        self.metric_history.append(avg_metrics)
        self.adjust_weights()
        R = 100 * (
            -self.weights[0] * avg_metrics[0] +
            self.weights[1] * avg_metrics[1] -
            self.weights[2] * avg_metrics[2] -
            self.weights[3] * avg_metrics[3] +
            bonus - penalty
        )
        self.last_R = np.clip(R, -100, 100)
        print(f"[REWARD] {self.tls_id}: R={self.last_R:.2f} (dens={avg_metrics[0]:.2f}, spd={avg_metrics[1]:.2f}, wait={avg_metrics[2]:.2f}, queue={avg_metrics[3]:.2f}) Weights: {self.weights}, Bonus: {bonus}, Penalty: {penalty}")
        return self.last_R

    def update_R_target(self, window=10):
        if len(self.reward_history) < window or self.phase_count % 10 != 0:
            return
        avg_R = np.mean(list(self.reward_history)[-window:])
        self.R_target = self.r_base + self.r_adjust * (avg_R - self.r_base)
        print(f"\n[TARGET UPDATE] R_target={self.R_target:.2f} (avg={avg_R:.2f})")
    def is_phase_ending(self):
        try:
            current_time = traci.simulation.getTime()
            next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
            time_remaining = next_switch - current_time
            phase_duration = traci.trafficlight.getPhaseDuration(self.tls_id)
            # Phase is ending if less than 2 seconds remain OR less than 10% of total phase duration remains
            return (time_remaining <= 2.0 or time_remaining <= phase_duration * 0.1)
        except:
            return False


# PATCH 4: Enhanced _load_apc_state_supabase method

    # PATCH 5: Add phase record storage to separate table (optional)
    def save_phase_record_to_supabase(self, phase_idx, duration, state_str, delta_t, raw_delta_t, penalty,
                                    reward=None, bonus=None, weights=None, event_type=None, lanes=None):
        try:
            # Use stored base_duration for this phase if available
            rec = self.load_phase_from_supabase(phase_idx)
            base_dur = rec.get("base_duration", self.min_green) if rec else self.min_green
            phase_record = {
                "tls_id": self.tls_id,
                "phase_idx": phase_idx,
                "duration": float(duration),
                "base_duration": float(base_dur),
                "state_str": state_str,
                "delta_t": float(delta_t),
                "raw_delta_t": float(raw_delta_t),
                "penalty": float(penalty),
                "reward": reward,
                "bonus": bonus if bonus is not None else 0,
                "extended_time": max(0.0, float(duration) - float(base_dur)),
                "event_type": event_type,
                "weights": weights if weights is not None else self.weights.tolist(),
                "lanes": lanes if lanes is not None else self.lane_ids[:],
                "sim_time": traci.simulation.getTime(),
                "updated_at": datetime.datetime.now().isoformat()
            }
            supabase.table("phase_records").insert(phase_record).execute()
        except Exception as e:
            print(f"[Supabase] Failed to save phase record: {e}")
    # PATCH 6: Add event logging to separate table
    def log_event_to_supabase(self, event):
        try:
            event_record = {
                "tls_id": self.tls_id,
                "event_type": event.get("action", "unknown"),
                "event_data": json.dumps(event),
                "sim_time": traci.simulation.getTime()
            }
            
            supabase.table("simulation_events").insert(event_record).execute()
            
        except Exception as e:
            print(f"[Supabase] Failed to log event: {e}")

    # Enhanced lane tracking system to prevent starvation


    # PATCH: Add method to process pending requests when phase ends
    def _initialize_base_durations(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        for idx, phase in enumerate(logic.getPhases()):
            found = False
            for p in self.apc_state.setdefault("phases", []):
                if p["phase_idx"] == idx:
                    found = True
                    break
            if not found:
                # Cache into apc_state so later lookups see a proper base_duration
                self.apc_state["phases"].append({
                    "phase_idx": idx,
                    "duration": float(phase.duration),
                    "base_duration": float(phase.duration),
                    "state": phase.state,
                    "extended_time": 0.0
                })
                self._save_apc_state_supabase()
                self.save_phase_record_to_supabase(
                    phase_idx=idx,
                    duration=phase.duration,
                    state_str=phase.state,
                    delta_t=0,
                    raw_delta_t=0,
                    penalty=0
                )
    def log_phase_adjustment(self, action_type, phase, old_duration, new_duration):
        print(f"[LOG] {action_type} phase {phase}: {old_duration} -> {new_duration}")    
 
    def find_phase_to_overwrite(self, new_state, exclude_indices=None):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        phases = logic.phases
        
        if exclude_indices is None:
            exclude_indices = []
        
        # Don't overwrite the current phase
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        exclude_indices.append(current_phase)
        
        # Calculate phase usage statistics if we haven't already
        if not hasattr(self, "phase_usage_count"):
            self.phase_usage_count = defaultdict(int)
            self.phase_last_used = defaultdict(lambda: 0)
        
        # Score each phase based on multiple factors
        phase_scores = {}
        current_time = traci.simulation.getTime()
        
        for idx, phase in enumerate(phases):
            # Skip excluded phases
            if idx in exclude_indices:
                continue
                
            # Skip yellow phases
            if 'y' in phase.state:
                continue
                
            # Calculate similarity score (how similar is this phase to the new one?)
            similarity = sum(1 for a, b in zip(phase.state, new_state) if a == b) / len(new_state)
            
            # Calculate usage score (less used phases get higher scores)
            usage_score = 1.0 / (self.phase_usage_count.get(idx, 1) + 1)
            
            # Calculate recency score (older phases get higher scores)
            time_since_used = current_time - self.phase_last_used.get(idx, 0)
            recency_score = min(1.0, time_since_used / 1000)  # Normalize to [0,1]
            
            # Combined score (higher = better to overwrite)
            # We prefer to overwrite phases that are:
            # 1. Similar to the new phase (easier transition)
            # 2. Used infrequently
            # 3. Haven't been used recently
            score = (
                0.4 * similarity +   # Weight for phase similarity
                0.4 * usage_score +  # Weight for infrequent usage 
                0.2 * recency_score  # Weight for recent usage
            )
            
            phase_scores[idx] = score
            
        # If we have no valid phases to overwrite, return None
        if not phase_scores:
            return None
            
        # Find the phase with the highest score
        best_phase_idx = max(phase_scores, key=phase_scores.get)
        
        print(f"[PHASE OVERWRITE] Selecting phase {best_phase_idx} to overwrite (score: {phase_scores[best_phase_idx]:.2f})")
        return best_phase_idx

    def overwrite_phase(self, phase_idx, new_state, new_duration):
        try:
            # Get current logic
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            phases = list(logic.phases)
            
            if phase_idx >= len(phases):
                print(f"[ERROR] Cannot overwrite phase {phase_idx}: index out of range")
                return False
                
            # Create the new phase
            new_phase = traci.trafficlight.Phase(new_duration, new_state)
            
            # Replace the old phase with the new one
            phases[phase_idx] = new_phase
            
            # Create new logic with the updated phases
            new_logic = traci.trafficlight.Logic(
                logic.programID,
                logic.type,
                logic.currentPhaseIndex,
                phases
            )
            
            # Apply the new logic
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
            
            # Update phase usage statistics
            self.phase_usage_count[phase_idx] = 0
            self.phase_last_used[phase_idx] = traci.simulation.getTime()
            
            # Update phase records
            self.save_phase_record_to_supabase(
                phase_idx=phase_idx,
                duration=new_duration,
                state_str=new_state,
                delta_t=0,
                raw_delta_t=0,
                penalty=0,
                event_type="phase_overwrite"
            )
            
            # Log the overwrite event
            self._log_apc_event({
                "action": "phase_overwrite",
                "phase_idx": phase_idx,
                "old_state": logic.phases[phase_idx].state if phase_idx < len(logic.phases) else "unknown",
                "new_state": new_state,
                "new_duration": new_duration,
                "sim_time": traci.simulation.getTime()
            })
            
            print(f"[PHASE OVERWRITE] Successfully overwrote phase {phase_idx} with new state: {new_state}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Phase overwrite failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_or_extend_phase(self, green_lanes, delta_t):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        valid_green_lanes = [lane for lane in green_lanes if lane in controlled_lanes]
        
        if not valid_green_lanes:
            print(f"[WARNING] No valid green lanes provided for phase creation")
            return None

        # Create or find phase state
        new_state = self.create_phase_state(green_lanes=valid_green_lanes)
        
        # Check if this phase already exists
        phase_idx = None
        base_duration = self.min_green
        
        # DEBUG: Print what we're looking for
        print(f"[DEBUG] Looking for phase with state: {new_state}")
        print(f"[DEBUG] Available phase states: {[p.state for p in logic.getPhases()]}")
        
        for idx, phase in enumerate(logic.getPhases()):
            if phase.state == new_state:
                # Found existing phase
                print(f"[DEBUG] Found existing phase {idx} with state {phase.state}")
                phase_record = self.load_phase_from_supabase(idx)
                if phase_record and "duration" in phase_record:
                    base_duration = phase_record["duration"]
                else:
                    base_duration = phase.duration
                phase_idx = idx
                break
        
        # Calculate new duration with delta_t
        duration = np.clip(base_duration + delta_t, self.min_green, self.max_green)
        
        if phase_idx is not None:
            # Phase exists, extend it
            print(f"[PHASE EXTEND] Extending phase {phase_idx} from {base_duration}s to {duration}s (delta_t={delta_t}s)")
            # Save the updated phase to Supabase
            self.save_phase_record_to_supabase(phase_idx, duration, new_state, delta_t, delta_t, penalty=0)
            
            # IMPORTANT: Actually change the phase in SUMO immediately
            traci.trafficlight.setPhase(self.tls_id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.tls_id, duration)
            
            # Update display if applicable
            if hasattr(self, "update_display"):
                self.update_display(phase_idx, duration)
            return phase_idx
        else:
            # Create new phase
            print(f"[PHASE CREATE] Creating new phase with state: {new_state}, duration: {duration}s")
            
            # Create green phase and add to logic
            new_phase = traci.trafficlight.Phase(duration, new_state)
            phases = list(logic.getPhases())
            new_phase_idx = len(phases)
            phases.append(new_phase)
            
            # Set new traffic light logic
            new_logic = traci.trafficlight.Logic(
                logic.programID, 
                logic.type, 
                new_phase_idx, # Set current phase to the new phase
                [traci.trafficlight.Phase(duration=ph.duration, state=ph.state) for ph in phases]
            )
            
            # Apply to SUMO
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
            traci.trafficlight.setPhase(self.tls_id, new_phase_idx)
            
            # Save to Supabase
            self.save_phase_record_to_supabase(new_phase_idx, duration, new_state, delta_t, delta_t, penalty=0)
            
            # Update display if applicable
            if hasattr(self, "update_display"):
                self.update_display(new_phase_idx, duration)
            
            print(f"[PHASE CREATE] New phase created at index {new_phase_idx}")
            return new_phase_idx

    def calculate_delta_t_and_penalty(self, R):
        # Raw delta-t is proportional to the reward difference
        raw_delta_t = self.alpha * (R - self.R_target)

        # Apply penalty for large adjustments
        penalty = max(0, abs(raw_delta_t) - self.large_delta_t)

        # Scale delta-t using tanh for smoothing and clip within desired range
        ext_t = 20 * np.tanh(raw_delta_t / 20)  # Increased scaling factor for smoother adjustments
        delta_t = np.clip(ext_t, -20, 20)  # Allow both positive and negative adjustments

        print(f"[DEBUG] [DELTA_T_PENALTY] R={R:.2f}, R_target={self.R_target:.2f}, raw={raw_delta_t:.2f}, Δt={delta_t:.2f}, penalty={penalty:.2f}")
        return raw_delta_t, delta_t, penalty
    


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
        self._db_writer.stop()
        self.flush_pending_supabase_writes()
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
        state = ['r'] * len(controlled_lanes)  # Default all to red
        
        def set_lanes(lanes, state_char):
            if not lanes:
                return
            for lane in lanes:
                if lane in controlled_lanes:
                    idx = controlled_lanes.index(lane)
                    if idx < len(state):  # Bounds check
                        state[idx] = state_char
        
        # Set lanes in priority order: red first, then yellow, then green
        set_lanes(red_lanes, 'r')
        set_lanes(yellow_lanes, 'y') 
        set_lanes(green_lanes, 'G')
        
        return "".join(state)
    def add_new_phase(self, green_lanes, green_duration=20, yellow_duration=3, yellow_lanes=None):
        print(f"[DEBUG] add_new_phase called with green_lanes={green_lanes}, green_duration={green_duration}, yellow_duration={yellow_duration}, yellow_lanes={yellow_lanes}")
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            phases = list(logic.getPhases())
            max_phases = 12  # strict SUMO limit

            # Check for existing phase with same green
            green_state = self.create_phase_state(green_lanes=green_lanes)
            for idx, phase in enumerate(phases):
                if phase.state == green_state:
                    return idx

            # If at phase limit, must overwrite instead of append
            if len(phases) >= max_phases:
                print(f"[PHASE LIMIT] {self.tls_id} at phase limit ({max_phases}), will overwrite")
                idx_to_overwrite = self.find_phase_to_overwrite(green_state)
                if idx_to_overwrite is not None:
                    self.overwrite_phase(idx_to_overwrite, green_state, green_duration)
                    return idx_to_overwrite
                # fallback: use phase 0
                print(f"[PHASE LIMIT] Could not find phase to overwrite, fallback to phase 0")
                return 0

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
    def find_best_phase_for_traffic(self):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        
        # Score each phase based on vehicles it would serve
        best_phase = None
        best_score = -1
        
        for phase_idx, phase in enumerate(logic.phases):
            # Skip yellow phases (any state with 'y')
            if 'y' in phase.state:
                continue
                
            phase_score = 0
            green_lanes = []
            
            # Calculate which lanes get green in this phase
            for lane_idx, lane_id in enumerate(controlled_lanes):
                if lane_idx < len(phase.state) and phase.state[lane_idx].upper() == 'G':
                    green_lanes.append(lane_id)
                    
            # Add up queue lengths and waiting times for green lanes
            for lane_id in green_lanes:
                try:
                    queue = traci.lane.getLastStepHaltingNumber(lane_id)
                    wait = traci.lane.getWaitingTime(lane_id)
                    phase_score += (queue * 0.8) + (wait * 0.1)
                except:
                    pass
                    
            # If better than previous best, update
            if phase_score > best_score:
                best_score = phase_score
                best_phase = phase_idx
                
        print(f"[BEST PHASE] Selected phase {best_phase} with score {best_score}")
        return best_phase
    # PATCH: Add helper method to find phase for lane
    def find_or_create_phase_for_lane(self, lane_id):
         logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
         controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
         
         if lane_id not in controlled_lanes:
             print(f"[DEBUG] find_or_create_phase_for_lane: {lane_id} not in controlled_lanes")
             return None
             
         # Look for existing phase
         lane_idx = controlled_lanes.index(lane_id)
         for idx, phase in enumerate(logic.getPhases()):
             if lane_idx < len(phase.state) and phase.state[lane_idx] in 'Gg':
                 print(f"[DEBUG] Existing phase {idx} serves lane {lane_id}")
                 return idx
         
         # Create new phase if needed
         print(f"[DEBUG] No phase found for {lane_id}, creating new phase")
         return self.add_new_phase_for_lane(lane_id)
    # PATCH: Override reorganize_or_create_phase to use pending requests
    
    def find_phase_for_lane(self, lane_id):
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            if lane_id not in controlled_lanes:
                return None
            lane_idx = controlled_lanes.index(lane_id)
            for idx, phase in enumerate(logic.getPhases()):
                state = phase.state
                if lane_idx < len(state) and state[lane_idx].upper() == 'G':
                    return idx
            return None
        except Exception as e:
            print(f"[ERROR] find_phase_for_lane failed for {lane_id}: {e}")
            return None
    def reorganize_or_create_phase(self, lane_id, event_type):
        try:
            is_left_turn = any(link[6] == 'l' for link in traci.lane.getLinks(lane_id))
            now = traci.simulation.getTime()
            if is_left_turn and now - self.protected_left_cooldown[lane_id] < 60:
                return False
                
            target_phase = self.find_or_create_phase_for_lane(lane_id)
            if target_phase is not None:
                # PATCH: Use pending request instead of immediate switch
                priority = 'emergency' if event_type == 'emergency_vehicle' else 'severe_congestion'
                success = self.request_phase_change(target_phase, priority_type=priority)
                
                if success:
                    self._log_apc_event({
                        "action": "reorganize_phase_requested",
                        "lane_id": lane_id,
                        "event_type": event_type,
                        "requested_phase": target_phase,
                        "priority_type": priority
                    })
                    if is_left_turn:
                        self.protected_left_cooldown[lane_id] = now
                return success
            return False
        except Exception as e:
            print(f"[ERROR] Phase reorganization failed: {e}")
            return False

    # PATCH: Add status method for debugging
    def get_pending_request_status(self):
        return {
            "pending_phase": self.pending_phase_request,
            "pending_extension": self.pending_extension_request,
            "priority_type": self.pending_priority_type,
            "request_age": traci.simulation.getTime() - self.pending_request_timestamp if self.pending_phase_request else 0,
            "phase_ending": self.is_phase_ending()
        }
  
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
    

    def calculate_delta_t(self, R):
        raw_delta_t = self.alpha * (R - self.R_target)
        delta_t = 10 * np.tanh(raw_delta_t / 20)
        print(f"[DELTA_T] R={R:.2f}, R_target={self.R_target:.2f}, Δt={delta_t:.2f}")
        return np.clip(delta_t, -10, 10)


    def add_new_phase_for_lane(self, lane_id, green_duration=None, yellow_duration=3):
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
        protected_lefts = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            for link in traci.lane.getLinks(lane_id):
                if link[6] == 'l':
                    protected_lefts.append(lane_id)
                    break
        return protected_lefts

    def get_conflicting_straight_lanes(self, left_lane):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        conflicting_lanes = []
        
        try:
            # Get the edge this left lane is on
            left_edge = traci.lane.getEdgeID(left_lane)
            
            # For each controlled lane, check if it conflicts
            for lane in controlled_lanes:
                if lane == left_lane:
                    continue
                    
                lane_edge = traci.lane.getEdgeID(lane)
                
                # Simple heuristic: lanes from different edges that aren't left turns
                # are potential conflicts
                if lane_edge != left_edge:
                    links = traci.lane.getLinks(lane)
                    is_left = any(len(link) > 6 and link[6] == 'l' for link in links)
                    
                    if not is_left:  # Straight or right turn lanes can conflict
                        conflicting_lanes.append(lane)
                        
        except Exception as e:
            print(f"[ERROR] Conflict detection failed: {e}")

        return conflicting_lanes
    def create_protected_left_phase_for_lane(self, left_lane):
        try:
            controlled_links = traci.trafficlight.getControlledLinks(self.tls_id)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]

            if not controlled_links:
                print(f"[ERROR] No controlled links found for {self.tls_id}")
                return None

            # Find all link indices for this left lane
            left_link_indices = [i for i, link in enumerate(controlled_links) if link[0][0] == left_lane]
            if not left_link_indices:
                print(f"[ERROR] No controlled links for lane {left_lane}")
                return None

            # Build the phase state: only the left-turn links are green
            protected_state = ''.join('G' if i in left_link_indices else 'r' for i in range(len(controlled_links)))
            
            # Check for existing identical phase first
            for idx, phase in enumerate(logic.phases):
                if phase.state == protected_state:
                    print(f"[PROTECTED LEFT] Existing protected left phase found at idx {idx}")
                    return idx
                    
            # Find a phase to overwrite
            # Exclude the current phase and any yellow phases
            exclude_indices = [
                i for i, phase in enumerate(logic.phases) 
                if 'y' in phase.state or i == traci.trafficlight.getPhase(self.tls_id)
            ]
            
            phase_to_overwrite = self.find_phase_to_overwrite(protected_state, exclude_indices)
            
            if phase_to_overwrite is not None:
                # Overwrite the selected phase
                duration = self.max_green
                success = self.overwrite_phase(phase_to_overwrite, protected_state, duration)
                if success:
                    print(f"[PROTECTED LEFT] Overwrote phase {phase_to_overwrite} with protected left for {left_lane}")
                    return phase_to_overwrite
            
            # If we couldn't find a phase to overwrite or overwriting failed, fallback
            print(f"[WARNING] Could not overwrite any phase, falling back to best existing phase")
            return self._find_best_existing_phase(logic.phases, protected_state)
                
        except Exception as e:
            print(f"[ERROR] Exception creating protected left phase: {e}")
            import traceback
            traceback.print_exc()
            return None
    def rl_create_or_overwrite_phase(self, state_vector, desired_green_lanes=None):
        if not hasattr(self.rl_agent, 'phase_overwrite_threshold'):
            # Initialize phase overwrite threshold (how often we overwrite vs append)
            self.rl_agent.phase_overwrite_threshold = 0.7
        
        # If no specific green lanes provided, use RL agent to determine them
        if desired_green_lanes is None:
            # Get current traffic conditions
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            traffic_scores = []
            
            for lane in controlled_lanes:
                queue, wait, _, _ = self.get_lane_stats(lane)
                # Score based on queue length and waiting time
                score = queue * 0.7 + min(wait / 10, 5)
                traffic_scores.append((lane, score))
            
            # Select top lanes with highest scores
            traffic_scores.sort(key=lambda x: x[1], reverse=True)
            num_lanes = min(3, max(1, len(traffic_scores)))
            desired_green_lanes = [lane for lane, _ in traffic_scores[:num_lanes]]
        
        # Create the new phase state
        new_state = self.create_phase_state(green_lanes=desired_green_lanes)
        
        # Count current phases
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        phase_count = len(logic.phases)
        max_phases = 12  # SUMO phase limit
        
        # Calculate new duration based on traffic
        total_queue = sum(self.get_lane_stats(lane)[0] for lane in desired_green_lanes)
        total_wait = sum(self.get_lane_stats(lane)[1] for lane in desired_green_lanes)
        
        duration = np.clip(
            self.min_green + total_queue * 1.5 + total_wait * 0.1,
            self.min_green,
            self.max_green
        )
        
        # Check if we're near the phase limit or randomly decide to overwrite
        if phase_count >= max_phases - 1 or np.random.random() < self.rl_agent.phase_overwrite_threshold:
            # Find a suitable phase to overwrite
            phase_to_overwrite = self.find_phase_to_overwrite(new_state)
            
            if phase_to_overwrite is not None:
                # Overwrite the phase
                success = self.overwrite_phase(phase_to_overwrite, new_state, duration)
                
                if success:
                    self._log_apc_event({
                        "action": "rl_overwrite_phase",
                        "phase_idx": phase_to_overwrite,
                        "green_lanes": desired_green_lanes,
                        "new_state": new_state,
                        "duration": duration
                    })
                    
                    # Adjust overwrite threshold - increase it slightly if successful
                    self.rl_agent.phase_overwrite_threshold = min(
                        0.9, 
                        self.rl_agent.phase_overwrite_threshold + 0.02
                    )
                    
                    return phase_to_overwrite
        
        # Fall back to creating a new phase if overwriting didn't work or wasn't chosen
        # (This will use your existing methods that append phases)
        try:
            # Use existing create_or_extend_phase but with a check for max phases
            if phase_count < max_phases - 1:
                new_phase_idx = self.create_or_extend_phase(desired_green_lanes, 0)
                
                if new_phase_idx is not None:
                    # Decrease overwrite threshold slightly when we append
                    self.rl_agent.phase_overwrite_threshold = max(
                        0.5, 
                        self.rl_agent.phase_overwrite_threshold - 0.01
                    )
                    
                    return new_phase_idx
            
            # If we've reached the limit, force an overwrite of the least used phase
            print("[PHASE LIMIT] Reached maximum phases, forcing phase overwrite")
            phase_to_overwrite = self.find_phase_to_overwrite(new_state)
            
            if phase_to_overwrite is not None:
                self.overwrite_phase(phase_to_overwrite, new_state, duration)
                return phase_to_overwrite
            else:
                # Last resort: reuse any existing phase with green for the desired lanes
                for idx, phase in enumerate(logic.phases):
                    phase_state = phase.state
                    controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
                    
                    for lane in desired_green_lanes:
                        if lane in controlled_lanes:
                            lane_idx = controlled_lanes.index(lane)
                            if lane_idx < len(phase_state) and phase_state[lane_idx].upper() == 'G':
                                print(f"[FALLBACK] Using existing phase {idx} for desired green lanes")
                                return idx
                                
                # If even that failed, return phase 0
                return 0
                
        except Exception as e:
            print(f"[ERROR] Failed to create or overwrite phase: {e}")
            import traceback
            traceback.print_exc()
            return 0
    def preload_phases_from_sumo(self):
        for idx, phase in enumerate(self._phase_defs):
            if not any(p['phase_idx'] == idx for p in self.apc_state.get('phases', [])):
                self.save_phase_record_to_supabase(
                    phase_idx=idx,
                    duration=phase.duration,
                    state_str=phase.state,
                    delta_t=0,
                    raw_delta_t=0,
                    penalty=0
                )

    # Inside AdaptivePhaseController, after updating phase duration:
    def log_phase_to_event_log(self, phase_idx, new_duration):
        # Find phase info in self.apc_state["phases"]
        phase = next((p for p in self.apc_state["phases"] if p["phase_idx"] == phase_idx), None)
        if not phase:
            base_duration = new_duration
            extended_time = 0
        else:
            base_duration = phase.get("base_duration", phase.get("duration", new_duration))
            extended_time = new_duration - base_duration
        # Now append to the event log (controller.phase_events)
        if hasattr(self, "controller") and hasattr(self.controller, "phase_events"):
            self.controller.phase_events.append({
                "tls_id": self.tls_id,
                "phase_idx": phase_idx,
                "base_duration": base_duration,
                "duration": new_duration,
                "extended_time": extended_time,
                "timestamp": datetime.datetime.now().isoformat()
            })
    def update_phase_duration_record(self, phase_idx, new_duration, extended_time=0):
        updated = False
        for p in self.apc_state.get("phases", []):
            if p["phase_idx"] == phase_idx:
                # Preserve original base_duration once set; do not shrink it later
                if "base_duration" not in p:
                    p["base_duration"] = float(new_duration - extended_time)
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

    def set_phase_from_API(self, phase_idx, requested_duration=None):

        print(f"[FIXED] set_phase_from_API({phase_idx}, requested_duration={requested_duration})")
        
        # Load phase from Supabase if available
        phase_record = self.load_phase_from_supabase(phase_idx)
        # Get base duration from record or SUMO logic (never fall back to min_green if logic is available)
        if phase_record:
            base_duration = phase_record.get("base_duration", phase_record.get("duration", self.min_green))
        else:
            try:
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
                phs = logic.getPhases()
                base_duration = float(phs[phase_idx].duration) if 0 <= phase_idx < len(phs) else self.min_green
            except Exception:
                base_duration = self.min_green
            
        # Calculate desired total duration for this activation (base + optional request), clipped
        desired_total = requested_duration if requested_duration is not None else base_duration
        desired_total = float(np.clip(desired_total, self.min_green, self.max_green))

        current_phase = traci.trafficlight.getPhase(self.tls_id)
        # Change phase if necessary (yellow transition handled)
        if current_phase != phase_idx:
            self.insert_yellow_phase_if_needed(current_phase, phase_idx)
            traci.trafficlight.setPhase(self.tls_id, phase_idx)
            # For a new activation, set remaining time to desired_total from "now"
            traci.trafficlight.setPhaseDuration(self.tls_id, desired_total)
            # Reset activation tracking
            self._reset_activation(phase_idx, base_duration, desired_total)
        else:
            # Same phase: do not restart it; only align remaining time if needed
            # Initialize activation if missing (e.g., first time or restored controller)
            if self.activation["phase_idx"] != phase_idx or self.activation["start_time"] == 0.0:
                self._reset_activation(phase_idx, base_duration, desired_total)
            # If caller provides a requested_duration, align remaining to the desired total
            if requested_duration is not None:
                self._maybe_update_phase_remaining(desired_total)

        # Update records/logs using the activation target
        elapsed = self._get_phase_elapsed()
        remaining = self._get_phase_remaining()
        total_now = max(desired_total, elapsed + remaining)  # conservative reporting
        extended_time = max(0.0, total_now - base_duration)
        self.update_phase_duration_record(phase_idx, total_now, extended_time)
        if hasattr(self, "log_phase_to_event_log"):
            self.log_phase_to_event_log(phase_idx, total_now)
        print(f"[FIXED/PATCH] Phase {current_phase} → {phase_idx}, desired_total={desired_total:.1f}s, now_total≈{total_now:.1f}s, extended≈{extended_time:.1f}s")
        return True


    def needs_yellow_transition(self, from_phase, to_phase):
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            from_state = logic.getPhases()[from_phase].state
            to_state = logic.getPhases()[to_phase].state
            
            # Need yellow if any lane goes from Green to Red
            for i in range(min(len(from_state), len(to_state))):
                if from_state[i].upper() == 'G' and to_state[i].upper() == 'R':
                    return True
            return False
        except:
            return False

    def find_yellow_transition_phase(self, from_phase, to_phase):
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            from_state = logic.getPhases()[from_phase].state
            to_state = logic.getPhases()[to_phase].state
            
            # Create expected yellow state
            yellow_state = []
            for i in range(min(len(from_state), len(to_state))):
                if from_state[i].upper() == 'G' and to_state[i].upper() == 'R':
                    yellow_state.append('y')
                else:
                    yellow_state.append(to_state[i])
            
            expected_yellow = ''.join(yellow_state)
            
            # Find matching phase
            for idx, phase in enumerate(logic.getPhases()):
                if phase.state == expected_yellow:
                    return idx
            return None
        except:
            return None

    def _delayed_phase_switch(self, phase_idx, requested_duration):
        try:
            traci.trafficlight.setPhase(self.tls_id, phase_idx)
            if requested_duration:
                traci.trafficlight.setPhaseDuration(self.tls_id, requested_duration)
            print(f"[DELAYED SWITCH] Completed transition to phase {phase_idx}")
        except Exception as e:
            print(f"[ERROR] Delayed phase switch failed: {e}")   
           
    def is_in_protected_left_phase(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        if current_phase >= len(logic.getPhases()):
            return None, None
        phase_state = logic.getPhases()[current_phase].state
        # Protected left: only one lane green, rest red, and that lane must be a left lane
        green_indices = [i for i, s in enumerate(phase_state) if s.upper() == 'G']
        if len(green_indices) == 1:
            lane_id = controlled_lanes[green_indices[0]]
            links = traci.lane.getLinks(lane_id)
            if any(len(link) > 6 and link[6] == 'l' for link in links):
                return lane_id, current_phase
        return None, None
    def step_extend_protected_left_if_blocked(self):

        lane_id, phase_idx = self.is_in_protected_left_phase()
        if lane_id is None:
            return False
            
        # Check if still blocked
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
        if not vehicles:
            return False
            
        # All vehicles stopped? Check speed and waiting time
        speeds = [traci.vehicle.getSpeed(vid) for vid in vehicles]
        front_vehicle = vehicles[0]
        stopped_time = traci.vehicle.getAccumulatedWaitingTime(front_vehicle)
        
        # FIXED: Check if blockage persists
        if max(speeds) < 0.2 and stopped_time > 5:
            # Still blocked - extend up to max_green total for this activation
            desired_total = float(self.max_green)
            # Ensure activation is aligned (if we just entered this phase without activation state)
            if self.activation["phase_idx"] != phase_idx:
                pr = self.load_phase_from_supabase(phase_idx)
                base_dur = pr.get("base_duration", self.min_green) if pr else self.min_green
                self._reset_activation(phase_idx, base_dur, desired_total)
            # Only update remaining if needed (no redundant appends)
            changed = self._maybe_update_phase_remaining(desired_total, buffer=0.5)
            if changed:
                current_phase = traci.trafficlight.getPhase(self.tls_id)
                elapsed = self._get_phase_elapsed()
                remaining = self._get_phase_remaining()
                total_now = elapsed + remaining
                base_d = self.activation["base_duration"] or self.min_green
                extended_time = max(0.0, total_now - base_d)
                print(f"[FIXED EXTEND/PATCH] Protected left phase {phase_idx} for lane {lane_id}: total≈{total_now:.1f}s (extended≈{extended_time:.1f}s)")
                self._log_apc_event({
                    "action": "extend_protected_left_active",
                    "lane_id": lane_id,
                    "phase": phase_idx,
                    "new_duration": total_now,
                    "extended_time": extended_time
                })
                return True
        # Not blocked anymore, should switch to a different phase
        elif traci.simulation.getTime() - self.last_phase_switch_sim_time > self.min_green:
            # Find next phase that serves vehicles
            best_phase = self.find_best_phase_for_traffic()
            if best_phase is not None and best_phase != phase_idx:
                print(f"[FIXED PHASE SWITCH] Protected left no longer needed, switching to phase {best_phase}")
                self.request_phase_change(best_phase, priority_type="normal")
                return True
                
        return False
    def ensure_phases_have_green(self):

        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        changed = False
        for idx, phase in enumerate(logic.getPhases()):
            if 'G' not in phase.state:
                # Find the first red (or any) and turn it green
                state_list = list(phase.state)
                for i, ch in enumerate(state_list):
                    if ch == 'r':
                        state_list[i] = 'G'
                        break
                else:
                    # If no red, just set the first position to green as a fallback
                    state_list[0] = 'G'
                new_state = ''.join(state_list)
                print(f"[PATCH] Phase {idx} had no green, fixing: {phase.state} → {new_state}")
                # Overwrite the phase with corrected state
                self.overwrite_phase(idx, new_state, phase.duration)
                changed = True
        if changed:
            print("[PATCH] All phases now have at least one green light.")
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

    def is_lane_green(self, lane_id):
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            if lane_id not in controlled_lanes:
                return False
                
            lane_idx = controlled_lanes.index(lane_id)
            current_phase_idx = traci.trafficlight.getPhase(self.tls_id)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            
            if current_phase_idx >= len(logic.getPhases()):
                return False
                
            phase_state = logic.getPhases()[current_phase_idx].state
            return (lane_idx < len(phase_state) and 
                    phase_state[lane_idx].upper() == 'G')
                    
        except Exception as e:
            print(f"[ERROR] Failed to check green status for {lane_id}: {e}")
            return False 
    def detect_blocked_left_turn_with_conflict(self):
        print(f"[DEBUG] Checking left-turn lanes for blockage...")
        
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            current_time = traci.simulation.getTime()
            
            # Get current traffic light state
            current_phase_idx = traci.trafficlight.getPhase(self.tls_id)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            
            if current_phase_idx >= len(logic.getPhases()):
                print(f"[DEBUG] Invalid phase index {current_phase_idx}, skipping left-turn check")
                return None, False
                
            phase_state = logic.getPhases()[current_phase_idx].state
            
            left_turn_candidates = []
            
            for lane_idx, lane_id in enumerate(controlled_lanes):
                # Check if this is a left-turn lane
                links = traci.lane.getLinks(lane_id)
                is_left = any(len(link) > 6 and link[6] == 'l' for link in links)
                
                if not is_left:
                    continue
                    
                # Check if lane currently has green light
                is_green = (lane_idx < len(phase_state) and 
                        phase_state[lane_idx].upper() == 'G')
                
                # Get lane statistics
                queue, waiting_time, mean_speed, density = self.get_lane_stats(lane_id)
                
                # Check for vehicles in the lane
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                has_vehicles = len(vehicles) > 0
                
                # Log detailed information for each left-turn lane
                #print(f"[DEBUG] Left lane {lane_id}: queue={queue}, speed={mean_speed:.2f}, "
                    #f"density={density:.3f}, green={is_green}, vehicles={len(vehicles)}")
                
                # Skip if no vehicles or not green
                if not has_vehicles:
                    print(f"[DEBUG] Left lane {lane_id}: No vehicles present")
                    continue
                    
                if not is_green:
                    print(f"[DEBUG] Left lane {lane_id}: Not currently green")
                    continue
                
                # Apply combined detection criteria
                queue_threshold = 3
                speed_threshold = 2.0
                density_threshold = 0.08
                
                # Check if queue meets minimum threshold
                if queue < queue_threshold:
                    print(f"[DEBUG] Left lane {lane_id}: Queue {queue} below threshold {queue_threshold}")
                    continue
                
                # Combined blockage detection
                speed_blocked = mean_speed < speed_threshold
                density_blocked = density > density_threshold
                
                if speed_blocked or density_blocked:
                    # Determine which criteria triggered
                    trigger_reason = []
                    if speed_blocked:
                        trigger_reason.append(f"speed criteria: {mean_speed:.2f} < {speed_threshold}")
                    if density_blocked:
                        trigger_reason.append(f"density criteria: {density:.3f} > {density_threshold}")
                    
                    trigger_description = " AND ".join(trigger_reason)
                    
                    print(f"[DEBUG] Left lane {lane_id}: BLOCKED ({trigger_description} AND queue={queue} >= {queue_threshold})")

                    if is_green and has_vehicles and queue >= queue_threshold:
                        self.left_block_steps[lane_id] += 1
                    else:
                        self.left_block_steps[lane_id] = 0

                    if self.left_block_steps[lane_id] < self.left_block_min_steps:
                        print(f"[DEBUG] Left lane {lane_id}: debounce {self.left_block_steps[lane_id]}/{self.left_block_min_steps} - not selecting yet")
                        continue
                    # Check for conflicting traffic
                    conflicting_lanes = self.get_conflicting_straight_lanes(lane_id)
                    has_conflict = any(
                        traci.lane.getLastStepVehicleNumber(conf_lane) > 0 
                        for conf_lane in conflicting_lanes
                    )
                    
                    if has_conflict:
                        print(f"[PROTECTED LEFT NEEDED] Lane {lane_id} requires protection due to {trigger_description}")
                        left_turn_candidates.append((lane_id, queue, mean_speed, density, trigger_description))
                    else:
                        print(f"[DEBUG] Left lane {lane_id}: Blocked but no conflicting traffic")
                else:
                    print(f"[DEBUG] Left lane {lane_id}: Not blocked (speed={mean_speed:.2f} >= {speed_threshold}, "
                        f"density={density:.3f} <= {density_threshold})")
            
            # Return the most severely blocked lane if any found
            if left_turn_candidates:
                # Sort by queue length (most blocked first)
                most_blocked = max(left_turn_candidates, key=lambda x: x[1])
                lane_id, queue, speed, density, reason = most_blocked
                
                #print(f"[PROTECTED LEFT SELECTED] Lane {lane_id} chosen for protection "
                    #f"(queue={queue}, {reason})")
                return lane_id, True
                
            print(f"[DEBUG] No left-turn lanes require protection")
            return None, False
            
        except Exception as e:
            print(f"[ERROR] Enhanced left turn detection failed: {e}")
            return None, False    
    def serve_protected_left_turn(self, left_lane):
        try:
            # Always create a new dedicated protected left phase
            phase_idx = self.create_protected_left_phase(left_lane)
            if phase_idx is None:
                print(f"[ERROR] Could not create protected left phase for {left_lane}")
                return False
            
            # Get queue info for dynamic timing
            queue = traci.lane.getLastStepHaltingNumber(left_lane)
            wait = traci.lane.getWaitingTime(left_lane)
            
            # Calculate dynamic green time based on queue length
            green_duration = min(self.max_green, 
                            max(self.min_green, queue * 2 + wait * 0.1))
            
            # Activate the protected left phase
            success = self.set_phase_from_API(phase_idx, requested_duration=green_duration)
            
            if success:
                print(f"[PROTECTED LEFT SUCCESS] Phase {phase_idx} activated for lane {left_lane} (duration: {green_duration}s)")
                return True
            else:
                print(f"[PROTECTED LEFT FAILED] Could not set phase {phase_idx}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Protected left handling failed: {e}")
            traceback.print_exc()
            return False

    def serve_true_protected_left_if_needed(self):
        lane_id, needs_protection = self.detect_blocked_left_turn_with_conflict()
        if not needs_protection or lane_id is None:
            return False
            
        # Get current phase and how long it's been active
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        current_time = traci.simulation.getTime()
        time_in_phase = current_time - self.last_phase_switch_sim_time
        
        # CRITICAL FIX: If we've been serving the same protected left phase for too long,
        # force a change to serve other movements
        if time_in_phase > 30:  # 30 seconds max for any protected left phase
            # Find a different phase that serves heavy traffic
            next_phase = self.find_best_phase_for_traffic()
            if next_phase is not None and next_phase != current_phase:
                print(f"[ROTATION] Protected left phase {current_phase} has been active for {time_in_phase:.1f}s. Rotating to phase {next_phase}")
                self.set_phase_from_API(next_phase)
                return True
        
        # Don't re-request the same phase we're already in
        phase_idx = self.create_protected_left_phase_for_lane(lane_id)
        if phase_idx is None:
            print(f"[PATCH] Could not create protected left phase for {lane_id}")
            return False
            
        # IMPORTANT FIX: Don't activate the same phase we're already in
        if phase_idx == current_phase:
            # Only extend the duration if needed
            remaining_time = traci.trafficlight.getNextSwitch(self.tls_id) - current_time
            if remaining_time < 15:  # Only extend if less than 15 seconds left
                print(f"[ROTATION] Already in protected left phase {phase_idx}, extending remaining to 15s")
                # Extend such that the remaining time is ~15s (i.e., total = elapsed + 15)
                desired_total = self._get_phase_elapsed() + 15.0
                self._maybe_update_phase_remaining(desired_total, buffer=0.2)            
            return True
            
        # Proceed with normal phase activation for a different phase
        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        wait = traci.lane.getWaitingTime(lane_id)
        green_duration = min(self.max_green, max(self.min_green, queue * 2 + wait * 0.1))

        # PATCH: don't preempt immediately unless min green satisfied; queue high-priority if not
        elapsed = current_time - self.last_phase_switch_sim_time
        if elapsed < self.min_green:
            # queue as emergency so it preempts at the soonest safe end-of-phase
            self.request_phase_change(phase_idx, priority_type='emergency', extension_duration=green_duration)
            print(f"[PATCH] Queued protected left phase {phase_idx} for lane {lane_id} (duration: {green_duration}s), elapsed={elapsed:.1f}s < min_green")
            return True

        self.set_phase_from_API(phase_idx, requested_duration=green_duration)
        print(f"[PATCH] Activated protected left phase {phase_idx} for lane {lane_id} (duration: {green_duration}s)")
        return True         
    def check_phase_limit(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        num_phases = len(logic.phases)
        max_phases = 12  # SUMO hard limit
        
        if num_phases >= max_phases:
            print(f"[WARNING] Traffic light {self.tls_id} at maximum phase limit ({max_phases})")
            return True
        return False

    # Use it before adding new phases in other methods
    def some_method_that_adds_phases(self):
        if self.check_phase_limit():
            # Consider reusing or overwriting phases instead of adding
            pass
    def insert_yellow_phase_if_needed(self, from_phase, to_phase):
        if from_phase == to_phase:
            return False
            
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
            if from_phase < 0 or from_phase >= len(logic.phases) or to_phase < 0 or to_phase >= len(logic.phases):
                print(f"[ERROR] Invalid phase indices: from={from_phase}, to={to_phase}")
                return False
                
            current_state = logic.phases[from_phase].state
            target_state = logic.phases[to_phase].state
            n = min(len(current_state), len(target_state))
            
            yellow_needed = False
            yellow_state = list('r' * len(current_state))
            
            # Build yellow state: 'y' for lanes going green->red, keep others
            for i in range(n):
                if current_state[i].upper() == 'G' and target_state[i].upper() == 'R':
                    yellow_state[i] = 'y'
                    yellow_needed = True
                else:
                    yellow_state[i] = current_state[i]
                    
            if not yellow_needed:
                return False
                
            yellow_state_str = ''.join(yellow_state)
            
            # First check if this yellow phase already exists
            yellow_phase_idx = None
            for idx, phase in enumerate(logic.phases):
                if phase.state == yellow_state_str:
                    yellow_phase_idx = idx
                    break
                    
            # If not found, check if we're at the phase limit
            if yellow_phase_idx is None:
                max_phases = 12  # SUMO hard limit
                
                if len(logic.phases) >= max_phases:
                    # At phase limit - find a suitable phase to overwrite
                    # Prioritize existing yellow phases for overwriting
                    candidates = []
                    for idx, phase in enumerate(logic.phases):
                        if 'y' in phase.state:
                            candidates.append(idx)
                            
                    if candidates:
                        # Overwrite an existing yellow phase
                        yellow_phase_idx = candidates[0]  # Take first yellow phase
                        # Update the existing yellow phase with new state
                        new_phases = list(logic.phases)
                        new_phases[yellow_phase_idx] = traci.trafficlight.Phase(3, yellow_state_str)
                        
                        new_logic = traci.trafficlight.Logic(
                            logic.programID, 
                            logic.type, 
                            from_phase,  # Current phase remains current
                            new_phases
                        )
                        
                        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
                        print(f"[FIXED] Overwrote yellow phase {yellow_phase_idx} with state {yellow_state_str}")
                    else:
                        # Can't add or overwrite - skip yellow phase
                        print(f"[WARNING] At phase limit ({max_phases}), cannot insert yellow phase")
                        return False
                else:
                    # We have room to add a new yellow phase
                    yellow_phase = traci.trafficlight.Phase(3, yellow_state_str)
                    phases = list(logic.phases) + [yellow_phase]
                    yellow_phase_idx = len(phases) - 1
                    
                    new_logic = traci.trafficlight.Logic(
                        logic.programID, 
                        logic.type, 
                        from_phase,  # Current phase remains current
                        phases
                    )
                    
                    traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, new_logic)
                    print(f"[FIXED] Created new yellow phase {yellow_phase_idx} with state {yellow_state_str}")
                    
            # Set to yellow phase and wait
            traci.trafficlight.setPhase(self.tls_id, yellow_phase_idx)
            traci.trafficlight.setPhaseDuration(self.tls_id, 3.0)  # 3 seconds yellow
            
            # Log the yellow transition
            self._log_apc_event({
                "action": "yellow_transition",
                "from_phase": from_phase,
                "to_phase": to_phase,
                "yellow_phase": yellow_phase_idx,
                "yellow_state": yellow_state_str
            })
            
            print(f"[FIXED YELLOW] Transitioning from phase {from_phase} → {to_phase} via yellow phase {yellow_phase_idx}")
            return True
        except Exception as e:
            print(f"[ERROR] Yellow phase insertion failed: {e}")
            return False    
    def generate_optimal_phase_set(self, controlled_lanes):
        phases = []
        phase_state_set = set()

        print(f"[PHASE GENERATION] Creating optimal phase set for {len(controlled_lanes)} lanes")

        # 1. Create green phases for each lane (ensures every lane gets served)
        for lane in controlled_lanes:
            green_state = self.create_phase_state(green_lanes=[lane])
            if green_state not in phase_state_set:
                phases.append(traci.trafficlight.Phase(self.min_green, green_state))
                phase_state_set.add(green_state)

        # 2. Create combination phases (optional: here, only pairs on different approaches)
        for i, lane1 in enumerate(controlled_lanes):
            for j, lane2 in enumerate(controlled_lanes[i+1:], i+1):
                try:
                    edge1 = traci.lane.getEdgeID(lane1)
                    edge2 = traci.lane.getEdgeID(lane2)
                    if edge1 != edge2:
                        combo_state = self.create_phase_state(green_lanes=[lane1, lane2])
                        if combo_state not in phase_state_set:
                            phases.append(traci.trafficlight.Phase(self.min_green, combo_state))
                            phase_state_set.add(combo_state)
                except Exception:
                    continue

        # 3. Generate ALL required yellow transitions for every phase-to-phase switch
        yellow_phases = []
        yellow_state_set = set()
        yellow_duration = 3
        n_phases = len(phases)
        for i, phase_from in enumerate(phases):
            for j, phase_to in enumerate(phases):
                if i == j:
                    continue
                from_state = phase_from.state
                to_state = phase_to.state
                yellow_needed = False
                yellow_state = []
                for k in range(min(len(from_state), len(to_state))):
                    if from_state[k].upper() == 'G' and to_state[k].upper() == 'R':
                        yellow_state.append('y')
                        yellow_needed = True
                    else:
                        if from_state[k].upper() == 'G' and to_state[k].upper() == 'R':
                            yellow_state.append('y')
                            yellow_needed = True
                        else:
                            yellow_state.append(from_state[k])
                yellow_state_str = ''.join(yellow_state)
                # Only add if needed, not duplicate, and not already a green phase
                if yellow_needed and yellow_state_str not in phase_state_set and yellow_state_str not in yellow_state_set:
                    yellow_phases.append(traci.trafficlight.Phase(yellow_duration, yellow_state_str))
                    yellow_state_set.add(yellow_state_str)

        # 4. Add all yellow phases to main phase list
        phases.extend(yellow_phases)
        phase_state_set.update(yellow_state_set)

        # 5. Verify every lane has at least one green phase
        served = [False] * len(controlled_lanes)
        for phase in phases:
            for idx, ch in enumerate(phase.state):
                if ch.upper() == 'G':
                    served[idx] = True
        for idx, was_served in enumerate(served):
            if not was_served:
                state = ''.join(['G' if i == idx else 'r' for i in range(len(controlled_lanes))])
                if state not in phase_state_set:
                    phases.append(traci.trafficlight.Phase(self.min_green, state))
                    phase_state_set.add(state)

        print(f"[PHASE GENERATION] Final phase set: {len(phases)} phases ({len(yellow_phases)} yellow transitions)")

        # 6. Log all phases for debugging
        for i, phase in enumerate(phases):
            phase_type = "YELLOW" if 'y' in phase.state else "GREEN"
            green_lanes = [controlled_lanes[j] for j in range(min(len(phase.state), len(controlled_lanes)))
                           if phase.state[j].upper() == 'G']
            print(f"  Phase {i}: {phase.state} ({phase_type}, duration={phase.duration}s) - Serves: {green_lanes}")

        return phases
    def enforce_min_green(self):
        current_sim_time = traci.simulation.getTime()
        elapsed = current_sim_time - self.last_phase_switch_sim_time
        if elapsed < self.min_green:
            print(f"[MIN_GREEN ENFORCED] {self.tls_id}: Only {elapsed:.2f}s since last switch, min_green={self.min_green}s")
            return False
        return True


    def adjust_phase_duration(self, delta_t):
        try:
            # Enforce minimum green time
            if not self.enforce_min_green() and not self.check_priority_conditions():
                print("[ADJUST BLOCKED] Min green active or priority conditions met.")
                return traci.trafficlight.getPhaseDuration(self.tls_id)

            current_phase = traci.trafficlight.getPhase(self.tls_id)
            # Initialize activation if missing
            if self.activation["phase_idx"] != current_phase or self.activation["start_time"] == 0.0:
                phase_record = self.load_phase_from_supabase(current_phase)
                base_duration = phase_record.get("base_duration", self.min_green) if phase_record else self.min_green
                # Estimate current total as elapsed + remaining to seed desired_total
                elapsed = self._get_phase_elapsed()
                remaining = self._get_phase_remaining()
                seed_total = max(base_duration, elapsed + remaining)
                self._reset_activation(current_phase, base_duration, seed_total)

            # Apply extension relative to base for this activation (no redundant appends)
            desired_total = self.apply_extension_delta(delta_t, buffer=0.3)
            elapsed = self._get_phase_elapsed()
            remaining = self._get_phase_remaining()
            new_total = max(desired_total or 0.0, elapsed + remaining)
            extended_time = max(0.0, new_total - (self.activation["base_duration"] or self.min_green))
            self.last_extended_time = extended_time

            print(f"\n[PHASE ADJUST PATCHED] Phase {current_phase}: desired_total={desired_total:.1f}s, now_total≈{new_total:.1f}s (Δt={delta_t:.1f}s, extended≈{extended_time:.1f}s)")
            print(f"  Weights: {self.weights}, Bonus: {getattr(self, 'last_bonus', 0)}, Penalty: {getattr(self, 'last_penalty', 0)}")
            return new_total
        except traci.TraCIException as e:
            print(f"[ERROR] Duration adjustment failed: {e}")
            return traci.trafficlight.getPhaseDuration(self.tls_id)   
    
    def assess_traffic_conditions(self):
        class DummyTrafficState:
            has_emergency = False
            emergency_lane = None
            max_queue = 0
            severe_threshold = 10000  # some big number
            most_congested_lane = None
            starvation_threshold = 10000
            def get_starved_lanes(self, threshold): return []
        return DummyTrafficState()
    def _get_phase_count(self, tls_id=None):
        try:
            if tls_id is None:
                tls_id = self.tls_id
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            return len(logic.getPhases())
        except Exception as e:
            print(f"[ERROR] _get_phase_count failed for {tls_id}: {e}")
            return 1  # Fallback to 1 phase (prevents crash)
    def control_step(self):
        self.phase_count += 1
        now = traci.simulation.getTime()
        print(f"[DEBUG] === control_step START === at sim time {now}")
        self.update_lane_serving_status()
        # --- ENHANCED: Protected left turn detection with detailed logging ---
        if self.step_extend_protected_left_if_blocked():
            print("[DEBUG] Protected left phase extended, skipping this step.")
            return
        current_time = traci.simulation.getTime()
        
        # Track time since last phase change
        time_since_change = current_time - self.last_phase_switch_sim_time
        
        # CRITICAL FIX: Force phase rotation after maximum time in one phase
        if time_since_change > self.max_green:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            next_phase = (current_phase + 1) % self._get_phase_count(self.tls_id)
            print(f"[FORCE ROTATION] Phase {current_phase} active for {time_since_change:.1f}s (>{self.max_green}s). Switching to phase {next_phase}")
            self.set_phase_from_API(next_phase)
            return
            
        # If any approach hasn't been served for too long, prioritize it
        longest_waiting_approach = None
        longest_wait = 0
        for lane_id in self.lane_ids:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            if queue > 0:  # Only consider lanes with vehicles
                lane_wait = current_time - self.last_served_time.get(lane_id, 0)
                if lane_wait > longest_wait:
                    longest_wait = lane_wait
                    longest_waiting_approach = lane_id
                    
        # Force service to starving approaches
        if longest_waiting_approach and longest_wait > 60:  # 60 seconds starvation limit
            phase_for_starving = self.find_phase_for_lane(longest_waiting_approach)
            if phase_for_starving is not None:
                # If we're already in that phase, align remaining time to a meaningful total (e.g., max_green)
                print(f"[STARVATION PREVENTION] Lane {longest_waiting_approach} waited {longest_wait:.1f}s. Activating/aligning phase {phase_for_starving}")
                self.set_phase_from_API(phase_for_starving, requested_duration=self.max_green)
                return
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        num_phases = len(logic.getPhases())
        print(f"[DEBUG] Current phase: {current_phase}, Num phases: {num_phases}")

        # 1. Gather per-lane statistics
        lane_stats = [
            (lid, *self.get_lane_stats(lid))
            for lid in self.lane_ids
        ]
        #print(f"[DEBUG] lane_stats: {lane_stats}")

        queues = [q for _, q, _, _, _ in lane_stats]
        waits = [w for _, _, w, _, _ in lane_stats]
        max_queue = max(queues) if queues else 0
        avg_queue = np.mean(queues) if queues else 0
        max_wait = max(waits) if waits else 0
        #print(f"[DEBUG] Queues: {queues}, Max: {max_queue}, Avg: {avg_queue:.2f}")
        #print(f"[DEBUG] Waits: {waits}, Max: {max_wait}")

        # 2. Enhanced protected left turn detection (only call ONCE here)
        blocked_left_lane, needs_protection = self.detect_blocked_left_turn_with_conflict()
        print(f"[DEBUG] Blocked left lane: {blocked_left_lane}, Needs protection? {needs_protection}")

        if needs_protection and blocked_left_lane:
            # REMOVED COOLDOWN: Immediate response to blocked left turns
            phase_idx = self.find_or_create_phase_for_lane(blocked_left_lane)
            print(f"[DEBUG] Protected left phase for {blocked_left_lane} is {phase_idx}")
            if phase_idx is not None:
                success = self.request_phase_change(
                    phase_idx,
                    priority_type='emergency',
                    extension_duration=self.max_green
                )
                print(f"[DEBUG] Request phase change for protected left: {success}")
                if success:
                    self._log_apc_event({
                        "action": "protected_left_turn_activated",
                        "lane_id": blocked_left_lane,
                        "phase_idx": phase_idx,
                        "reason": "enhanced_blockage_detection",
                        "detection_method": "combined_speed_density"
                    })
                    print(f"[PROTECTED LEFT ACTIVATED] Lane {blocked_left_lane} given emergency priority")
                    return
                else:
                    print(f"[WARNING] Failed to activate protected left for {blocked_left_lane}")

        # 3. Find most congested and most starved lanes
        maxq_idx = queues.index(max_queue) if queues else -1
        congested_lane = self.lane_ids[maxq_idx] if maxq_idx >= 0 else None
        print(f"[DEBUG] Most congested lane: {congested_lane}")

        # Track time since green for each lane to identify starvation
        starved_lanes = []
        starve_threshold = self.max_green * 1.5
        for lid in self.lane_ids:
            time_since_served = now - self.last_served_time.get(lid, 0)
            if time_since_served > starve_threshold:
                starved_lanes.append((lid, time_since_served))
        print(f"[DEBUG] Starved lanes (>{starve_threshold}s): {starved_lanes}")
        starved_lanes.sort(key=lambda x: x[1], reverse=True)

        # 4. Improved prioritization logic
        congestion_priority = False
        if max_queue > (1.5 * max(1, avg_queue)) and max_queue > 3:
            congestion_priority = True
        print(f"[DEBUG] Congestion priority? {congestion_priority}")

        # 5. Starvation prevention - highest priority
        critical_starvation = False
        critically_starved_lane = None
        critical_starve_threshold = self.max_green * 3
        for lid, time_since_served in starved_lanes:
            if time_since_served > critical_starve_threshold:
                critical_starvation = True
                critically_starved_lane = lid
                print(f"[DEBUG] Critically starved lane: {lid} (not served for {time_since_served}s)")
                break

        # 6. Prioritized handling with forced action
        if critical_starvation and critically_starved_lane is not None:
            phase_idx = self.find_or_create_phase_for_lane(critically_starved_lane)
            print(f"[DEBUG] Forcing phase for critically starved lane {critically_starved_lane}: {phase_idx}")
            if phase_idx is not None:
                print(f"[CRITICAL STARVATION] Forcing immediate phase {phase_idx} for lane {critically_starved_lane}")
                self.set_phase_from_API(phase_idx, requested_duration=self.max_green)
                self.last_served_time[critically_starved_lane] = now
                return

        elif starved_lanes:
            starved_lane, _ = starved_lanes[0]
            phase_idx = self.find_or_create_phase_for_lane(starved_lane)
            print(f"[DEBUG] Starved lane: {starved_lane}, phase: {phase_idx}, current: {current_phase}")
            if phase_idx is not None and phase_idx != current_phase:
                print(f"[STARVATION FIX] Forcing phase {phase_idx} for starved lane {starved_lane}")
                self.request_phase_change(phase_idx, priority_type="emergency", extension_duration=self.max_green)
                return

        elif congestion_priority and congested_lane is not None:
            phase_idx = self.find_or_create_phase_for_lane(congested_lane)
            ext = int(min(self.max_green * 2, self.min_green + 2.5 * max_queue + 0.7 * max_wait))
            print(f"[DEBUG] Congested lane: {congested_lane}, phase: {phase_idx}, ext: {ext}")
            if phase_idx is not None and phase_idx != current_phase:
                time_remaining = traci.trafficlight.getNextSwitch(self.tls_id) - current_time
                elapsed = current_time - self.last_phase_switch_sim_time
                # If min green satisfied, act now; otherwise queue a pending request.
                if elapsed >= self.min_green:
                    print("[CONGESTION FIX] Preempting now for severe congestion")
                    self.set_phase_from_API(phase_idx, requested_duration=ext)
                else:
                    print(f"[CONGESTION FIX] Queueing request (min green not met, elapsed={elapsed:.1f}s)")
                    self.request_phase_change(phase_idx, priority_type="heavy_congestion", extension_duration=ext)
                return

        if self.is_phase_ending():
            print("[DEBUG] Phase ending, processing pending requests.")
            if self.process_pending_requests_on_phase_end():
                return

        traffic_state = self.assess_traffic_conditions()
        event_type, event_lane = self.check_special_events()
        print(f"[DEBUG] Special event: {event_type}, lane: {event_lane}")
        if event_type == 'emergency_vehicle':
            target_phase = self.find_or_create_phase_for_lane(event_lane)
            print(f"[DEBUG] Emergency vehicle found, target phase: {target_phase}")
            if target_phase is not None:
                self.request_phase_change(target_phase, priority_type='emergency')
            return

        # Fallback: RL logic - lowest priority
        optimal_phase = self.rl_agent.select_optimal_phase(traffic_state)
        state = np.array([
            current_phase,
            num_phases,
            *[self.get_lane_stats(lane_id)[0] for lane_id in self.lane_ids],
            *[self.get_lane_stats(lane_id)[1] for lane_id in self.lane_ids],
            self.phase_count
        ])
        print(f"[DEBUG] RL state: {state}")

        self.rl_agent.action_size = num_phases
        try:
            action_result = self.rl_agent.get_action(state, tl_id=self.tls_id, action_size=num_phases)
            print(f"[DEBUG] RL agent action_result: {action_result}")
            if isinstance(action_result, tuple) or isinstance(action_result, list):
                action = int(action_result[0])
                phase_duration = action_result[1]
            else:
                action = int(action_result)
                phase_duration = None
            if action >= num_phases:
                print(f"[WARNING] RL agent gave out-of-bounds action: {action}")
                action = 0
        except Exception as e:
            print(f"[RL] Fallback to round-robin due to error: {e}")
            action = (current_phase + 1) % num_phases
            phase_duration = None

        if action != current_phase:
            print(f"[DEBUG] RL agent requests phase change: {current_phase} → {action}, duration: {phase_duration}")
            self.request_phase_change(action, priority_type='normal', extension_duration=phase_duration)

        reward = self.calculate_reward()
        print(f"[DEBUG] Step reward: {reward}")
        if hasattr(self, 'prev_state') and hasattr(self, 'prev_action'):
            print(f"[DEBUG] RL agent Q-table update: prev_state, prev_action: {self.prev_action}")
            self.rl_agent.update_q_table(self.prev_state, self.prev_action, reward, state, tl_id=self.tls_id)
        self.prev_state = state
        self.prev_action = action

        try:
            if self.pending_phase_request is None:
                # Compute extension delta from reward vs target
                _, delta_t, _ = self.calculate_delta_t_and_penalty(reward)
                # Apply extension without redundant resets
                self.apply_extension_delta(delta_t, buffer=0.3)
        except Exception as e:
            print(f"[PATCH][WARN] Extension delta application skipped: {e}")

        self.reward_history.append(reward)
        if self.phase_count % 10 == 0:
            print(f"[DEBUG] Updating RL target network at phase_count {self.phase_count}")
            self.update_R_target()
        try:
            self.emit_extension_telemetry(threshold=0.5)
        except Exception as e:
            print(f"[TELEMETRY][WARN] Could not emit extension telemetry: {e}")

        print(f"[DEBUG] === control_step END ===")
class EnhancedQLearningAgent:
    def __init__(
        self, state_size, action_size, adaptive_controller,
        learning_rate=0.1, discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
        q_table_file="enhanced_q_table.pkl", mode="train", adaptive_params=None,
        max_action_space=20, optimistic_init=10.0
    ):
        self.state_size = state_size
        self.max_action_space = max_action_space
        self.action_size = min(action_size, max_action_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.training_data = []
        self.q_table_file = q_table_file
        self._loaded_training_count = 0
        self.reward_history = []
        self.mode = mode
        #self.logger = logging.getLogger(__name__)
        self.optimistic_init = optimistic_init
        self.adaptive_params = adaptive_params or {
            'min_green': 30, 'max_green': 80, 'starvation_threshold': 30, 'reward_scale': 40,
            'queue_weight': 0.6, 'wait_weight': 0.3, 'flow_weight': 0.5, 'speed_weight': 0.2, 
            'left_turn_priority': 1.2, 'empty_green_penalty': 15, 'congestion_bonus': 10,
            'severe_congestion_threshold': 10
        }
        self.severe_threshold = self.adaptive_params.get('severe_congestion_threshold', 10)
        self.adaptive_controller = adaptive_controller
        if mode == "eval":
            self.epsilon = 0.0
        elif mode == "adaptive":
            self.epsilon = 0.01
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

    def _state_to_key(self, state, tl_id=None):
        try:
            arr = np.round(np.array(state), 2) if isinstance(state, (np.ndarray, list)) else state
            key = tuple(arr.tolist()) if isinstance(arr, np.ndarray) else tuple(arr)
            return (tl_id, key) if tl_id is not None else key
        except Exception:
            return (tl_id, (0,)) if tl_id is not None else (0,)

    def get_action(self, state, tl_id=None, action_size=None, strategy="epsilon_greedy", valid_actions_mask=None):
        action_size = action_size or self.action_size
        key = self._state_to_key(state, tl_id)
        if key not in self.q_table or len(self.q_table[key]) < self.max_action_space:
            arr = np.full(self.max_action_space, self.optimistic_init, dtype=float)
            if key in self.q_table and len(self.q_table[key]) > 0:
                arr[:len(self.q_table[key])] = self.q_table[key]
            self.q_table[key] = arr
        qs = self.q_table[key][:self.max_action_space]
        mask = np.zeros(self.max_action_space, dtype=bool)
        mask[:action_size] = 1
        if valid_actions_mask is not None:
            mask &= valid_actions_mask[:self.max_action_space]
        masked_qs = np.where(mask, qs, -np.inf)
        # Exploration
        if self.mode == "train" and np.random.rand() < self.epsilon:
            valid_idxs = np.where(mask)[0]
            return int(np.random.choice(valid_idxs)) if len(valid_idxs) > 0 else 0
        # Exploitation
        if np.all(np.isneginf(masked_qs)):
            return 0
        return int(np.argmax(masked_qs))
# Add to the RL agent class:

    def select_and_apply_phase(self, state_vector, traffic_state=None):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.adaptive_controller.tls_id)
        priority_lanes = []
        
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for vid in vehicles:
                try:
                    if 'emergency' in traci.vehicle.getTypeID(vid):
                        priority_lanes.append(lane)
                        break
                except:
                    pass
     
        if not priority_lanes:
            blocked_left_lane, needs_protection = self.adaptive_controller.detect_blocked_left_turn_with_conflict()
            if needs_protection and blocked_left_lane:
                priority_lanes.append(blocked_left_lane)
                is_protected_left = True
            else:
                is_protected_left = False
        else:
            is_protected_left = False
        
        # If we have priority lanes, create or overwrite a phase for them
        if priority_lanes:
            if is_protected_left:
                # Use the protected left turn phase creation with overwrite
                phase_idx = self.adaptive_controller.create_protected_left_phase_for_lane(priority_lanes[0])
            else:
                # Create or overwrite phase for emergency vehicles
                phase_idx = self.adaptive_controller.rl_create_or_overwrite_phase(
                    state_vector, 
                    desired_green_lanes=priority_lanes
                )
                
            # Apply the selected phase
            if phase_idx is not None:
                # PATCH: for non-priority RL actions, request the change so it happens at phase end (reduces churn)
                self.adaptive_controller.request_phase_change(
                    phase_idx,
                    priority_type='normal',
                    extension_duration=None
                )
                return phase_idx
            
        # No priority lanes, use regular RL decision making
        # Get action from Q-table or policy
        action = self.get_action(state_vector)
        
        # Check if action is a phase index or needs to be interpreted
        if isinstance(action, int) and action < traci.trafficlight.getPhase(self.adaptive_controller.tls_id):
            # Direct phase index
            phase_idx = action
        else:
            # Interpret action and create/overwrite phase
            # Calculate which lanes should be green based on queue lengths
            lanes_by_queue = sorted(
                controlled_lanes, 
                key=lambda l: traci.lane.getLastStepHaltingNumber(l),
                reverse=True
            )
            
            # Take top N lanes with highest queues
            top_lanes = lanes_by_queue[:min(3, len(lanes_by_queue))]
            
            # Create or overwrite phase
            phase_idx = self.adaptive_controller.rl_create_or_overwrite_phase(
                state_vector, 
                desired_green_lanes=top_lanes
            )
        
        # Apply the selected phase
        if phase_idx is not None:
            self.adaptive_controller.set_phase_from_API(
                phase_idx,
                requested_duration=None  # Use default duration
            )
            return phase_idx
        
        # Fallback to current phase
        return traci.trafficlight.getPhase(self.adaptive_controller.tls_id)
    def update_q_table(self, state, action, reward, next_state, tl_id=None, extra_info=None, action_size=None):
        if self.mode == "eval" or not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return
        if reward is None or np.isnan(reward) or np.isinf(reward):
            return
        action_size = action_size or self.action_size
        sk, nsk = self._state_to_key(state, tl_id), self._state_to_key(next_state, tl_id)
        for k in [sk, nsk]:
            if k not in self.q_table or len(self.q_table[k]) < self.max_action_space:
                arr = np.full(self.max_action_space, self.optimistic_init)
                if k in self.q_table and len(self.q_table[k]) > 0:
                    arr[:len(self.q_table[k])] = self.q_table[k]
                self.q_table[k] = arr
        q, nq = self.q_table[sk][action], np.max(self.q_table[nsk][:self.max_action_space])
        new_q = q + self.learning_rate * (reward + self.discount_factor * nq - q)
        if not (np.isnan(new_q) or np.isinf(new_q)):
            self.q_table[sk][action] = new_q
        # Log training data for future analysis/persistence
        entry = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'q_value': self.q_table[sk][action],
            'timestamp': time.time(),
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'tl_id': tl_id,
            'adaptive_params': self.adaptive_params.copy()
        }
        if extra_info:
            entry.update({k: v for k, v in extra_info.items() if k != "reward"})
        self.training_data.append(entry)
        self._update_adaptive_parameters(reward)

    def _update_adaptive_parameters(self, performance_value):
        # Optionally update adaptive params based on performance (stub for extensibility)
        pass

    def switch_or_extend_phase(self, state, green_lanes, force_protected_left=False):
        print(f"[DEBUG][RL Agent] switch_or_extend_phase with state={state}, green_lanes={green_lanes}, force_protected_left={force_protected_left}")
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
        self.adaptive_controller.set_phase_from_API(phase_idx)
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

    def select_optimal_phase(self, traffic_state):
        if getattr(traffic_state, "has_emergency", False):
            return self.get_emergency_phase(traffic_state.emergency_lane)
        severe_threshold = self.adaptive_params.get('severe_congestion_threshold', 10)
        if getattr(traffic_state, "max_queue", 0) > severe_threshold:
            return self.get_congestion_relief_phase(getattr(traffic_state, "most_congested_lane", None))
        starved_lanes = traffic_state.get_starved_lanes(self.adaptive_params.get('starvation_threshold', 30))
        if starved_lanes:
            return self.get_starvation_relief_phase(starved_lanes[0])
        return self.select_phase(traffic_state)

    def select_phase(self, traffic_state):
        try:
            state_vector = np.array([
                getattr(traffic_state, 'max_queue', 0),
                getattr(traffic_state, 'emergency_lane', -1) if getattr(traffic_state, 'has_emergency', False) else -1,
                getattr(traffic_state, 'starvation_threshold', 0),
            ])
            action = self.get_action(state_vector)
            return action
        except Exception as e:
            print(f"[select_phase ERROR]: {e}")
            return 0

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
        return self.adaptive_controller.create_or_extend_protected_left_phase(left_lane, delta_t)

    def calculate_total_reward(self, adaptive_R, rl_reward):
        return adaptive_R + rl_reward

    def _get_action_name(self, action):
        return {
            0: "Set Green", 1: "Next Phase", 2: "Extend Phase",
            3: "Shorten Phase", 4: "Balanced Phase"
        }.get(action, f"Unknown Action {action}")

    def load_model(self, filepath=None):
        filepath = filepath or self.q_table_file
        print(f"Attempting to load Q-table from: {filepath}")
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
                print(f"Loaded Q-table with {len(self.q_table)} states from {filepath}")
                if adaptive_params:
                    print("📋 Loaded adaptive parameters from previous run")
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

    # ---- Example stubs for hierarchical selection ----
    def get_emergency_phase(self, emergency_lane):
        return 0

    def get_congestion_relief_phase(self, congested_lane):
        return 0

    def get_starvation_relief_phase(self, starved_lane):
        return 0

class UniversalSmartTrafficController:

    DILEMMA_ZONE_THRESHOLD = 12.0  # meters

    def __init__(self, sumocfg_path=None, mode="train", config=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.mode = mode
        self.step_count = 0
        lane_ids = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        tls_id = traci.trafficlight.getIDList()[0]
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
        self.adaptive_phase_controller = AdaptivePhaseController(
            lane_ids=lane_ids,
            tls_id=tls_id,
            alpha=1.0,
            min_green=10,
            max_green=60
        )
        # PATCH: wire controller reference so APCs can emit telemetry/events to controller.phase_events
        self.adaptive_phase_controller.controller = self
        self.apc = self.adaptive_phase_controller
        self.rl_agent = EnhancedQLearningAgent(
            state_size=12,
            action_size=8,
            adaptive_controller=self.adaptive_phase_controller,
            mode=mode
        )
        self.adaptive_phase_controller.rl_agent = self.rl_agent
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

        # RL/Adaptive controller setup (single intersection default)

        self.adaptive_phase_controller = AdaptivePhaseController(
            lane_ids=lane_ids,
            tls_id=tls_id,
            alpha=1.0,
            min_green=10,
            max_green=60
        )
        self.apc = self.adaptive_phase_controller
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
        # Print phase info for each intersection
        self.tl_max_phases = {}
        for tl_id in traci.trafficlight.getIDList():
            phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            for i, phase in enumerate(phases):
                print(f"  Phase {i}: {phase.state} (duration {getattr(phase, 'duration', '?')})")

        # Main lane and traffic light setup
        self.lane_id_list = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
        
        for tl_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.tl_max_phases[tl_id] = len(logic.phases)
        self.tl_action_sizes = {tl_id: len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                                for tl_id in traci.trafficlight.getIDList()}
        tls_list = traci.trafficlight.getIDList()

        if tls_list:
            tls_id = tls_list[0]
            lane_ids = traci.trafficlight.getControlledLanes(tls_id)
            self.adaptive_phase_controller = AdaptivePhaseController(
                lane_ids=lane_ids,
                tls_id=tls_id,
                alpha=1.0,
                min_green=10,
                max_green=60
            )
            # PATCH: wire controller reference
            self.adaptive_phase_controller.controller = self
            self.rl_agent = EnhancedQLearningAgent(
                state_size=12,
                action_size=self.tl_action_sizes[tls_id],
                adaptive_controller=self.adaptive_phase_controller,
                mode=self.mode
            )
            self.adaptive_phase_controller.rl_agent = self.rl_agent

        # One APC per intersection
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
            # PATCH: wire controller reference
            apc.controller = self
            apc.rl_agent = self.rl_agent
            self.adaptive_phase_controllers[tls_id] = apc
        self.lane_id_to_idx = {lid: i for i, lid in enumerate(self.lane_id_list)}
        self.idx_to_lane_id = dict(enumerate(self.lane_id_list))
        for lid in self.lane_id_list:
            self.last_green_time[lid] = 0.0
        self.subscribe_lanes(self.lane_id_list)
        self.left_turn_lanes, self.right_turn_lanes = self.detect_turning_lanes()
        self.lane_lengths = {lid: traci.lane.getLength(lid) for lid in self.lane_id_list}
        self.lane_edge_ids = {lid: traci.lane.getEdgeID(lid) for lid in self.lane_id_list}


        left, right = set(), set()
        for lid in self.lane_id_list:

            for c in traci.lane.getLinks(lid):
                idx = 6 if len(c) > 6 else 3 if len(c) > 3 else None
                if idx and c[idx] == 'l':
                    left.add(lid)
                if idx and c[idx] == 'r':
                    right.add(lid)
        return left, right

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
        for lane in controlled_lanes:
            idx = self.lane_id_to_idx.get(lane)
            if idx is not None and current_time - self.last_green_time[idx] > self.adaptive_params['starvation_threshold']:
                return lane
        return None
    
    def _handle_protected_left_turn(self, tl_id, controlled_lanes, lane_data, current_time):
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
        return lane_id in self.left_turn_lanes

    def _get_traffic_light_logic(self, tl_id):
        if tl_id not in self.tl_logic_cache:
            try:
                self.tl_logic_cache[tl_id] = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            except Exception as e:
                print(f"Error getting logic for TL {tl_id}: {e}")
                return None
        return self.tl_logic_cache[tl_id]

    def _get_phase_count(self, tl_id):
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if logic:
                return len(logic.phases)
            return len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id))
        except Exception as e:
            print(f"Error getting phase count for {tl_id}: {e}")
            return 4

    def _get_phase_name(self, tl_id, phase_idx):
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
    def initialize_controller_phases(self):
        print("[PHASE GENERATION] Hybrid initialization: using sumocfg phases as base set...")
        for tls_id in traci.trafficlight.getIDList():
            apc = self.adaptive_phase_controllers[tls_id]

            base_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            base_phases = [Phase(p.duration, p.state) for p in base_logic.phases]
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)

            # PATCH: Fix phases after loading
            base_phases = fix_phase_states_and_missing_greens(base_phases, controlled_lanes, min_green=apc.min_green)

            print(f"[PHASE GENERATION] Using {len(base_phases)} base phases from sumocfg for {tls_id}")

            # Create new logic with ONLY base phases
            new_logic = Logic(
                programID=f"CONTROLLER-{tls_id}",
                type=0,
                currentPhaseIndex=0,
                phases=base_phases
            )
            try:
                traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_logic)
                print(f"[PHASE GENERATION PATCH] Set {len(base_phases)} phases for {tls_id} (sumocfg base only, fixed)")

                # Save each phase to Supabase (for RL/controller reference)
                for idx, phase in enumerate(base_phases):
                    apc.save_phase_record_to_supabase(
                        phase_idx=idx,
                        duration=phase.duration,
                        state_str=phase.state,
                        delta_t=0,
                        raw_delta_t=0,
                        penalty=0,
                        event_type="sumocfg_base"
                    )

            except Exception as e:
                print(f"[ERROR PATCH] Failed to set base phases for {tls_id}: {e}")
    
    def _add_new_green_phase_for_lane(self, tl_id, lane_id, min_green=None, yellow=3):
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
        
        # Build green state for the target lane
        phase_state[lane_idx] = 'G'
        green_state_str = "".join(phase_state)
        
        # Retrieve max phases for this TLS (auto-detected from initial logic)
        max_phases = None
        # Preferred: use a controller attribute if set
        if hasattr(self, "tl_max_phases") and tl_id in self.tl_max_phases:
            max_phases = self.tl_max_phases[tl_id]
        else:
            # Fallback: use initial number of phases found in the first loaded logic
            try:
                all_logics = traci.trafficlight.getAllProgramLogics(tl_id)
                if all_logics:
                    max_phases = len(all_logics[0].phases)
                else:
                    max_phases = 12  # fallback default
            except Exception:
                max_phases = 12  # fallback default

        # If we are at or above the phase limit, overwrite instead of appending
        if len(phases) >= max_phases:
            print(f"[PHASE LIMIT] {tl_id} at max phases ({max_phases}), will overwrite")
            apc = self.adaptive_phase_controllers[tl_id]
            idx_to_overwrite = apc.find_phase_to_overwrite(green_state_str)
            if idx_to_overwrite is not None:
                apc.overwrite_phase(idx_to_overwrite, green_state_str, min_green)
                return idx_to_overwrite
            print(f"[PHASE LIMIT] Could not find phase to overwrite, fallback to phase 0")
            return 0

        # Otherwise, append as usual
        # Green phase for the target lane
        new_green_phase = traci.trafficlight.Phase(min_green, green_state_str, 0, 0)
        # Yellow phase for the target lane (only that lane yellow, rest red)
        phase_state[lane_idx] = 'y'
        yellow_state_str = "".join(phase_state)
        new_yellow_phase = traci.trafficlight.Phase(yellow, yellow_state_str, 0, 0)
        # Add both phases to the end
        phases.append(new_green_phase)
        phases.append(new_yellow_phase)
        # Update logic and push to SUMO
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type, len(phases) - 2, phases
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_id, new_logic)
        print(f"[NEW PHASE] Added green+yellow phase for lane {lane_id} at {tl_id}")

        # Update phase logic cache and RL agent action space if applicable
        if hasattr(self, "tl_logic_cache"):
            self.tl_logic_cache[tl_id] = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        if hasattr(self, "tl_action_sizes"):
            self.tl_action_sizes[tl_id] = len(phases)
        if hasattr(self, "rl_agent") and hasattr(self.rl_agent, "action_size"):
            self.rl_agent.action_size = max(self.tl_action_sizes.values())
        # Optionally encourage RL agent to try new phase
        if hasattr(self, "rl_agent") and hasattr(self.rl_agent, "epsilon"):
            self.rl_agent.epsilon = min(0.7, self.rl_agent.epsilon + 0.4)
        print(f"[ACTION SPACE] tl_id={tl_id} now n_phases={len(phases)}")
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
                        apc.set_phase_from_API(pending_phase)
                        self.last_phase_change[tl_id] = current_time
                        del self.pending_next_phase[tl_id]
                        logic = self._get_traffic_light_logic(tl_id)
                        n_phases = len(logic.phases) if logic else 0
                        current_phase = traci.trafficlight.getPhase(tl_id)
                        if n_phases == 0 or current_phase >= n_phases or current_phase < 0:
                            apc.set_phase_from_API(max(0, n_phases - 1))
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
                    new_phase_added = False
                    if starved_phase is None:
                       starved_phase = self._add_new_green_phase_for_lane(
                            tl_id, most_starved_lane, min_green=self.adaptive_params['min_green'], yellow=3)
                       logic = self._get_traffic_light_logic(tl_id)
                       new_phase_added = True
                    if starved_phase is not None and current_phase != starved_phase:
                       switched = self._switch_phase_with_yellow_if_needed(
                            tl_id, current_phase, starved_phase, logic, controlled_lanes, lane_data, current_time)
                       logic = self._get_traffic_light_logic(tl_id)
                       n_phases = len(logic.phases) if logic else 1
                       current_phase = traci.trafficlight.getPhase(tl_id)
                       if current_phase >= n_phases:
                           apc.set_phase_from_API(n_phases - 1)
                        # PATCH: Always set via APC PKL, not direct setPhase/setPhaseDuration
                       if not switched:
                           apc.set_phase_from_API(starved_phase)
                           self.last_phase_change[tl_id] = current_time
                        # Encourage RL agent to try new phase
                       if new_phase_added:
                           self.rl_agent.epsilon = min(0.5, self.rl_agent.epsilon + 0.1)
                    self.last_green_time[self.lane_id_to_idx[most_starved_lane]] = current_time
                    self.debug_green_lanes(tl_id, lane_data)
                    continue

                # --- RL phase overwrite PATCH START ---
                # Enable overwrite flag once before main RL logic
                if not hasattr(self.rl_agent, 'overwrite_enabled'):
                    self.rl_agent.overwrite_enabled = True
                    print("[PHASE OVERWRITE] Enabled phase overwriting for RL agent")


                # Normal RL control (old logic, fallback)
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
                if self.rl_agent.overwrite_enabled:
                    state = self._create_intersection_state_vector(tl_id, self.intersection_data)
                    phase_idx = self.rl_agent.select_and_apply_phase(state)
                    # Update any necessary stats or tracking
                    self.last_phase_change[tl_id] = current_time
                    continue
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
                        apc.set_phase_from_API(action)
                        self.last_phase_change[tl_id] = current_time
                        self._process_rl_learning(self.intersection_data, lane_data, current_time)
                self.debug_green_lanes(tl_id, lane_data)

        except Exception as e:
            self.logger.error(f"Error in run_step: {e}", exc_info=True)
    def _phase_has_traffic(self, logic, action, controlled_lanes, lane_data):
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
        try:
            total = sum(c for (tl, _), c in self.phase_utilization.items() if tl == tl_id)
            if not total: return 1.0
            count = self.phase_utilization.get((tl_id, phase_index), 0)
            return min(1.0, max(0.1, count/total))
        except: return 1.0

    def _adjust_traffic_lights(self, lane_data, lane_status, current_time):
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
        amb = [l for l in controlled_lanes if lane_data.get(l, {}).get('ambulance')]
        if amb: return self._handle_ambulance_priority(tl_id, amb, lane_data, current_time)
        left = [l for l in controlled_lanes if lane_data.get(l, {}).get('left_turn') and
                (lane_data[l]['queue_length'] > 3 or lane_data[l]['waiting_time'] > 10)]
        if left: return self._handle_protected_left_turn(tl_id, left, lane_data, current_time)
        return False

    def _handle_ambulance_priority(self, tl_id, controlled_lanes, lane_data, current_time):
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
        base = self.adaptive_params['min_green']
        queue = min(lane_data['queue_length'] * 0.7, 15)
        density = min(lane_data['density'] * 5, 10)
        bonus = 10 if lane_data.get('ambulance') else 0
        total = base + queue + density + bonus
        return min(max(total, base), self.adaptive_params['max_green'])

    def _find_phase_for_lane(self, tl_id, target_lane):
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
    start_universal_simulation(
        sumocfg_path=args.sumo,
        use_gui=args.gui,
        max_steps=args.max_steps,
        episodes=args.episodes,
        num_retries=args.num_retries,
        retry_delay=args.retry_delay,
        mode=args.mode)

def start_universal_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1, num_retries=1, retry_delay=1, mode="train"):
    global controller
    controller = None

    def simulation_loop():
        global controller
        try:
            for episode in range(episodes):
                print(f"\n{'='*50}\n🚦 STARTING UNIVERSAL EPISODE {episode + 1}/{episodes}\n{'='*50}")
                sumo_binary = "sumo-gui" if use_gui else "sumo"
                sumo_cmd = [
                    os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary),
                    '-c', sumocfg_path, '--start', '--quit-on-end'
                ]
                traci.start(sumo_cmd)
                controller = UniversalSmartTrafficController(sumocfg_path=sumocfg_path, mode=mode)
                controller.initialize()
                if hasattr(controller, "initialize_controller_phases"):
                    print("[PATCH] Overriding SUMO default TL phases with controller logic.")
                    controller.initialize_controller_phases()
                    for tl_id in traci.trafficlight.getIDList():
                        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                        for i, phase in enumerate(logic.phases):
                            print(f"  [AFTER OVERRIDE] {tl_id} Phase {i}: {phase.state} (duration {getattr(phase, 'duration', '?')})")
                controller.current_episode = episode + 1
                step, tstart = 0, time.time()
                while traci.simulation.getMinExpectedNumber() > 0:
                    if max_steps and step >= max_steps:
                        print(f"Reached max steps ({max_steps}), ending episode.")
                        break
                    controller.run_step()
                    traci.simulationStep()
                    step += 1
                    if step % 1000 == 0:
                        print(f"Episode {episode + 1}: Step {step} completed, elapsed: {time.time()-tstart:.2f}s")
                print(f"Episode {episode + 1} completed after {step} steps")
                controller.end_episode()
                traci.close()
                if episode < episodes - 1:
                    time.sleep(2)
            print(f"\n🎉 All {episodes} episodes completed!")
        except Exception as e:
            print(f"Error in universal simulation: {e}")
        finally:
            try:
                traci.close()
            except Exception:
                pass
            print("Simulation resources cleaned up")
    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.start()
    while controller is None or not hasattr(controller, "adaptive_phase_controllers"):
        time.sleep(0.1)
    while True:
        try:
            if traci.trafficlight.getIDList():
                break
        except Exception:
            pass
        time.sleep(0.1)
    display = SmartIntersectionTrafficDisplay(controller.phase_events, controller=controller, poll_interval=1)  
    display.start()
    sim_thread.join()
    display.stop()
    return controller
if __name__ == "__main__":
    main()