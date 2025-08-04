from collections import defaultdict
import os, sys, traci, threading, warnings, time, argparse, traceback, pickle, datetime, logging
import numpy as np
warnings.filterwarnings('ignore')
from traffic_light_display import TrafficLightPhaseDisplay
from traci._trafficlight import Logic, Phase
from supabase import create_client
import os, json, datetime
import threading
from collections import deque
MAX_PHASE_DURATION = 60 
from traffic_light_display import TrafficLightPhaseDisplay


SUPABASE_URL = "https://zckiwulodojgcfwyjrcx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")

os.environ.setdefault('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)
def to_builtin_type(obj):
    if isinstance(obj, dict):
        return {to_builtin_type(k): to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin_type(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj
# --- Template Phase Manager ---
class TemplatePhaseManager:
    def __init__(self, supabase):
        self.supabase = supabase
        self.templates = {}
        self.last_loaded = None
        self.lock = threading.Lock()

    def load_templates(self):
        try:
            response = self.supabase.table("template_phases").select("*").execute()
            with self.lock:
                self.templates.clear()
                for item in response.data:
                    tname = item.get("template_name")
                    if tname not in self.templates:
                        self.templates[tname] = []
                    self.templates[tname].append(item)
                self.last_loaded = datetime.datetime.now()
                logger.info(f"Loaded {len(response.data)} phase templates")
        except Exception as e:
            logger.warning(f"Failed to load template phases: {e}")

    def get_template(self, template_name):
        with self.lock:
            return self.templates.get(template_name, [])

    def refresh_if_needed(self, interval_seconds=60):
        if not self.last_loaded or (datetime.datetime.now() - self.last_loaded).total_seconds() > interval_seconds:
            self.load_templates()

    def apply_template_to_phase(self, tl_id, lane_id, event_type):
        """Apply template-based phase creation for specific events"""
        templates = self.get_template(event_type) or self.get_template("default")
        if not templates:
            return None
            
        # Find best template for this lane/event
        best_template = None
        for template in templates:
            if template.get("lane_pattern") and lane_id in template["lane_pattern"]:
                best_template = template
                break
        
        if not best_template and templates:
            best_template = templates[0]  # Use first available
            
        return best_template

# --- Async Batch Writer ---
class AsyncBatchWriter(threading.Thread):
    def __init__(self, supabase, batch_interval=5):
        super().__init__(daemon=True)
        self.supabase = supabase
        self.batch_interval = batch_interval
        self.buffers = {
            "apc_states": deque(),
            "rl_decisions": deque(),
            "phase_events": deque(),
        }
        self.lock = threading.Lock()
        self.running = True

    def buffer(self, table, data):
        with self.lock:
            self.buffers[table].append(data)

    def run(self):
        while self.running:
            self.flush_all()
            time.sleep(self.batch_interval)

    def flush_all(self):
        with self.lock:
            for table, buf in self.buffers.items():
                if buf:
                    try:
                        payload = list(buf)
                        self.supabase.table(table).insert(payload).execute()
                        buf.clear()
                    except Exception as e:
                        logger.warning(f"[Supabase] Batch write to {table} failed: {e}")

    def stop(self):
        self.running = False
        self.flush_all()

class SmartIntersectionController:
    def __init__(self, supabase):
        self.supabase = supabase
        self.template_mgr = TemplatePhaseManager(supabase)
        self.template_mgr.load_templates()
        self.batch_writer = AsyncBatchWriter(supabase)
        self.batch_writer.start()
        self.controllers = {}  # {tl_id: AdaptivePhaseController}
        self.supabase = supabase
        self._control_lock = threading.Lock()  # ✅ Add lock
    def initialize(self):
        """Initialize controllers after TraCI connection is established"""
        self._load_all_controllers()

    def _load_all_controllers(self):
        for tl_id in traci.trafficlight.getIDList():
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            apc = AdaptivePhaseController(
                lane_ids=lanes,
                tl_id=tl_id,
                supabase=self.supabase,
                template_mgr=self.template_mgr,
                batch_writer=self.batch_writer,
            )
            
            # Force adaptive control override
            apc.force_adaptive_control()
            
            self.controllers[tl_id] = apc
            logger.info(f"Initialized controller for {tl_id} with {len(lanes)} lanes")
    def run_all(self):
        if not self.controllers:
            self.initialize()
        for apc in self.controllers.values():
            apc.control_step()

    def shutdown(self):
        for apc in self.controllers.values():
            apc.shutdown()
        self.batch_writer.stop()


class EmergencyCoordinator:
    def __init__(self):
        self.active_emergencies = {}  # {tl_id: {lane_id: start_time}}
        self.emergency_queue = {}     # {tl_id: [(priority, lane_id, start_time)]}
        self.min_emergency_duration = 15
        self.detection_cooldown = {}
        self.last_detection_time = {}
        
    def _calculate_priority(self, lane_id, current_time):
        """Calculate emergency priority based on vehicle type, distance, and waiting time"""
        try:
            priority_score = 0
            
            # Get vehicles on the lane
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            emergency_vehicles = []
            
            for vid in vehicle_ids:
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    if 'emergency' in v_type or 'priority' in v_type:
                        emergency_vehicles.append(vid)
                except traci.TraCIException:
                    continue
            
            if not emergency_vehicles:
                return 0
            
            # Calculate priority for each emergency vehicle
            for vid in emergency_vehicles:
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    
                    # Base priority by vehicle type
                    if 'ambulance' in v_type:
                        type_priority = 100
                    elif 'fire' in v_type:
                        type_priority = 90
                    elif 'police' in v_type:
                        type_priority = 80
                    else:
                        type_priority = 70  # Generic emergency
                    
                    # Distance factor - closer vehicles get higher priority
                    try:
                        lane_length = traci.lane.getLength(lane_id)
                        vehicle_pos = traci.vehicle.getLanePosition(vid)
                        distance_to_end = lane_length - vehicle_pos
                        # Higher priority for vehicles closer to intersection
                        distance_factor = max(0, (50 - distance_to_end) / 50) * 20
                    except:
                        distance_factor = 10  # Default if position can't be determined
                    
                    # Waiting time factor
                    try:
                        waiting_time = traci.vehicle.getWaitingTime(vid)
                        waiting_factor = min(waiting_time * 2, 30)  # Cap at 30
                    except:
                        waiting_factor = 0
                    
                    vehicle_priority = type_priority + distance_factor + waiting_factor
                    priority_score = max(priority_score, vehicle_priority)
                    
                except traci.TraCIException:
                    continue
            
            return priority_score
            
        except Exception as e:
            logger.warning(f"Error calculating emergency priority for {lane_id}: {e}")
            return 50  # Default priority
    
    def get_next_emergency(self, tl_id):
        """Get the next emergency from the queue for the given traffic light"""
        try:
            if tl_id not in self.emergency_queue or not self.emergency_queue[tl_id]:
                return None
            
            # Sort by priority (highest first) and return the lane_id
            self.emergency_queue[tl_id].sort(key=lambda x: x[0], reverse=True)
            
            if self.emergency_queue[tl_id]:
                priority, lane_id, start_time = self.emergency_queue[tl_id].pop(0)
                
                # Add to active emergencies
                if tl_id not in self.active_emergencies:
                    self.active_emergencies[tl_id] = {}
                self.active_emergencies[tl_id][lane_id] = traci.simulation.getTime()
                
                logger.info(f"[EMERGENCY_NEXT] Serving queued emergency on {lane_id} with priority {priority}")
                return lane_id
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting next emergency for {tl_id}: {e}")
            return None
    
    def emergency_completed(self, tl_id, lane_id):
        """Mark emergency as completed and check for next"""
        if tl_id in self.active_emergencies and lane_id in self.active_emergencies[tl_id]:
            del self.active_emergencies[tl_id][lane_id]
            logger.info(f"[EMERGENCY_COMPLETED] Emergency on {lane_id} completed")
        
        # Return next emergency if available
        next_lane = self.get_next_emergency(tl_id)
        if next_lane:
            logger.info(f"[EMERGENCY_NEXT] Next emergency: {next_lane}")
            return next_lane
        return None
    
    def _is_emergency_vehicle_present(self, lane_id):
        """Check if emergency vehicle is present with debouncing"""
        current_time = traci.simulation.getTime()
        
        # Check cooldown
        if lane_id in self.detection_cooldown:
            if current_time - self.detection_cooldown[lane_id] < 2.0:
                return False
        
        emergency_vehicles = []
        try:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for vid in vehicle_ids:
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    if 'emergency' in v_type or 'priority' in v_type:
                        # Check if this is a new detection
                        last_seen = self.last_detection_time.get(vid, 0)
                        if current_time - last_seen > 5.0:
                            emergency_vehicles.append(vid)
                            self.last_detection_time[vid] = current_time
                except traci.TraCIException:
                    continue
        except Exception as e:
            logger.warning(f"Error detecting emergency vehicles on {lane_id}: {e}")
            return False
        
        if emergency_vehicles:
            self.detection_cooldown[lane_id] = current_time
            return True
        return False
    
    def register_emergency(self, tl_id, lane_id, current_time):
        """Register emergency with proper validation"""
        # Validate emergency is actually present
        if not self._is_emergency_vehicle_present(lane_id):
            return False
            
        if tl_id not in self.active_emergencies:
            self.active_emergencies[tl_id] = {}
            self.emergency_queue[tl_id] = []
        
        # Check if this lane already has an active emergency
        if lane_id in self.active_emergencies[tl_id]:
            elapsed = current_time - self.active_emergencies[tl_id][lane_id]
            if elapsed < self.min_emergency_duration:
                logger.debug(f"[EMERGENCY_DUPLICATE] {lane_id} already being served")
                return False
        
        # Check if another emergency is being served
        for active_lane, start_time in self.active_emergencies[tl_id].items():
            elapsed = current_time - start_time
            if elapsed < self.min_emergency_duration:
                priority = self._calculate_priority(lane_id, current_time)
                self.emergency_queue[tl_id].append((priority, lane_id, current_time))
                logger.info(f"[EMERGENCY_QUEUE] {lane_id} queued (active: {active_lane})")
                return False
        
        # Serve immediately
        self.active_emergencies[tl_id][lane_id] = current_time
        logger.info(f"[EMERGENCY_SERVE] {lane_id} served immediately")
        return True
class AdaptivePhaseController:
    def __init__(self, lane_ids, tl_id, supabase, template_mgr=None, batch_writer=None,
                 alpha=1.0, min_green=30, max_green=80, r_base=0.5, r_adjust=0.1,
                 severe_congestion_threshold=0.8, large_delta_t=20):
        self.lane_ids = lane_ids
        self.tl_id = tl_id
        self.alpha = alpha
        self.min_green = min_green
        self.max_green = min(max_green, 60)
        self.supabase = supabase
        self.template_mgr = template_mgr
        self.batch_writer = batch_writer
        self.logger = logger
        self.emergency_coordinator = EmergencyCoordinator()
        self.phase_repeat_counter = defaultdict(int)
        self.last_served_time = defaultdict(lambda: 0)
        self.severe_congestion_global_cooldown_time = 5
        self.severe_congestion_threshold = severe_congestion_threshold
        self.large_delta_t = large_delta_t
        self.weights = np.array([0.4, 0.2, 0.2, 0.2])
        self.weight_history = []
        self.metric_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=50)
        self.R_target = r_base
        self.r_base = r_base
        self.r_adjust = r_adjust
        self.phase_count = 0
        self.rl_agent = None
        self._scheduled_phase_updates = {}
        self.last_phase_idx = None
        self.last_phase_switch_sim_time = 0
        self.last_extended_time = 0
        self.protected_left_cooldown = defaultdict(float)
        self.severe_congestion_cooldown = {}
        self.severe_congestion_global_cooldown = 0
        self.last_emergency_event = {}
        self.emergency_cooldown = {}
        self.emergency_global_cooldown = 0
        self.last_bonus = 0
        self.last_penalty = 0
        self._local_state = {"events": [], "phases": []}
        self._db_lock = threading.Lock()
        
        # Emergency handling state tracking
        self._emergency_processed = {}  # Track processed emergencies to prevent duplicates
        self._last_emergency_check = 0  # Throttle emergency checks
        self._emergency_completions = {}  # Track emergency completion times
        
        self.load_state_from_supabase()
        self.load_static_config_from_templates()
        self.prev_state = None
        self.prev_action = None  
  
    def flush_pending_supabase_writes(self):
        with self._db_lock:
            if hasattr(self, '_pending_db_ops') and self._pending_db_ops:
                state_json = json.dumps(self._pending_db_ops[-1])
                try:
                    response = self.supabase.table("apc_states").update({
                        "data": state_json,
                        "updated_at": datetime.datetime.now().isoformat()
                    }).eq("tl_id", self.tl_id).eq("state_type", "full").execute()
                    if not response.data:
                        self.supabase.table("apc_states").insert({
                            "tl_id": self.tl_id,
                            "state_type": "full",
                            "data": state_json,
                            "updated_at": datetime.datetime.now().isoformat()
                        }).execute()
                    self._pending_db_ops.clear()
                except Exception as e:
                    print(f"[Supabase] Write failed: {e}")    
    def load_state_from_supabase(self):
        """Load controller state from Supabase database"""
        try:
            response = self.supabase.table("apc_states").select("data").eq("tl_id", self.tl_id).eq("state_type", "full").execute()
            if response.data and len(response.data) > 0:
                self._local_state = json.loads(response.data[0]["data"])
                logger.info(f"[Supabase] Loaded state for {self.tl_id}")
            else:
                self._local_state = {"events": [], "phases": []}
                logger.info(f"[Supabase] No existing state for {self.tl_id}, starting fresh")
        except Exception as e:
            logger.error(f"[Supabase] Failed to load state for {self.tl_id}: {e}")
            self._local_state = {"events": [], "phases": []}
    def save_state_to_supabase(self):
        """Save controller state to Supabase database"""
        try:
            self._local_state["queues"] = self._local_state.get("queues", [])
            self._local_state["waits"] = self._local_state.get("waits", [])
            self._local_state["speeds"] = self._local_state.get("speeds", [])
            self._local_state["densities"] = self._local_state.get("densities", [])
            self._local_state["emergency_lanes"] = self._local_state.get("emergency_lanes", [])
            self._local_state["blocked_lanes"] = self._local_state.get("blocked_lanes", [])
            self._local_state["phase_id"] = self._local_state.get("phase_id", 0)
            self._local_state["phase_duration"] = self._local_state.get("phase_duration", self.min_green)

            payload = {
                "tl_id": self.tl_id,
                "state_type": "full",
                "queues": self._local_state["queues"],
                "waits": self._local_state["waits"],
                "speeds": self._local_state["speeds"],
                "densities": self._local_state["densities"],
                "emergency_lanes": self._local_state["emergency_lanes"],
                "blocked_lanes": self._local_state["blocked_lanes"],
                "phase_id": self._local_state["phase_id"],
                "phase_duration": self._local_state["phase_duration"],
                "data": json.dumps(to_builtin_type(self._local_state)),
                "updated_at": datetime.datetime.now().isoformat()
            }

            if self.batch_writer:
                self.batch_writer.buffer("apc_states", payload)
            else:
                self.supabase.table("apc_states").upsert(payload).execute()
        except Exception as e:
            logger.error(f"[Supabase] Failed to save state for {self.tl_id}: {e}")


    def load_phase_from_supabase(self, phase_idx):
        """Load specific phase data from Supabase state"""
        for phase in self._local_state.get("phases", []):
            if phase["phase_idx"] == phase_idx:
                return phase
        return None
    def _save_apc_state_supabase(self):
        if hasattr(self, '_pending_db_ops'):
            self._pending_db_ops.append(self._local_state.copy())
    def log_rl_decision(self, phase_idx, action, reward):
        if self.batch_writer:
            self.batch_writer.buffer("rl_decisions", {
                "tl_id": self.tl_id,
                "sim_time": traci.simulation.getTime(),
                "phase_idx": phase_idx,
                "chosen_action": action,
                "reward": reward,
            })

    # --- Template and Phase Caching ---

    def load_static_config_from_templates(self):
        if self.template_mgr:
            self.template_mgr.refresh_if_needed()
            self.phase_templates = self.template_mgr.get_template(self.tl_id) or self.template_mgr.get_template("default")

    def save_state_async(self):
        if self.batch_writer:
            self.batch_writer.buffer("apc_states", {
                "tl_id": self.tl_id,
                "state_type": "full",
                "data": json.dumps(to_builtin_type(self._local_state)),
                "updated_at": datetime.datetime.now().isoformat()
            })
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

    def _load_apc_state_supabase(self):
        response = supabase.table("apc_states").select("data").eq("tl_id", self.tl_id).eq("state_type", "full").execute()
        if response.data and len(response.data) > 0:
            self._local_state = json.loads(response.data[0]["data"])
            # Fix any oversized durations in loaded state
            for phase in self._local_state.get("phases", []):
                if phase["duration"] > MAX_PHASE_DURATION:
                    print(f"[CLEANUP] Phase {phase['phase_idx']} had duration {phase['duration']} > cap, setting to {MAX_PHASE_DURATION}")
                    phase["duration"] = MAX_PHASE_DURATION
        else:
            self._local_state = {"events": [], "phases": []}

    def save_phase_to_supabase(self, phase_idx, duration, state_str, delta_t, raw_delta_t, penalty,
                            reward=None, bonus=None, weights=None, event_type=None, lanes=None):
        """Save phase data to Supabase state"""
        duration = min(duration, MAX_PHASE_DURATION)
        
        entry = {
            "phase_idx": phase_idx,
            "duration": duration,
            "state": state_str,
            "delta_t": delta_t,
            "raw_delta_t": raw_delta_t,
            "penalty": penalty,
            "timestamp": datetime.datetime.now().isoformat(),
            "sim_time": traci.simulation.getTime(),
            "reward": reward if reward is not None else getattr(self, "last_R", None),
            "bonus": bonus if bonus is not None else self.last_bonus,
            "weights": weights if weights is not None else self.weights.tolist(),
            "event_type": event_type,
            "lanes": lanes if lanes is not None else self.lane_ids[:]
        }
        
        # Update or add phase
        phases = self._local_state.setdefault("phases", [])
        found = False
        for i, phase in enumerate(phases):
            if phase["phase_idx"] == phase_idx:
                entry["base_duration"] = phase.get("base_duration", entry["duration"])
                phases[i] = entry
                found = True
                break
        
        if not found:
            entry["base_duration"] = entry["duration"]
            phases.append(entry)
        
        self.save_state_to_supabase()    
    def create_or_extend_phase(self, green_lanes, delta_t):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        valid_green_lanes = [lane for lane in green_lanes if lane in controlled_lanes]
        if not valid_green_lanes:
            return None

        # Handle protected left turns
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
                
                duration_cap = min(self.max_green * 1.5 if delta_t > 10 else self.max_green, MAX_PHASE_DURATION)
                duration = np.clip(base_duration + delta_t, self.min_green, duration_cap)
                
                if phase_idx is not None:
                    self.save_phase_to_supabase(phase_idx, duration, protected_state, delta_t, delta_t, penalty=0)
                    logger.info(f"[PROTECTED_LEFT] Extended phase {phase_idx} to {duration}s for lane {lane}")
                    return phase_idx
                
                # Create new protected left phase
                new_phase = traci.trafficlight.Phase(duration, protected_state)
                phases = list(logic.getPhases()) + [new_phase]
                new_logic = traci.trafficlight.Logic(
                    logic.programID, logic.type, len(phases) - 1, phases
                )
                traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
                traci.trafficlight.setPhase(self.tl_id, len(phases) - 1)
                
                phase_idx = len(phases) - 1
                self.save_phase_to_supabase(phase_idx, duration, protected_state, delta_t, delta_t, penalty=0)
                logger.info(f"[PROTECTED_LEFT] Created new phase {phase_idx} with {duration}s for lane {lane}")
                return phase_idx

        # Handle regular phases
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
            logger.info(f"[PHASE_EXTEND] Extended phase {phase_idx} to {duration}s for lanes {valid_green_lanes}")
            return phase_idx
        
        # Create new phase
        new_phase = traci.trafficlight.Phase(duration, new_state)
        phases = list(logic.getPhases()) + [new_phase]
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type, len(phases) - 1, phases
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
        traci.trafficlight.setPhase(self.tl_id, len(phases) - 1)
        
        phase_idx = len(phases) - 1
        self.save_phase_to_supabase(phase_idx, duration, new_state, delta_t, delta_t, penalty=0)
        logger.info(f"[PHASE_CREATE] Created new phase {phase_idx} with {duration}s for lanes {valid_green_lanes}")
        return phase_idx   
    def log_apc_event(self, event):
        """Log APC event to Supabase state"""
        event["timestamp"] = datetime.datetime.now().isoformat()
        event["sim_time"] = traci.simulation.getTime()
        event["tl_id"] = self.tl_id
        event["weights"] = self.weights.tolist()
        event["bonus"] = self.last_bonus
        event["penalty"] = self.last_penalty
        
        self._local_state["events"].append(event)
        
        # ✅ Limit event history
        max_events = 1000
        if len(self._local_state["events"]) > max_events:
            self._local_state["events"] = self._local_state["events"][-max_events//2:]
        
        self.save_state_to_supabase()

    def get_phase_history_from_supabase(self):
        """Get phase history from Supabase for analysis"""
        try:
            # FIXED column name from tl_id to tl_id
            response = self.supabase.table("apc_states").select("data").eq("tl_id", self.tl_id).order("updated_at", desc=True).limit(10).execute()
            history = []
            for record in response.data:
                state = json.loads(record["data"])
                history.append(state)
            return history
        except Exception as e:
            print(f"[Supabase] Failed to get phase history: {e}")
            return []

    def cleanup_old_supabase_records(self, keep_days=7):
        """Clean up old Supabase records to prevent database bloat"""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
            # FIXED column name from tl_id to tl_id
            self.supabase.table("apc_states").delete().eq("tl_id", self.tl_id).lt("updated_at", cutoff_date.isoformat()).execute()
            print(f"[Supabase] Cleaned up old records for {self.tl_id}")
        except Exception as e:
            print(f"[Supabase] Failed to cleanup old records: {e}")
    def update_phase_duration_in_supabase(self, phase_idx, new_duration):
        """Update phase duration in Supabase state"""
        new_duration = min(new_duration, MAX_PHASE_DURATION)
        
        phases = self._local_state.get("phases", [])
        for phase in phases:
            if phase["phase_idx"] == phase_idx:
                phase["duration"] = new_duration
                phase["updated_at"] = datetime.datetime.now().isoformat()
                break
        
        self.save_state_to_supabase()
        
        # Log the update
        self.log_apc_event({
            "action": "phase_duration_update",
            "phase_idx": phase_idx,
            "duration": new_duration,
            "tl_id": self.tl_id
        })
    def ensure_supabase_program_active(self):
        current_program = traci.trafficlight.getProgram(self.tl_id)
        if current_program != "SUPABASE-OVERRIDE":
            logger.info(f"[PATCH] Switching {self.tl_id} to SUPABASE-OVERRIDE program.")
            traci.trafficlight.setProgram(self.tl_id, "SUPABASE-OVERRIDE")
    def _override_sumo_logic_from_supabase(self, current_phase_idx=None):
        phase_seq = []
        for rec in sorted(self._local_state.get("phases", []), key=lambda x: x["phase_idx"]):
            phase_seq.append((rec["state"], rec["duration"]))

        if not phase_seq:
            logger.warning(f"[PATCH] No Supabase phase sequence for {self.tl_id}, skipping override.")
            return

        if current_phase_idx is None or not (0 <= current_phase_idx < len(phase_seq)):
            current_phase_idx = 0

        phases = [Phase(duration, state) for (state, duration) in phase_seq]
        new_logic = Logic(
            programID="SUPABASE-OVERRIDE",
            type=traci.constants.TL_LOGIC_PROGRAM,
            currentPhaseIndex=current_phase_idx,
            phases=phases,
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
        traci.trafficlight.setProgram(self.tl_id, "SUPABASE-OVERRIDE")  # <---- CRITICAL PATCH
        logger.info(f"[PATCH] Overrode SUMO program for {self.tl_id} with {len(phases)} Supabase phases.")
    def set_phase_from_supabase(self, phase_idx, requested_duration=None):
        logger.debug(f"[Supabase] Setting phase {phase_idx} for {self.tl_id}")

        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
        phases = logic.getPhases()

        # --- PATCH: Always check phase index before setting ---
        if not (0 <= phase_idx < len(phases)):
            logger.error(f"[ERROR] Phase index {phase_idx} is out of range (0-{len(phases)-1}) for {self.tl_id}")
            # Try to recover by overriding SUMO's logic from Supabase/PKL if possible
            if self._local_state.get("phases"):
                logger.warning(f"[PATCH] Attempting to override SUMO logic from Supabase for {self.tl_id}")
                self._override_sumo_logic_from_supabase(current_phase_idx=phase_idx)
                # Try again after override
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
                phases = logic.getPhases()
                if not (0 <= phase_idx < len(phases)):
                    logger.error(f"[PATCH] After override, phase index {phase_idx} still invalid.")
                    return False
            else:
                logger.error(f"[PATCH] No Supabase state available to override logic for {self.tl_id}")
                return False

        phase_record = self.load_phase_from_supabase(phase_idx)
        duration = requested_duration if requested_duration is not None else (
            phase_record["duration"] if phase_record else phases[phase_idx].duration
        )
        duration = min(duration, MAX_PHASE_DURATION)

        try:
            traci.trafficlight.setPhase(self.tl_id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.tl_id, duration)
            logger.debug(f"[Supabase] Successfully set phase {phase_idx} for {self.tl_id} with duration {duration}s")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to set phase {phase_idx}: {e}")
            return False
    def calculate_delta_t_and_penalty(self, R):
        raw_delta_t = self.alpha * (R - self.R_target)
        penalty = max(0, abs(raw_delta_t) - 10)
        ext_t = 20 * np.tanh(raw_delta_t / 20)
        delta_t = np.clip(ext_t, -20, 20)
        return raw_delta_t, delta_t, penalty
    def get_full_phase_sequence(self):
        phase_records = sorted(self._local_state.get("phases", []), key=lambda x: x["phase_idx"])
        if not phase_records:
            return [(p.state, p.duration) for p in self._phase_defs]
        return [(rec["state"], rec["duration"]) for rec in phase_records]


    def _log_apc_event(self, event):
        event["timestamp"] = datetime.datetime.now().isoformat()
        event["sim_time"] = traci.simulation.getTime()
        event["tl_id"] = self.tl_id
        event["weights"] = self.weights.tolist()
        event["bonus"] = getattr(self, "last_bonus", 0)
        event["penalty"] = getattr(self, "last_penalty", 0)
        self._local_state["events"].append(event)
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
        """Gracefully shutdown and save final state to Supabase"""
        try:
            self.save_state_to_supabase()
            logger.info(f"[Supabase] Controller {self.tl_id} shutdown complete")
        except Exception as e:
            logger.error(f"[ERROR] Shutdown failed for {self.tl_id}: {e}")    
    def get_lane_stats(self, lane_id):
        try:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            waiting_time = traci.lane.getWaitingTime(lane_id)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            length = max(1.0, traci.lane.getLength(lane_id))
            density = traci.lane.getLastStepVehicleNumber(lane_id) / length
            return queue, waiting_time, mean_speed, density
        except Exception:
            return 0, 0, 0, 0
        
    def create_or_extend_phase_with_template(self, green_lanes, delta_t, event_type="congestion"):
        """Enhanced phase creation using templates"""
        # Try to use template first
        if self.template_mgr and green_lanes:
            template = self.template_mgr.apply_template_to_phase(self.tl_id, green_lanes[0], event_type)
            if template:
                logger.info(f"[TEMPLATE] Using template for {event_type} on {green_lanes}")
                return self.create_phase_from_template(template, green_lanes, delta_t)
        
        # Fallback to existing logic
        return self.create_or_extend_phase(green_lanes, delta_t)    
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
    def create_phase_from_template(self, template, green_lanes, delta_t):
        """Create phase using template specifications"""
        try:
            base_duration = template.get("duration", self.min_green)
            min_duration = template.get("min_duration", self.min_green)
            max_duration = template.get("max_duration", self.max_green)
            
            # Calculate template-enhanced duration
            duration = np.clip(base_duration + delta_t, min_duration, max_duration)
            
            # Create phase state
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            new_state = self.create_phase_state(green_lanes=green_lanes)
            
            # Check if phase already exists
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            for idx, phase in enumerate(logic.getPhases()):
                if phase.state == new_state:
                    # Update existing phase with template duration
                    self.save_phase_to_supabase(idx, duration, new_state, delta_t, delta_t, penalty=0,
                                              event_type=template.get("name", "template"))
                    logger.info(f"[TEMPLATE] Updated phase {idx} with template duration {duration}s")
                    return idx
            
            # Create new phase with template
            new_phase = traci.trafficlight.Phase(duration, new_state)
            phases = list(logic.getPhases()) + [new_phase]
            new_logic = traci.trafficlight.Logic(
                logic.programID, logic.type, len(phases) - 1, phases
            )
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
            traci.trafficlight.setPhase(self.tl_id, len(phases) - 1)
            
            phase_idx = len(phases) - 1
            self.save_phase_to_supabase(phase_idx, duration, new_state, delta_t, delta_t, penalty=0,
                                      event_type=template.get("name", "template"))
            
            logger.info(f"[TEMPLATE] Created new phase {phase_idx} using template '{template.get('name')}' with duration {duration}s")
            return phase_idx
            
        except Exception as e:
            logger.error(f"[TEMPLATE] Failed to create phase from template: {e}")
            return None
    
    def log_phase_switch(self, new_phase_idx):
        current_time = traci.simulation.getTime()
        elapsed = current_time - self.last_phase_switch_sim_time

        # Emergency vehicles override min green
        event_type, _ = self.check_special_events()
        is_emergency = event_type == "emergency_vehicle"
        
        # Block switch if min green not met and no emergency
        if elapsed < self.min_green and not is_emergency:
            logger.debug(f"[MIN_GREEN BLOCK] Phase switch blocked (elapsed: {elapsed:.1f}s < {self.min_green}s)")
            return False

        # Flicker prevention
        if self.last_phase_idx == new_phase_idx and not is_emergency:
            logger.debug(f"[PHASE SWITCH BLOCKED] Flicker prevention triggered for {self.tl_id}")
            return False

        try:
            traci.trafficlight.setPhase(self.tl_id, new_phase_idx)
            traci.trafficlight.setPhaseDuration(self.tl_id, max(self.min_green, self.max_green))
            new_state = traci.trafficlight.getRedYellowGreenState(self.tl_id)
            logger.info(f"Phase {new_phase_idx} state: {new_state}")
            self.last_phase_idx = new_phase_idx
            self.last_phase_switch_sim_time = current_time

            event = {
                "action": "phase_switch",
                "old_phase": self.last_phase_idx,
                "new_phase": new_phase_idx,
                "new_state": new_state,
                "emergency_override": is_emergency,
                "elapsed_since_last": elapsed
            }
            self.log_apc_event(event)
            
            status = "[EMERGENCY_SWITCH]" if is_emergency else "[PHASE_SWITCH]"
            logger.info(f"{status} {self.tl_id}: {self.last_phase_idx}→{new_phase_idx} (elapsed: {elapsed:.1f}s)")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Phase switch failed: {e}")
            return False
    def safe_phase_transition(self, tl_id, current_phase, target_phase, emergency=False):
        current_time = traci.simulation.getTime()
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
        if current_phase == target_phase:
            return True
        
        needs_yellow = self._check_yellow_needed(logic.getPhases()[current_phase].state, 
                                                logic.getPhases()[target_phase].state)
        if needs_yellow and not emergency:
            yellow_phase_idx = self._get_or_create_yellow_phase(logic, 
                                                            logic.getPhases()[current_phase].state, 
                                                            logic.getPhases()[target_phase].state)
            traci.trafficlight.setPhase(tl_id, yellow_phase_idx)
            traci.trafficlight.setPhaseDuration(tl_id, 4)
            self._schedule_delayed_phase_switch(tl_id, target_phase, current_time + 4)
            return True
        
        traci.trafficlight.setPhase(tl_id, target_phase)
        if emergency:
            self._schedule_delayed_phase_switch(tl_id, self.get_next_normal_phase(), current_time + 45)
        return True

    def get_next_normal_phase(self):
        return 0
    def _check_yellow_needed(self, current_state, target_state):
        """Check if any lane goes from green to red"""
        for i in range(min(len(current_state), len(target_state))):
            if current_state[i].upper() == 'G' and target_state[i].upper() == 'R':
                return True
        return False

    def _get_or_create_yellow_phase(self, logic, current_state, target_state):
        """Get existing or create new yellow transition phase"""
        yellow_state = list(current_state)
        for i in range(min(len(current_state), len(target_state))):
            if current_state[i].upper() == 'G' and target_state[i].upper() == 'R':
                yellow_state[i] = 'y'
        
        yellow_state_str = ''.join(yellow_state)
        
        # Check if this yellow phase exists
        for idx, phase in enumerate(logic.getPhases()):
            if phase.state == yellow_state_str:
                return idx
        
        # Create new yellow phase
        new_phase = traci.trafficlight.Phase(4, yellow_state_str)
        phases = list(logic.getPhases()) + [new_phase]
        new_logic = traci.trafficlight.Logic(logic.programID, logic.type, len(phases)-1, phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
        
        return len(phases) - 1
    def check_priority_conditions(self):
        event_type, event_lane = self.check_special_events()
        if event_type == "emergency_vehicle":
            return True
        return False

    def create_phase_state(self, green_lanes=None, yellow_lanes=None, red_lanes=None):
        """Enhanced phase state creation with validation"""
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            state = ['r'] * len(controlled_lanes)
            
            def set_lanes_safe(lanes, state_char):
                if not lanes:
                    return
                for lane in lanes:
                    if lane in controlled_lanes:
                        idx = controlled_lanes.index(lane)
                        if 0 <= idx < len(state):
                            state[idx] = state_char
                    else:
                        logger.warning(f"Lane {lane} not controlled by {self.tl_id}")
            
            set_lanes_safe(green_lanes, 'G')
            set_lanes_safe(yellow_lanes, 'y')
            set_lanes_safe(red_lanes, 'r')
            
            state_str = "".join(state)
            
            # Validate state
            if not self._validate_phase_state(state_str, green_lanes):
                logger.error(f"Invalid phase state created: {state_str} for greens: {green_lanes}")
                # Create fallback safe state
                state_str = 'r' * len(controlled_lanes)
                if green_lanes and green_lanes[0] in controlled_lanes:
                    idx = controlled_lanes.index(green_lanes[0])
                    state = list(state_str)
                    state[idx] = 'G'
                    state_str = "".join(state)
            
            logger.debug(f"Created phase state: {state_str} for greens: {green_lanes}")
            return state_str
            
        except Exception as e:
            logger.error(f"Error creating phase state: {e}")
            # Return safe default state
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            return 'r' * len(controlled_lanes)
    
    def add_new_phase(self, green_lanes, green_duration=20, yellow_duration=3, yellow_lanes=None):
        print(f"[DEBUG] add_new_phase called with green_lanes={green_lanes}, green_duration={green_duration}, yellow_duration={yellow_duration}, yellow_lanes={yellow_lanes}")
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
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
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
            print(f"[DEBUG] setCompleteRedYellowGreenDefinition called for {self.tl_id} with {len(phases)} phases")
            traci.trafficlight.setPhase(self.tl_id, len(phases)-2)
            print(f"[DEBUG] setPhase called for {self.tl_id} to phase {len(phases)-2}")
            event = {
                "action": "add_new_phase",
                "tl_id": self.tl_id,
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
        """Enhanced emergency detection across ALL controlled lanes"""
        now = traci.simulation.getTime()
        
        # Throttle checking to every 1 second
        if hasattr(self, "_last_special_check") and now - self._last_special_check < 1:
            return None, None
        self._last_special_check = now
        
        # Check ALL controlled lanes, not just self.lane_ids
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            logger.debug(f"[EMERGENCY_CHECK] Checking {len(controlled_lanes)} controlled lanes for {self.tl_id}")
            
            for lane_id in controlled_lanes:
                try:
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    for vid in vehicle_ids:
                        try:
                            v_type = traci.vehicle.getTypeID(vid)
                            # More comprehensive emergency vehicle detection
                            if any(keyword in v_type.lower() for keyword in ['emergency', 'priority', 'ambulance', 'fire', 'police']):
                                logger.info(f"[EMERGENCY_DETECTED] Found {v_type} vehicle {vid} on lane {lane_id}")
                                return 'emergency_vehicle', lane_id
                        except traci.TraCIException:
                            continue
                except traci.TraCIException:
                    continue
                    
        except Exception as e:
            logger.error(f"Error in emergency detection: {e}")
        
        return None, None   
    def calculate_dynamic_phase_duration(self, phase_idx, base_duration, delta_t):
        """Calculate enhanced phase duration based on current traffic conditions"""
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            if phase_idx >= len(logic.getPhases()):
                return base_duration
                
            phase_state = logic.getPhases()[phase_idx].state
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            
            total_queue = 0
            total_wait = 0
            green_lanes = 0
            
            for lane_idx, lane_id in enumerate(controlled_lanes):
                if lane_idx < len(phase_state) and phase_state[lane_idx].upper() == 'G':
                    queue, wait, _, _ = self.get_lane_stats(lane_id)
                    total_queue += queue
                    total_wait += wait
                    green_lanes += 1
            
            if green_lanes == 0:
                return base_duration
            
            avg_queue = total_queue / green_lanes
            avg_wait = total_wait / green_lanes
            
            queue_factor = min(avg_queue / 10.0, 2.0)
            wait_factor = min(avg_wait / 60.0, 1.5)
            
            enhancement = 1.0 + queue_factor + wait_factor
            enhanced_duration = min(base_duration * enhancement + delta_t, MAX_PHASE_DURATION)
            
            logger.debug(f"[DYNAMIC_DURATION] Phase {phase_idx}: base={base_duration}s, enhanced={enhanced_duration:.1f}s")
            return enhanced_duration
            
        except Exception as e:
            logger.error(f"[ERROR] calculate_dynamic_phase_duration: {e}")
            return base_duration


    def reorganize_or_create_phase(self, lane_id, event_type):
        """Simplified emergency handling with better error recovery"""
        try:
            current_time = traci.simulation.getTime()
            
            if event_type == 'emergency_vehicle':
                logger.info(f"[EMERGENCY_OVERRIDE] Processing emergency for {lane_id}")
                
                # Check current SUMO state first
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
                logger.info(f"[EMERGENCY_DEBUG] SUMO currently has {len(logic.getPhases())} phases")
                
                # Create or find emergency phase
                phase_idx = self._create_validated_emergency_phase(lane_id)
                if phase_idx is None:
                    logger.error(f"[EMERGENCY_FAILED] Could not create emergency phase for {lane_id}")
                    # Try fallback: use an existing phase that gives green to this lane
                    return self._emergency_fallback(lane_id)
                
                # Apply emergency phase
                success = self._apply_emergency_phase_safely(phase_idx, lane_id)
                if success:
                    logger.info(f"[EMERGENCY_SUCCESS] Emergency phase {phase_idx} active for {lane_id}")
                    # Schedule completion check
                    self._schedule_emergency_completion(lane_id, current_time + 45)
                    return True
                else:
                    logger.error(f"[EMERGENCY_FAILED] Could not apply emergency phase {phase_idx}")
                    return self._emergency_fallback(lane_id)
            
            return False
            
        except Exception as e:
            logger.error(f"[ERROR] Emergency handling failed: {e}")
            traceback.print_exc()
            return False   
    def _emergency_fallback(self, emergency_lane):
        """Fallback emergency handling when normal creation fails"""
        try:
            logger.info(f"[EMERGENCY_FALLBACK] Attempting fallback for {emergency_lane}")
            
            # Get current SUMO phases
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            
            if emergency_lane not in controlled_lanes:
                logger.error(f"[EMERGENCY_FALLBACK] Lane {emergency_lane} not controlled")
                return False
            
            emergency_idx = controlled_lanes.index(emergency_lane)
            
            # Find any existing phase that gives green to this lane
            for phase_idx, phase in enumerate(logic.getPhases()):
                if (emergency_idx < len(phase.state) and 
                    phase.state[emergency_idx].upper() == 'G'):
                    
                    logger.info(f"[EMERGENCY_FALLBACK] Using existing phase {phase_idx}")
                    
                    # Apply this phase
                    traci.trafficlight.setPhase(self.tl_id, phase_idx)
                    traci.trafficlight.setPhaseDuration(self.tl_id, 45)
                    
                    self.last_phase_idx = phase_idx
                    self.last_phase_switch_sim_time = traci.simulation.getTime()
                    
                    return True
            
            logger.error(f"[EMERGENCY_FALLBACK] No suitable phase found for {emergency_lane}")
            return False
            
        except Exception as e:
            logger.error(f"[EMERGENCY_FALLBACK] Failed: {e}")
            return False
    def _validate_phase_state(self, state_str, green_lanes):
        """
        Validation for phase state string.
        - Ensures at least one green, no conflicting greens, and green_lanes (if provided) get green.
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        if len(state_str) != len(controlled_lanes):
            logger.error(f"[PHASE VALIDATION] State length mismatch: {len(state_str)} vs {len(controlled_lanes)}")
            return False

        # At least one green
        if 'G' not in state_str:
            logger.error("[PHASE VALIDATION] No green in phase state.")
            return False

        # All green lanes must be in green_lanes (if provided)
        if green_lanes:
            for lane in green_lanes:
                if lane in controlled_lanes:
                    idx = controlled_lanes.index(lane)
                    if state_str[idx] != 'G':
                        logger.error(f"[PHASE VALIDATION] Lane {lane} not green in state!")
                        return False

        # (Optional) Add more checks for conflicting greens (e.g., protected left vs straight)
        # For now, accept any single-green phase as valid.
        return True
    def _create_validated_emergency_phase(self, emergency_lane):
        """Create emergency phase with comprehensive validation and proper SUMO sync"""
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            if emergency_lane not in controlled_lanes:
                logger.error(f"Emergency lane {emergency_lane} not controlled by {self.tl_id}")
                logger.info(f"Controlled lanes: {controlled_lanes}")
                return None

            # Create emergency state (green for emergency lane, red for others)
            emergency_state = self.create_phase_state(green_lanes=[emergency_lane])
            logger.info(f"[EMERGENCY_PHASE] Created state: {emergency_state} for lane {emergency_lane}")

            # Validate state
            if not self._validate_phase_state(emergency_state, [emergency_lane]):
                logger.error(f"Failed to create valid emergency state for {emergency_lane}")
                return None

            # Get CURRENT SUMO logic (this is critical!)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            logger.info(f"[EMERGENCY_PHASE] Current SUMO has {len(logic.getPhases())} phases")
            
            # Check if emergency phase already exists IN CURRENT SUMO LOGIC
            for idx, phase in enumerate(logic.getPhases()):
                if phase.state == emergency_state:
                    logger.info(f"[EMERGENCY_PHASE] Found existing SUMO phase {idx} with matching state")
                    # Verify this phase index is actually valid
                    if idx < len(logic.getPhases()):
                        return idx
                    else:
                        logger.error(f"Phase index {idx} is out of range for current logic")
                        break

            # Create new emergency phase
            emergency_duration = min(45, MAX_PHASE_DURATION)
            new_phase = traci.trafficlight.Phase(emergency_duration, emergency_state)
            
            # Get fresh phase list and add new phase
            current_phases = list(logic.getPhases())
            current_phases.append(new_phase)
            
            logger.info(f"[EMERGENCY_PHASE] Adding new phase. Old count: {len(logic.getPhases())}, New count: {len(current_phases)}")

            # Create and apply new logic to SUMO
            new_logic = traci.trafficlight.Logic(
                logic.programID, logic.type, 0, current_phases
            )
            
            # Apply to SUMO first
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
            
            # Verify the update worked
            updated_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            new_phase_idx = len(current_phases) - 1
            
            if new_phase_idx >= len(updated_logic.getPhases()):
                logger.error(f"Failed to add phase to SUMO. Expected {new_phase_idx}, SUMO has {len(updated_logic.getPhases())} phases")
                return None
            
            # Verify the state matches
            actual_state = updated_logic.getPhases()[new_phase_idx].state
            if actual_state != emergency_state:
                logger.error(f"Phase state mismatch. Expected: {emergency_state}, Got: {actual_state}")
                return None
            
            logger.info(f"[EMERGENCY_PHASE] Successfully created phase {new_phase_idx} in SUMO")
            
            # Save to local state AFTER successful SUMO update
            self.save_phase_to_supabase(
                new_phase_idx, emergency_duration, emergency_state, 0, 0, 0,
                event_type="emergency", lanes=[emergency_lane]
            )

            return new_phase_idx

        except Exception as e:
            logger.error(f"Error creating emergency phase: {e}")
            traceback.print_exc()
            return None   
   
    def _is_emergency_vehicle_present(self, lane_id):
        """Check if emergency vehicle is present with improved detection"""
        current_time = traci.simulation.getTime()
        
        # Reduce cooldown to 0.5 seconds (was 2.0)
        if lane_id in self.detection_cooldown:
            if current_time - self.detection_cooldown[lane_id] < 0.5:
                return False
        
        try:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            emergency_found = False
            
            for vid in vehicle_ids:
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    if 'emergency' in v_type.lower() or 'priority' in v_type.lower():
                        # Remove the "new detection" check - it's causing false negatives
                        emergency_found = True
                        self.last_detection_time[vid] = current_time
                        break
                except traci.TraCIException:
                    continue
                    
            if emergency_found:
                self.detection_cooldown[lane_id] = current_time
                return True
            else:
                # Clear cooldown if no emergency vehicles found
                if lane_id in self.detection_cooldown:
                    del self.detection_cooldown[lane_id]
                return False
                
        except Exception as e:
            logger.warning(f"Error detecting emergency vehicles on {lane_id}: {e}")
            return False
    def _apply_emergency_phase_safely(self, phase_idx, emergency_lane):
        try:
            self.ensure_supabase_program_active() # <-- Add this line
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            if phase_idx >= len(logic.getPhases()):
                logger.error(f"Phase index {phase_idx} out of range [0,{len(logic.getPhases())-1}] for {self.tl_id}")
                return False
            
            logger.info(f"[EMERGENCY_APPLY] Applying phase {phase_idx}/{len(logic.getPhases())-1} for {emergency_lane}")
            
            # Apply phase immediately (emergency override)
            traci.trafficlight.setPhase(self.tl_id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.tl_id, 45)
            
            # Verify application
            applied_phase = traci.trafficlight.getPhase(self.tl_id)
            if applied_phase != phase_idx:
                logger.error(f"Failed to apply phase {phase_idx}, SUMO shows {applied_phase}")
                return False
            
            # Verify state gives green to emergency lane
            current_state = traci.trafficlight.getRedYellowGreenState(self.tl_id)
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            
            if emergency_lane in controlled_lanes:
                emergency_idx = controlled_lanes.index(emergency_lane)
                if emergency_idx >= len(current_state) or current_state[emergency_idx] != 'G':
                    logger.error(f"Emergency lane {emergency_lane} not green! State: {current_state}")
                    return False
            
            # Update tracking
            self.last_phase_idx = phase_idx
            self.last_phase_switch_sim_time = traci.simulation.getTime()
            
            # Log event
            self.log_apc_event({
                "action": "emergency_phase_applied",
                "phase_idx": phase_idx,
                "emergency_lane": emergency_lane,
                "state": current_state,
                "duration": 45,
                "total_phases": len(logic.getPhases())
            })
            
            logger.info(f"[EMERGENCY_SUCCESS] Applied phase {phase_idx} for {emergency_lane}, state: {current_state}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying emergency phase: {e}")
            traceback.print_exc()
            return False

    def _schedule_emergency_completion(self, lane_id, completion_time):
        """Schedule emergency completion check"""
        self._emergency_completions[lane_id] = completion_time
    def force_adaptive_control(self):
        """Force the traffic light to use adaptive control instead of default SUMO logic"""
        try:
            # Get current logic
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            
            # Create a simple 2-phase adaptive logic to start
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            
            # Phase 1: North-South green
            phase1_state = 'G' * len(controlled_lanes)  # Start with all green for testing
            
            # Phase 2: East-West green  
            phase2_state = 'r' * len(controlled_lanes)  # All red for safety
            
            # Create new adaptive phases
            new_phases = [
                traci.trafficlight.Phase(self.min_green, phase1_state),
                traci.trafficlight.Phase(3, 'y' * len(controlled_lanes)),  # Yellow transition
                traci.trafficlight.Phase(self.min_green, phase2_state),
                traci.trafficlight.Phase(3, 'y' * len(controlled_lanes))   # Yellow transition
            ]
            
            # Create new logic
            new_logic = traci.trafficlight.Logic(
                "adaptive", 0, 0, new_phases
            )
            
            # Apply the new logic
            traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
            traci.trafficlight.setPhase(self.tl_id, 0)
            
            logger.info(f"[ADAPTIVE_OVERRIDE] Forced adaptive control for {self.tl_id}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to force adaptive control: {e}")
            return False
    def check_emergency_completions(self):
        """Check for completed emergencies and serve next in queue"""
        current_time = traci.simulation.getTime()
        
        completed_emergencies = []
        
        for lane_id, completion_time in list(self._emergency_completions.items()):
            if current_time >= completion_time:
                # Check if emergency vehicle has cleared
                if self._emergency_vehicle_cleared(lane_id):
                    completed_emergencies.append(lane_id)
                    logger.info(f"[EMERGENCY_COMPLETE] Emergency on {lane_id} completed")
                else:
                    # Extend if emergency still present
                    self._emergency_completions[lane_id] = current_time + 15
                    logger.info(f"[EMERGENCY_EXTEND] Extended {lane_id} emergency by 15s")

        # Process completed emergencies
        for lane_id in completed_emergencies:
            del self._emergency_completions[lane_id]
            self.emergency_coordinator.emergency_completed(self.tl_id, lane_id)
            
            # Check for next queued emergency
            next_emergency = self.emergency_coordinator.get_next_emergency(self.tl_id)
            if next_emergency:
                logger.info(f"[EMERGENCY_NEXT] Serving queued emergency: {next_emergency}")
                self.reorganize_or_create_phase(next_emergency, 'emergency_vehicle')

    def _emergency_vehicle_cleared(self, lane_id):
        """Enhanced check if emergency vehicle has cleared the lane"""
        try:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            lane_length = traci.lane.getLength(lane_id)
            
            for vid in vehicle_ids:
                try:
                    v_type = traci.vehicle.getTypeID(vid)
                    if 'emergency' in v_type or 'priority' in v_type:
                        # Check vehicle position - if close to end, consider cleared
                        pos = traci.vehicle.getLanePosition(vid)
                        if pos < lane_length * 0.85:  # Still in lane
                            return False
                except traci.TraCIException:
                    continue
                    
            return True  # No emergency vehicles found or they've mostly passed
            
        except Exception as e:
            logger.warning(f"Error checking emergency clearance for {lane_id}: {e}")
            return True  # Assume cleared on error    
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
        elif avg_status <= status_threshold / 2:
            bonus = 1

        self.last_bonus = bonus
        self.last_penalty = penalty
        return bonus, penalty
    def get_state_vector(self):
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0].getPhases())
        queues = [self.get_lane_stats(lane_id)[0] for lane_id in self.lane_ids]
        waits = [self.get_lane_stats(lane_id)[1] for lane_id in self.lane_ids]
        return np.array([current_phase, num_phases] + queues + waits)    
    def rl_action_space(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
        return ['stay'] + list(range(len(logic.getPhases())))

    def calculate_reward(self, bonus=0, penalty=0):
        MAX_VALUES = [0.2, 13.89, 300, 50]
        current_max = [
            max(0.1, max(traci.lane.getLastStepVehicleNumber(lid)/max(1, traci.lane.getLength(lid)) for lid in self.lane_ids)),
            max(5.0, max(traci.lane.getLastStepMeanSpeed(lid) for lid in self.lane_ids)),
            max(30.0, max(traci.lane.getWaitingTime(lid) for lid in self.lane_ids)),
            max(5.0, max(traci.lane.getLastStepHaltingNumber(lid) for lid in self.lane_ids))
        ]
        max_vals = [min(MAX_VALUES[i], current_max[i]) for i in range(4)]
        metrics = np.zeros(4)
        valid_lanes = 0
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
        if valid_lanes == 0:
            return 0
        avg_metrics = metrics / valid_lanes
        self.adjust_weights()  # always adjust before reward
        R = 100 * (
            -self.weights[0] * avg_metrics[0] +
            self.weights[1] * avg_metrics[1] -
            self.weights[2] * avg_metrics[2] -
            self.weights[3] * avg_metrics[3] +
            bonus - penalty
        )
        return np.clip(R, -100, 100)
    def calculate_delta_t(self, R):
        raw_delta_t = self.alpha * (R - self.R_target)
        delta_t = 10 * np.tanh(raw_delta_t / 20)
        return np.clip(delta_t, -10, 10)

    
    def update_R_target(self, window=10):
        if len(self.reward_history) < window or self.phase_count % 10 != 0:
            return
        avg_R = np.mean(list(self.reward_history)[-window:])
        self.R_target = self.r_base + self.r_adjust * (avg_R - self.r_base)


    def get_protected_left_lanes(self):
        """Return list of lanes that serve as protected left turns."""
        protected_lefts = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        for lane_id in controlled_lanes:
            for link in traci.lane.getLinks(lane_id):
                if link[6] == 'l':
                    protected_lefts.append(lane_id)
                    break
        return protected_lefts


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
    def serve_protected_left_if_needed(self):
        """
        Detects blocked left-turn lanes (vehicles not moving and queue > 0 with conflicting straight traffic).
        If found, creates or extends a dedicated protected left phase and serves it immediately.
        Returns True if a protected left phase was served, False otherwise.
        """
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]

            # 1. Find blocked left-turn lane with conflicting straight traffic
            for lane_id in controlled_lanes:
                if not self._is_left_turn(lane_id):
                    continue
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                if not vehicles:
                    continue
                speeds = [traci.vehicle.getSpeed(vid) for vid in vehicles]
                if max(speeds, default=0) >= 0.2:
                    continue  # Not blocked
                conflicting_lanes = self._get_conflicting_straight_lanes(lane_id)
                if not any(traci.lane.getLastStepVehicleNumber(l) > 0 for l in conflicting_lanes):
                    continue  # No conflicting traffic

                # 2. Either find or create a protected left phase
                protected_state = ''.join(
                    'G' if i == controlled_lanes.index(lane_id) else 'r'
                    for i in range(len(controlled_lanes)))
                for idx, phase in enumerate(logic.getPhases()):
                    if phase.state == protected_state:
                        phase_idx = idx
                        break
                else:
                    # Create the phase if not found
                    new_phase = traci.trafficlight.Phase(self.max_green, protected_state)
                    phases = list(logic.getPhases()) + [new_phase]
                    new_logic = traci.trafficlight.Logic(
                        logic.programID, logic.type, len(phases) - 1, phases)
                    traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
                    phase_idx = len(phases) - 1

                # 3. Switch to the protected left phase
                traci.trafficlight.setPhase(self.tl_id, phase_idx)
                traci.trafficlight.setPhaseDuration(self.tl_id, self.max_green)
                self.log_apc_event({
                    "action": "serve_protected_left",
                    "lane_id": lane_id,
                    "phase_idx": phase_idx,
                    "protected_state": protected_state
                })
                print(f"[PROTECTED_LEFT] Served phase {phase_idx} for lane {lane_id} at {self.tl_id}")
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Protected left logic failed: {e}")
            return False

    def _is_left_turn(self, lane_id):
        """Returns True if the lane is a left-turn lane."""
        return any(len(link) > 6 and link[6] == 'l' for link in traci.lane.getLinks(lane_id))

    def _get_conflicting_straight_lanes(self, left_lane):
        """
        Given a left turn lane, returns a list of controlled lanes that are conflicting straight.
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        left_links = traci.lane.getLinks(left_lane)
        left_turn_targets = [link[0] for link in left_links if len(link) > 6 and link[6] == 'l']
        conflicting = []
        for lane in controlled_lanes:
            for link in traci.lane.getLinks(lane):
                if len(link) > 6 and link[6] in ['s', 'S'] and link[0] in left_turn_targets:
                    conflicting.append(lane)
        return conflicting    

    def preload_phases_from_sumo(self):
        for idx, phase in enumerate(self._phase_defs):
            if not any(p['phase_idx'] == idx for p in self._local_state.get('phases', [])):
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
        for p in self._local_state.get("phases", []):
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
            "tl_id": self.tl_id
        })

    def set_phase_from_pkl(self, phase_idx, requested_duration=None):
        print(f"[DB]/[Supabase][SET] set_phase_from_pkl({phase_idx}, requested_duration={requested_duration}) called")
        phase_record = self.load_phase_from_supabase(phase_idx)
        if phase_record:
            duration = requested_duration if requested_duration is not None else phase_record["duration"]
            duration = min(duration, MAX_PHASE_DURATION)  # CAP here
            extended_time = duration - phase_record.get("duration", duration) if requested_duration is not None else 0
            traci.trafficlight.setPhase(self.tl_id, phase_record["phase_idx"])
            traci.trafficlight.setPhaseDuration(self.tl_id, duration)
            self.update_phase_duration_record(phase_record["phase_idx"], duration, extended_time)            
            if hasattr(self, "update_display"):
                current_time = traci.simulation.getTime()
                next_switch = traci.trafficlight.getNextSwitch(self.tl_id)
                self.update_display(phase_idx, duration, current_time, next_switch, extended_time)
            return True
        else:
            print(f"[APC-RL] Phase {phase_idx} not found in PKL, attempting to create or substitute.")
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
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
                for p in self._local_state.get("phases", []):
                    if p["state"] == target_state:
                        print(f"[DB]/[Supabase][SUBSTITUTE] Found Supabase phase with matching state. Using phase_idx={p['phase_idx']}.")
                        traci.trafficlight.setPhase(self.tl_id, p["phase_idx"])
                        traci.trafficlight.setPhaseDuration(self.tl_id, p["duration"])
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
                traci.trafficlight.setPhase(self.tl_id, phase_idx)
                traci.trafficlight.setPhaseDuration(self.tl_id, fallback_duration)
                self.update_phase_duration_record(phase_idx, fallback_duration)
                if hasattr(self, "update_display"):
                    self.update_display(phase_idx, fallback_duration)
                return True
            print(f"[ERROR] Could not find or create a suitable phase for index {phase_idx}.")
            return False


    def serve_true_protected_left_if_needed(self):
        """Enhanced protected left turn detection and service"""
        lane_id, needs_protection = self.detect_blocked_left_turn_conflict()
        if not needs_protection:
            return False

        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        wait = traci.lane.getWaitingTime(lane_id)
        print(f"[PROTECTED_LEFT] Detected blocked left lane {lane_id}. Queue={queue}, Wait={wait}")

        phase_idx = self.create_true_protected_left_phase(lane_id)
        if phase_idx is None:
            print(f"[PROTECTED_LEFT] Failed to create protected left phase for {lane_id}")
            return False

        now = traci.simulation.getTime()
        switched = self.log_phase_switch(phase_idx)
        if not switched:
            return False

        # Dynamic duration based on traffic
        green_duration = min(self.max_green, max(self.min_green, queue * 2 + wait * 0.1))
        traci.trafficlight.setPhaseDuration(self.tl_id, green_duration)
        
        print(f"[PROTECTED_LEFT] Activated phase {phase_idx} for {lane_id} with duration {green_duration:.1f}s")
        
        self.log_apc_event({
            "action": "serve_true_protected_left",
            "lane_id": lane_id,
            "phase": phase_idx,
            "green_duration": green_duration,
            "queue": queue,
            "wait": wait
        })
        
        return True    
    def insert_yellow_phase_if_needed(self, next_phase_idx):
        """
        When switching phases, if any lane is going from green to red,
        insert a yellow phase before switching.
        """
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
            current_phase = traci.trafficlight.getPhase(self.tl_id)
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
                        traci.trafficlight.setPhase(self.tl_id, idx)
                        traci.trafficlight.setPhaseDuration(self.tl_id, 3)
                        print(f"[YELLOW INSERT] Inserted yellow phase before switch at {self.tl_id}")
                        time.sleep(0.2)  # Let yellow phase run (small delay, adjust if needed)
                        break
                else:
                    # Create new yellow phase if not found
                    yellow_phase = traci.trafficlight.Phase(3, ''.join(yellow_state))
                    phases = list(logic.getPhases()) + [yellow_phase]
                    new_logic = traci.trafficlight.Logic(
                        logic.programID, logic.type, len(phases) - 1, phases
                    )
                    traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
                    traci.trafficlight.setPhase(self.tl_id, len(phases) - 1)
                    traci.trafficlight.setPhaseDuration(self.tl_id, 3)
                    print(f"[YELLOW INSERT] Created & inserted new yellow phase before switch at {self.tl_id}")
                    time.sleep(0.2)
        except Exception as e:
            print(f"[ERROR] Yellow phase insertion failed: {e}")


    def enforce_min_green(self):
        """Return True if enough time has passed since last phase switch."""
        current_sim_time = traci.simulation.getTime()
        elapsed = current_sim_time - self.last_phase_switch_sim_time
        if elapsed < self.min_green:
            print(f"[MIN_GREEN ENFORCED] {self.tl_id}: Only {elapsed:.2f}s since last switch, min_green={self.min_green}s")
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
            current_phase = traci.trafficlight.getPhase(self.tl_id)
            if not is_priority and phase_idx == current_phase:
                # Not priority: don't allow changing the duration of the currently running phase
                print("[ADJUST BLOCKED] Wait for phase to finish before applying duration change.")
                return traci.trafficlight.getPhaseDuration(self.tl_id)

            # Get base duration from DB/PKL if possible
            phase_record = self.load_phase_from_supabase(phase_idx) if hasattr(self, "load_phase_from_supabase") else None
            if phase_record and "duration" in phase_record:
                base_duration = phase_record["duration"]
            else:
                # fallback: get from SUMO (may not be accurate if not active)
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
                base_duration = logic.getPhases()[phase_idx].duration if phase_idx < len(logic.getPhases()) else self.min_green

            new_duration = np.clip(base_duration + delta_t, self.min_green, min(self.max_green, MAX_PHASE_DURATION))
            extended_time = new_duration - base_duration
            self.last_extended_time = extended_time

            # Schedule the change if not currently running, or apply immediately if allowed (priority)
            if is_priority and (phase_idx == current_phase):
                traci.trafficlight.setPhaseDuration(self.tl_id, new_duration)
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
                next_switch = traci.trafficlight.getNextSwitch(self.tl_id)
                self.update_display(phase_idx, new_duration, current_time, next_switch, extended_time)
            return new_duration
        except traci.TraCIException as e:
            print(f"[ERROR] Duration adjustment failed: {e}")
            return traci.trafficlight.getPhaseDuration(self.tl_id)
    def schedule_phase_duration_update(self, phase_idx, new_duration):
        self._scheduled_phase_updates[phase_idx] = new_duration

    def apply_scheduled_duration_if_needed(self, phase_idx):
        if phase_idx in self._scheduled_phase_updates:
            duration = self._scheduled_phase_updates.pop(phase_idx)
            duration = min(duration, MAX_PHASE_DURATION)
            traci.trafficlight.setPhaseDuration(self.tl_id, duration)
            
            # Log to Supabase
            self.log_apc_event({
                "action": "apply_scheduled_duration",
                "tl_id": self.tl_id,
                "phase_idx": phase_idx,
                "applied_duration": duration,
                "sim_time": traci.simulation.getTime(),
            })
    def control_step(self):
        """Enhanced control step with proper emergency handling"""
        try:
            current_time = traci.simulation.getTime()
            self.phase_count += 1
            
            # Check for emergency completions first
            self.check_emergency_completions()
            
            # Check for new emergencies
            event_type, event_lane = self.check_special_events()
            
            if event_type == 'emergency_vehicle' and event_lane:
                logger.info(f"[EMERGENCY_DETECTED] Processing emergency on {event_lane}")
                if self.reorganize_or_create_phase(event_lane, event_type):
                    return  # Emergency handled, exit control step
            
            # Only do normal RL control if no active emergencies
            active_emergencies = self.emergency_coordinator.active_emergencies.get(self.tl_id, {})
            if not active_emergencies:
                # Normal RL control
                try:
                    state = self.get_state_vector()
                    if hasattr(self, 'rl_agent') and self.rl_agent:
                        action = self.rl_agent.get_action(state, tl_id=self.tl_id)
                        current_phase = traci.trafficlight.getPhase(self.tl_id)
                        
                        # Only switch if different phase and min green time elapsed
                        if action != current_phase:
                            elapsed = current_time - self.last_phase_switch_sim_time
                            if elapsed >= self.min_green:
                                if self.set_phase_from_supabase(action):
                                    logger.info(f"[RL_SWITCH] {self.tl_id}: Phase {current_phase} → {action}")
                                    self.last_phase_switch_sim_time = current_time
                        
                        # Update Q-learning
                        reward = self.calculate_reward()
                        if hasattr(self, 'prev_state') and hasattr(self, 'prev_action'):
                            self.rl_agent.update_q_table(self.prev_state, self.prev_action, reward, state)
                        self.prev_state = state
                        self.prev_action = action
                    else:
                        # Fallback: cycle through phases if no RL agent
                        current_phase = traci.trafficlight.getPhase(self.tl_id)
                        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
                        num_phases = len(logic.getPhases())
                        
                        elapsed = current_time - self.last_phase_switch_sim_time
                        if elapsed >= self.min_green:
                            next_phase = (current_phase + 1) % num_phases
                            if self.set_phase_from_supabase(next_phase):
                                logger.info(f"[FALLBACK_SWITCH] {self.tl_id}: Phase {current_phase} → {next_phase}")
                                self.last_phase_switch_sim_time = current_time
                except Exception as e:
                    logger.error(f"Error in normal control: {e}")
            else:
                # Emergency active - log status
                active_lanes = list(active_emergencies.keys())
                logger.debug(f"[EMERGENCY_ACTIVE] {self.tl_id} serving emergencies on: {active_lanes}")

        except Exception as e:
            logger.error(f"[ERROR] control_step failed for {self.tl_id}: {e}")
            traceback.print_exc()
        
class EnhancedQLearningAgent:
    def __init__(self, state_size, action_size, adaptive_controller, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 q_table_file="enhanced_q_table.pkl", mode="train", adaptive_params=None):
        self.state_size = state_size
        self.action_size = action_size
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
        self.logger = logging.getLogger(__name__)
        self.adaptive_params = adaptive_params or {
            'min_green': 30, 'max_green': 80, 'starvation_threshold': 40, 'reward_scale': 40,
            'queue_weight': 0.6, 'wait_weight': 0.3, 'flow_weight': 0.5, 'speed_weight': 0.2, 
            'left_turn_priority': 1.2, 'empty_green_penalty': 15, 'congestion_bonus': 10
        }
        self.adaptive_controller = adaptive_controller
        if mode == "eval":
            self.epsilon = 0.0
        elif mode == "adaptive":
            self.epsilon = 0.01
        logger.info(f"AGENT INIT: mode={self.mode}, epsilon={self.epsilon}")

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
            return np.random.randint(action_size)
        return int(np.argmax(qs))
   
   
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
            traci.trafficlight.setPhase(self.adaptive_controller.tl_id, phase_record["phase_idx"])
            traci.trafficlight.setPhaseDuration(self.adaptive_controller.tl_id, phase_record["duration"])
            self.adaptive_controller.update_display(phase_record["phase_idx"], phase_record["duration"])
            print(f"[APC-RL] RL agent set phase {phase_record['phase_idx']} duration={phase_record['duration']} (from PKL)")
        else:
            print(f"[APC-RL] No PKL record for RL agent phase {phase_idx}, fallback to default.")
            traci.trafficlight.setPhase(self.adaptive_controller.tl_id, phase_idx)
            traci.trafficlight.setPhaseDuration(self.adaptive_controller.tl_id, self.adaptive_controller.max_green)
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
        next_switch = traci.trafficlight.getNextSwitch(self.tl_id)
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
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        left_idx = controlled_lanes.index(left_lane)
        protected_state = ''.join(['G' if i == left_idx else 'r' for i in range(len(controlled_lanes))])
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]

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
                traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
                traci.trafficlight.setPhase(self.tl_id, idx)
                self.save_phase_to_supabase(idx, new_duration, protected_state, delta_t, delta_t, penalty=0)
                self._log_apc_event({
                    "action": "extend_protected_left_phase",
                    "tl_id": self.tl_id,
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
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tl_id, new_logic)
        traci.trafficlight.setPhase(self.tl_id, len(phases) - 2)
        self.save_phase_to_supabase(len(phases) - 2, self.max_green, protected_state, delta_t, delta_t, penalty=0)
        self._log_apc_event({
            "action": "create_protected_left_phase",
            "tl_id": self.tl_id,
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
    
    def _get_action_name(self, action):
        return {
            0: "Set Green", 1: "Next Phase", 2: "Extend Phase",
            3: "Shorten Phase", 4: "Balanced Phase"
        }.get(action, f"Unknown Action {action}")

    def load_model(self, filepath=None):
        filepath = filepath or self.q_table_file
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
                if adaptive_params: 
                    self.adaptive_params = adaptive_params
                logger.info(f"Loaded Q-table with {len(self.q_table)} states from {filepath}")
                return True, adaptive_params
            logger.info("No existing Q-table, starting fresh")
            return False, {}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False, {}

    def decay_epsilon(self):
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        logger.info(f"🚦 Epsilon after training/episode: {old_epsilon} -> {self.epsilon}")

    def save_model(self, filepath=None, adaptive_params=None):
        filepath = filepath or self.q_table_file
        try:
            meta = {
                'last_updated': datetime.datetime.now().isoformat(),
                'training_count': len(self.training_data),
                'average_reward': np.mean([x.get('reward', 0) for x in self.training_data[-100:]]) if self.training_data else 0,
            }
            params = {k: getattr(self, k) for k in ['state_size','action_size','learning_rate','discount_factor','epsilon','epsilon_decay','min_epsilon']}
            model_data = {
                'q_table': {k: v.tolist() for k, v in self.q_table.items()},
                'training_data': self.training_data,
                'params': params,
                'metadata': meta,
                'adaptive_params': adaptive_params or self.adaptive_params
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✅ Model saved with {len(self.training_data)} training entries to {filepath}")
            self.training_data = []
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def get_action_name(self, action):
        return {
            0: "Set Green", 1: "Next Phase", 2: "Extend Phase",
            3: "Shorten Phase", 4: "Balanced Phase"
        }.get(action, f"Unknown Action {action}")
    

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
        tl_id = traci.trafficlight.getIDList()[0]
        self.adaptive_phase_controller = AdaptivePhaseController(
            lane_ids=lane_ids,
            tl_id=tl_id,
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
            tl_id = tls_list[0]
            lane_ids = traci.trafficlight.getControlledLanes(tl_id)
            self.adaptive_phase_controller = AdaptivePhaseController(
                lane_ids=lane_ids,
                tl_id=tl_id,
                alpha=1.0,
                min_green=10,
                max_green=60
            )
            self.rl_agent = EnhancedQLearningAgent(
                state_size=12,
                action_size=self.tl_action_sizes[tl_id],
                adaptive_controller=self.adaptive_phase_controller,
                mode=self.mode
            )
            self.adaptive_phase_controller.rl_agent = self.rl_agent
        # Setup AdaptivePhaseController for each intersection
        self.adaptive_phase_controllers = {}
        for tl_id in tls_list:
            lane_ids = traci.trafficlight.getControlledLanes(tl_id)
            apc = AdaptivePhaseController(
                lane_ids=lane_ids,
                tl_id=tl_id,
                alpha=1.0,
                min_green=10,
                max_green=60
            )
            apc.rl_agent = self.rl_agent
            self.adaptive_phase_controllers[tl_id] = apc

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
                traci.trafficlight.setPhase(apc.tl_id, phase_record["phase_idx"])
                traci.trafficlight.setPhaseDuration(apc.tl_id, phase_record["duration"])
            else:
                traci.trafficlight.setPhase(apc.tl_id, phase_idx)
                traci.trafficlight.setPhaseDuration(apc.tl_id, apc.max_green)
            
            if phase_record:
                traci.trafficlight.setPhase(self.adaptive_controller.tl_id, phase_record["phase_idx"])
                traci.trafficlight.setPhaseDuration(self.adaptive_controller.tl_id, phase_record["duration"])
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
            for tl_id, apc in self.adaptive_phase_controllers.items():
                apc.control_step()

            # Defensive re-initialization of APCs if needed (for dynamic networks)
            for tl_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                if tl_id not in self.adaptive_phase_controllers:
                    self.adaptive_phase_controllers[tl_id] = AdaptivePhaseController(
                        lane_ids=lanes,
                        tl_id=tl_id,
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
    parser = argparse.ArgumentParser(description="Run modular SUMO RL traffic simulation")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--mode', choices=['train', 'eval', 'adaptive'], default='train')
    args = parser.parse_args()

    # Create the controller
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    smart_controller = SmartIntersectionController(supabase)

    phase_events = []
    display = TrafficLightPhaseDisplay(phase_events, poll_interval=500)

    def simulation_loop():
        nonlocal phase_events
        try:
            for episode in range(args.episodes):
                logger.info(f"\n{'='*50}\n🚦 STARTING ENHANCED EPISODE {episode + 1}/{args.episodes}\n{'='*50}")
                sumo_binary = "sumo-gui" if args.gui else "sumo"
                sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary), '-c', args.sumo, '--start', '--quit-on-end']

                try:
                    traci.start(sumo_cmd)
                    logger.info("[TRACI] Successfully connected to SUMO")
                except Exception as e:
                    logger.error(f"[ERROR] Could not start TraCI/SUMO: {e}")
                    return

                smart_controller.initialize()
                logger.info("[CONTROLLER] Smart intersection controller initialized")

                # Create RL agent for the controllers
                if smart_controller.controllers:
                    first_controller = list(smart_controller.controllers.values())[0]
                    rl_agent = EnhancedQLearningAgent(
                        state_size=12,
                        action_size=8,
                        adaptive_controller=first_controller,
                        mode=args.mode
                    )
                    rl_agent.load_model()
                    for controller in smart_controller.controllers.values():
                        controller.rl_agent = rl_agent
                    
                    logger.info("[RL_AGENT] Enhanced Q-Learning agent initialized and loaded")

                    # Link phase events for live display
                    if hasattr(first_controller, "_local_state") and "events" in first_controller._local_state:
                        phase_events.clear()
                        phase_events.extend(first_controller._local_state["events"])
                        display.event_log = phase_events
                        logger.info("[DISPLAY] Connected to live phase events")

                step = 0
                last_status_time = 0
                try:
                    while traci.simulation.getMinExpectedNumber() > 0:
                        if args.max_steps and step >= args.max_steps:
                            logger.info(f"Reached max steps ({args.max_steps}), ending episode.")
                            break
                        
                        smart_controller.run_all()
                        traci.simulationStep()
                        step += 1
                        
                        current_time = traci.simulation.getTime()
                        if step % 1000 == 0 or (current_time - last_status_time) >= 30:
                            logger.info(f"Episode {episode + 1}: Step {step} completed (Time: {current_time:.1f}s)")
                            
                            for tl_id in traci.trafficlight.getIDList():
                                current_phase = traci.trafficlight.getPhase(tl_id)
                                next_switch = traci.trafficlight.getNextSwitch(tl_id)
                                time_remaining = next_switch - current_time
                                logger.info(f"  TL {tl_id}: Phase {current_phase}, Time remaining: {time_remaining:.1f}s")
                            
                            last_status_time = current_time
                            
                except (traci.exceptions.FatalTraCIError, ConnectionResetError, OSError) as e:
                    logger.warning(f"[WARN] SUMO/TraCI connection closed: {e}")
                except Exception as e:
                    logger.error(f"[ERROR] Exception in main simulation loop: {e}")
                    traceback.print_exc()
                finally:
                    try:
                        traci.close()
                    except Exception:
                        pass

                # FIXED: Properly save Q-table at end of episode
                if smart_controller.controllers and args.mode == "train":
                    first_controller = list(smart_controller.controllers.values())[0]
                    if hasattr(first_controller, 'rl_agent') and first_controller.rl_agent:
                        first_controller.rl_agent.decay_epsilon()
                        first_controller.rl_agent.save_model()
                        logger.info(f"[RL_AGENT] Q-table saved after episode {episode + 1}")

                smart_controller.controllers.clear()

                if episode < args.episodes - 1:
                    time.sleep(2)
            logger.info(f"\n🎉 All {args.episodes} episodes completed with enhanced smart control!")
        finally:
            try:
                traci.close()
            except Exception:
                pass
            smart_controller.shutdown()
            logger.info("Enhanced simulation resources cleaned up")
            if display is not None:
                try:
                    display.stop()
                except Exception:
                    pass

    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.start()
    
    logger.info("[DISPLAY] Starting enhanced traffic light display...")
    display.start()
    sim_thread.join()

if __name__ == "__main__":
    main()