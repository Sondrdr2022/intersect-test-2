import traci
import logging
import numpy as np
from collections import defaultdict
import time

logger = logging.getLogger("controller")

try:
    from config import PHASE_CAP, MIN_YELLOW_S, YELLOW_AUDIT_SUPPRESS_WINDOW_S
except Exception:
    PHASE_CAP = 32
    MIN_YELLOW_S = 4.0
    YELLOW_AUDIT_SUPPRESS_WINDOW_S = 5.0

# Simple rate-limit memory for [YELLOW AUDIT] spam
_yellow_audit_last = {}  # key: (tls_id, i, j) -> last_log_time

def log_diag(context, **kwargs):
    msg = f"[DIAGNOSTIC][{context}] " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.error(msg)

def get_current_logic(tls_id: str):
    try:
        prog = traci.trafficlight.getProgram(tls_id)
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        for logic in (logics or []):
            if getattr(logic, "programID", None) == prog:
                return logic
        return logics[0] if logics else None
    except Exception as e:
        logger.error(f"[LOGIC] Failed to get current logic for {tls_id}: {e}")
        return None

def get_or_create_all_red_phase(tls_id: str, clearance_s: float = 3.0, phase_cap: int = None) -> int | None:
    if phase_cap is None:
        phase_cap = PHASE_CAP
    try:
        logic = get_current_logic(tls_id)
        if not logic:
            return None
        n_links = len(traci.trafficlight.getControlledLinks(tls_id))
        all_red = 'r' * max(0, n_links)
        phases = list(logic.getPhases())
        for idx, ph in enumerate(phases):
            if ph.state == all_red:
                return idx
        if len(phases) >= phase_cap:
            ow_idx = next((i for i, ph in enumerate(phases) if 'y' in ph.state), None)
            if ow_idx is None:
                ow_idx = len(phases) - 1
            phases[ow_idx] = traci.trafficlight.Phase(float(clearance_s), all_red)
            new_logic = traci.trafficlight.Logic(logic.programID, logic.type,
                                                 min(logic.currentPhaseIndex, len(phases)-1), phases)
            traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_logic)
            return ow_idx
        phases.append(traci.trafficlight.Phase(float(clearance_s), all_red))
        new_logic = traci.trafficlight.Logic(logic.programID, logic.type,
                                             min(logic.currentPhaseIndex, len(phases)-1), phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_logic)
        return len(phases) - 1
    except Exception as e:
        logger.error(f"[ALL_RED] Failed for {tls_id}: {e}")
        return None

def collect_lane_stats(
    lane_ids,
    vehicle_classes,
    all_vehicles=None,
    left_turn_lanes=None,
    right_turn_lanes=None,
    lane_lengths=None,
    lane_edge_ids=None,
    lane_to_tl=None
):
    """
    Centralized lane stats collector. Returns dict keyed by lane_id.
    Args:
        lane_ids: list of lane IDs.
        vehicle_classes: dict {vehicle_id: class}.
        all_vehicles: set of vehicle IDs to include (optional; filters vehicle_ids).
        left_turn_lanes, right_turn_lanes: sets of lane IDs (optional).
        lane_lengths: dict {lane_id: length} (optional).
        lane_edge_ids: dict {lane_id: edge_id} (optional).
        lane_to_tl: dict {lane_id: tl_id} (optional).
    Returns:
        lane_data: dict {lane_id: {...stats...}}
    """
    lane_data = {}
    results_dict = {lid: traci.lane.getSubscriptionResults(lid) for lid in lane_ids}
    vehicle_count = np.zeros(len(lane_ids), dtype=np.float32)
    mean_speed = np.zeros(len(lane_ids), dtype=np.float32)
    queue_length = np.zeros(len(lane_ids), dtype=np.float32)
    waiting_time = np.zeros(len(lane_ids), dtype=np.float32)
    lane_length = np.array([lane_lengths.get(lid, traci.lane.getLength(lid)) if lane_lengths else traci.lane.getLength(lid) for lid in lane_ids], dtype=np.float32)
    ambulance_mask = np.zeros(len(lane_ids), dtype=bool)
    vehicle_ids_arr = []
    for idx, lane_id in enumerate(lane_ids):
        results = results_dict[lane_id]
        vids = results.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, [])
        if all_vehicles is not None:
            vids = [vid for vid in vids if vid in all_vehicles]
        vehicle_ids_arr.append(vids)
        vehicle_count[idx] = results.get(traci.constants.LAST_STEP_VEHICLE_NUMBER, 0)
        mean_speed[idx] = max(results.get(traci.constants.LAST_STEP_MEAN_SPEED, 0), 0.0)
        queue_length[idx] = results.get(traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
        try:
            waiting_time[idx] = float(traci.lane.getWaitingTime(lane_id))
        except Exception:
            waiting_time[idx] = 0.0
        ambulance_mask[idx] = any(vehicle_classes.get(vid) == 'emergency' for vid in vids)
    densities = np.divide(vehicle_count, lane_length, out=np.zeros_like(vehicle_count), where=lane_length > 0)
    left_turn_mask = np.array([lane_id in (left_turn_lanes or set()) for lane_id in lane_ids])
    right_turn_mask = np.array([lane_id in (right_turn_lanes or set()) for lane_id in lane_ids])
    for idx, lane_id in enumerate(lane_ids):
        vids = vehicle_ids_arr[idx]
        def safe_count_classes(vids_inner):
            counts = defaultdict(int)
            for vid in vids_inner:
                vclass = vehicle_classes.get(vid)
                if vclass:
                    counts[vclass] += 1
            return counts
        lane_data[lane_id] = {
            'queue_length': float(queue_length[idx]),
            'waiting_time': float(waiting_time[idx]),
            'density': float(densities[idx]),
            'mean_speed': float(mean_speed[idx]),
            'vehicle_ids': vids,
            'flow': float(vehicle_count[idx]),
            'lane_id': lane_id,
            'ambulance': bool(ambulance_mask[idx]),
            'vehicle_classes': safe_count_classes(vids),
            'left_turn': bool(left_turn_mask[idx]),
            'right_turn': bool(right_turn_mask[idx]),
            'lane_length': lane_length[idx],
            'edge_id': lane_edge_ids[lane_id] if lane_edge_ids and lane_id in lane_edge_ids else "",
            'tl_id': lane_to_tl[lane_id] if lane_to_tl and lane_id in lane_to_tl else "",
        }
    return lane_data

# ====================== NEW: Global yellow-phase enforcer ======================

try:
    from config import MIN_YELLOW_S
except Exception:
    MIN_YELLOW_S = 4.0

def ensure_global_yellow_phases(tls_id: str, yellow_duration: float | None = None, phase_cap: int = None) -> bool:
    if phase_cap is None:
        phase_cap = PHASE_CAP
    try:
        logic = get_current_logic(tls_id)
        if not logic:
            return False
        phases = list(logic.getPhases())
        if not phases:
            return False

        n_links = len(traci.trafficlight.getControlledLinks(tls_id))
        default_y = float(yellow_duration if yellow_duration is not None else MIN_YELLOW_S)

        existing_states = {ph.state: i for i, ph in enumerate(phases)}
        missing_yellow_states = []

        for ph in phases:
            st = ph.state
            if not st:
                continue
            if 'y' in st.lower():
                continue
            if set(st.lower()) == {'r'}:
                continue
            y_state = ''.join(('y' if ch.upper() == 'G' else 'r') for ch in st[:n_links])
            if 'y' not in y_state:
                continue
            if y_state not in existing_states and y_state not in missing_yellow_states:
                missing_yellow_states.append(y_state)

        if not missing_yellow_states:
            get_or_create_all_red_phase(tls_id, clearance_s=3.0, phase_cap=phase_cap)
            return False

        changed = False
        while missing_yellow_states and len(phases) < phase_cap:
            yst = missing_yellow_states.pop(0)
            phases.append(traci.trafficlight.Phase(default_y, yst))
            changed = True

        if missing_yellow_states:
            for idx, ph in enumerate(phases):
                if not missing_yellow_states:
                    break
                if 'y' in ph.state:
                    yst = missing_yellow_states.pop(0)
                    phases[idx] = traci.trafficlight.Phase(default_y, yst)
                    changed = True

        if missing_yellow_states:
            logger.warning(f"[YELLOW_ENFORCER] {tls_id}: Could not install {len(missing_yellow_states)} yellow states due to phase cap ({phase_cap}).")

        if changed:
            new_logic = traci.trafficlight.Logic(
                logic.programID, logic.type,
                min(getattr(logic, "currentPhaseIndex", 0), len(phases)-1),
                phases
            )
            traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_logic)

        get_or_create_all_red_phase(tls_id, clearance_s=3.0, phase_cap=phase_cap)
        return changed
    except Exception as e:
        logger.error(f"[YELLOW_ENFORCER] Failed for {tls_id}: {e}")
        return False    
def audit_and_repair_yellow_phases_all_tls(controller):
    import traci
    problems_all = []
    now = time.time()

    for tls_id in traci.trafficlight.getIDList():
        logic = get_current_logic(tls_id)
        if not logic:
            continue
        phases = list(logic.getPhases())
        n = len(phases)
        problems_this_tls = []
        for i, from_ph in enumerate(phases):
            for j, to_ph in enumerate(phases):
                if i == j:
                    continue
                nmin = min(len(from_ph.state), len(to_ph.state))
                has_gr = any(
                    from_ph.state[k].upper() == 'G' and to_ph.state[k].upper() == 'R'
                    for k in range(nmin)
                )
                if not has_gr:
                    continue
                y_state = ''.join(
                    'y' if (from_ph.state[k].upper() == 'G' and to_ph.state[k].upper() == 'R') else from_ph.state[k]
                    for k in range(nmin)
                )
                found = False
                for ph in phases:
                    if ph.state[:nmin] == y_state:
                        found = True
                        break
                if not found:
                    key = (tls_id, i, j)
                    last = _yellow_audit_last.get(key, 0)
                    if now - last >= YELLOW_AUDIT_SUPPRESS_WINDOW_S:
                        logger.warning(f"[YELLOW AUDIT] {tls_id}: Missing yellow between phase {i} ({from_ph.state}) and {j} ({to_ph.state})")
                        _yellow_audit_last[key] = now
                    problems_this_tls.append((tls_id, i, j))

        if problems_this_tls:
            changed = ensure_global_yellow_phases(tls_id, phase_cap=PHASE_CAP)
            if changed:
                logger.warning(f"[YELLOW AUDIT] {tls_id}: Repaired missing yellow phases.")
                # Only re-init if the APC exposes such a method
                apc = getattr(controller, "adaptive_phase_controllers", {}).get(tls_id)
                if apc and hasattr(apc, "_invalidate_logic_cache"):
                    apc._invalidate_logic_cache()
        problems_all.extend(problems_this_tls)

    return problems_all

def enforce_yellow_phases_all_controllers(controller):
    # unchanged except it will benefit from PHASE_CAP via APC methods
    from collections.abc import Mapping
    logger = logging.getLogger("controller")
    ctrls = getattr(controller, "adaptive_phase_controllers", None)
    if not ctrls or not isinstance(ctrls, Mapping):
        logger.error("[NETWORK YELLOW PATCH] No adaptive_phase_controllers found on controller")
        return
    num_fixed = 0
    for tls_id, apc in ctrls.items():
        logic = apc._get_logic()
        phases = getattr(logic, "phases", None)
        if not phases:
            continue
        n = len(phases)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                from_state = phases[i].state
                to_state = phases[j].state
                nmin = min(len(from_state), len(to_state))
                if any(from_state[k].upper() == 'G' and to_state[k].upper() == 'R' for k in range(nmin)):
                    yellow_idx, yellow_dur = apc.get_or_create_yellow_phase(i, j, apc.max_adaptive_yellow, allow_overwrite=True)
                    if yellow_idx is not None:
                        num_fixed += 1
                        logger.info(f"[NETWORK YELLOW PATCH] {tls_id}: Ensured yellow for {i}->{j} (idx={yellow_idx}, dur={yellow_dur:.2f})")
    logger.info(f"[NETWORK YELLOW PATCH] Completed. Total yellow transitions ensured: {num_fixed}")