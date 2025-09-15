import traci
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger("controller")

def log_diag(context, **kwargs):
    """
    Centralized diagnostic logger for controller events.
    Usage: log_diag("set_phase_from_API", phase_idx=..., base=..., requested=...)
    """
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

def get_or_create_all_red_phase(tls_id: str, clearance_s: float = 3.0, phase_cap: int = 12) -> int | None:
    """
    Build or locate an all-red phase for the given TLS.
    Returns the phase index, or None if not possible.
    """
    try:
        logic = get_current_logic(tls_id)
        if not logic:
            return None
        n_links = len(traci.trafficlight.getControlledLinks(tls_id))
        all_red = 'r' * max(0, n_links)
        phases = list(logic.getPhases())
        # Find an existing all-red
        for idx, ph in enumerate(phases):
            if ph.state == all_red:
                return idx
        # Overwrite if at phase cap (prefer yellow phase)
        if len(phases) >= phase_cap:
            ow_idx = next((i for i, ph in enumerate(phases) if 'y' in ph.state), None)
            if ow_idx is None:
                # Fallback: overwrite last phase
                ow_idx = len(phases) - 1
            phases[ow_idx] = traci.trafficlight.Phase(float(clearance_s), all_red)
            new_logic = traci.trafficlight.Logic(logic.programID, logic.type,
                                                 min(logic.currentPhaseIndex, len(phases)-1), phases)
            traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_logic)
            return ow_idx
        # Append new
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