import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
from collections import defaultdict, deque
import threading
import numpy as np
import traci
from utils import get_current_logic, log_diag

logger = logging.getLogger("controller")

class EventType(Enum):
    EMERGENCY_VEHICLE = "emergency_vehicle"
    ACCIDENT = "accident"
    HEAVY_CONGESTION = "heavy_congestion"
    SPILLBACK = "spillback"
    GRIDLOCK = "gridlock"
    PHASE_FAILURE = "phase_failure"
    DEMAND_SURGE = "demand_surge"
    WEATHER = "weather"
    CONSTRUCTION = "construction"
    SPECIAL_EVENT = "special_event"

class CoordinationStrategy(Enum):
    GREEN_WAVE = "green_wave"
    EMERGENCY_PREEMPTION = "emergency_preemption"
    SPILLBACK_PREVENTION = "spillback_prevention"
    LOAD_BALANCING = "load_balancing"
    METERING = "metering"
    CLEARANCE = "clearance"
    ADAPTIVE_TIMING = "adaptive_timing"

@dataclass
class TrafficEvent:
    event_id: str
    event_type: EventType
    location: Tuple[float, float]
    affected_lanes: List[str]
    affected_intersections: Set[str]
    severity: float
    timestamp: float
    duration_estimate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

@dataclass
class IntersectionGroup:
    group_id: str
    members: Set[str]
    group_type: str
    leader: Optional[str] = None
    coordination_strategy: Optional[CoordinationStrategy] = None
    priority_level: int = 1
    active_since: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
class EventDrivenCorridorCoordinator:
    def __init__(self, controller, config: Optional[Dict] = None):
        self.controller = controller
        cfg = config or {}
        self.detection_interval = float(cfg.get("detection_interval", 5.0))
        self.coordination_radius = float(cfg.get("coordination_radius", 500.0))
        self.min_group_size = int(cfg.get("min_group_size", 2))
        self.max_group_size = int(cfg.get("max_group_size", 8))
        self.event_timeout = float(cfg.get("event_timeout", 300.0))
        self.congestion_threshold = float(cfg.get("congestion_threshold", 0.6))
        self.spillback_threshold = float(cfg.get("spillback_threshold", 0.8))
        self.emergency_detection_distance = float(cfg.get("emergency_detection_distance", 200.0))
        self._intersection_snapshots: Dict[str, dict] = {}
        self._network_escalation_active = False
        self._last_network_eval = 0.0
        self.network_emergency_threshold = 0.8
        self.network_emergency_ratio = 0.5
        self.active_events: Dict[str, TrafficEvent] = {}
        self.active_groups: Dict[str, IntersectionGroup] = {}
        self.intersection_positions: Dict[str, Tuple[float, float]] = {}
        self.lane_to_intersection: Dict[str, str] = {}
        self._upstream_tls: Dict[str, Set[str]] = defaultdict(set)
        self._downstream_tls: Dict[str, Set[str]] = defaultdict(set)
        self._congestion_clusters: List[List[str]] = []
        self._active_responses: Dict[str, str] = {}
        self._response_effectiveness: Dict[str, float] = defaultdict(float)
        self._priority_locks: Dict[str, float] = {}
        self._active_priorities: Dict[str, Any] = {}
        self._last_phase_by_tls: Dict[str, int] = {}
        self.adjacency_matrix: Dict[str, Set[str]] = defaultdict(set)
        self.distance_matrix: Dict[Tuple[str, str], float] = {}
        self.last_detection_time = 0.0
        self.vehicle_tracking: Dict[str, Dict] = {}
        self.congestion_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.phase_locks: Dict[str, Dict] = {}
        self.timing_overrides: Dict[str, Dict] = {}
        self.group_coordination_state: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self._build_network_topology()
        self._rebuild_direction_maps()
        self._log_mapping_status()
        self.edge_to_lanes = None
        logger.info(f"[EVENT_COORDINATOR] Initialized for {len(self.controller.adaptive_phase_controllers)} intersections")

    def _log_mapping_status(self):
        logger.info(f"[COORDINATOR] Known intersections: {list(self.intersection_positions.keys())}")
        logger.info(f"[COORDINATOR] Lane-to-intersection map (sample): {list(self.lane_to_intersection.items())[:10]}")
        for lane, tl in self.lane_to_intersection.items():
            logger.info(f"[COORDINATOR] Lane {lane} -> Intersection {tl}")
    def _build_edge_to_lanes_cache(self):
        """
        Build a mapping from edge ID to list of lane IDs, for all edges.
        This works on all SUMO/TraCI versions.
        """
        logger.info("[PATCH] Building edge-to-lanes mapping (fallback for old SUMO/TraCI).")
        edge_to_lanes = {}
        for lid in traci.lane.getIDList():
            eid = traci.lane.getEdgeID(lid)
            edge_to_lanes.setdefault(eid, []).append(lid)
        self.edge_to_lanes = edge_to_lanes

    def _get_lanes_for_edge(self, edge_id):
        """
        Return list of lane IDs for the given edge_id.
        Uses traci.edge.getLaneIDs if available, otherwise efficient fallback.
        """
        # Try the fast way first
        edge = getattr(traci, "edge", None)
        get_lane_ids = getattr(edge, "getLaneIDs", None)
        if callable(get_lane_ids):
            try:
                return get_lane_ids(edge_id)
            except Exception as e:
                logger.warning(f"[PATCH] traci.edge.getLaneIDs failed for edge {edge_id}: {e}")
                # Fallback to manual cache

        # Fallback: build or use cached mapping
        if self.edge_to_lanes is None:
            self._build_edge_to_lanes_cache()
        return self.edge_to_lanes.get(edge_id, [])
    def get_phase_bias(self, tl_id: str):
        try:
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0] if tl_id in traci.trafficlight.getIDList() else None
            if not apc or not logic or not hasattr(self, 'lane_data'):
                return None
            n = len(logic.phases)
            if n == 0:
                return None
            base = np.ones(n, dtype=float)
            resp = self._active_responses.get(tl_id)
            # Gather queue info for heuristic mapping: phase -> total halting vehicles
            queue_by_phase = []
            for pidx, ph in enumerate(logic.phases):
                st = ph.state
                total_q = 0
                cons = traci.trafficlight.getControlledLanes(tl_id)
                for i, lane in enumerate(cons):
                    if i < len(st) and st[i].upper() == 'G':
                        total_q += self.lane_data.get(lane, {}).get('queue_length', 0)
                queue_by_phase.append(total_q)
            q_arr = np.array(queue_by_phase, dtype=float)
            q_norm = (q_arr / (q_arr.max() + 1e-6)) if q_arr.max() > 0 else q_arr
            # ... rest of function unchanged ...
            return base
        except Exception:
            return None

    def _maybe_escalate_network(self, now: float):
        if now - self._last_network_eval < 10.0:
            return
        self._last_network_eval = now
        severities = []
        for tl_id in self.controller.adaptive_phase_controllers.keys():
            try:
                sev = self._calculate_tl_congestion_severity(tl_id)
                severities.append(sev)
            except Exception:
                continue
        if not severities:
            return
        high = [s for s in severities if s >= self.network_emergency_threshold]
        ratio = len(high) / len(severities)
        if ratio >= self.network_emergency_ratio and not self._network_escalation_active:
            self._network_escalation_active = True
            logger.error(f"[NETWORK][ESCALATION] Activated network_emergency (ratio={ratio:.2f})")
            for tl in self.controller.adaptive_phase_controllers:
                self._active_responses[tl] = "network_emergency"
        elif self._network_escalation_active and ratio < (self.network_emergency_ratio * 0.6):
            self._network_escalation_active = False
            logger.warning(f"[NETWORK][ESCALATION] Deactivated network_emergency (ratio={ratio:.2f})")
            for tl, resp in list(self._active_responses.items()):
                if resp == "network_emergency":
                    del self._active_responses[tl]
    def get_allowed_phase_mask(self, tl_id: str):
        """
        Return a boolean list (len = current phases) where True = allowed.
        None = no hard restriction.
        Example: in metering, allow only subset of phases every other cycle.
        """
        try:
            resp = self._active_responses.get(tl_id)
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            n = len(logic.phases)
            if n == 0:
                return None
            mask = [True] * n
            if resp == "metering":
                # Simple example: allow only even-indexed service phases (not yellows)
                for i, ph in enumerate(logic.phases):
                    if i % 2 == 1 and 'Y' not in ph.state.upper():
                        mask[i] = False
                if not any(mask):
                    return None
                return mask
            if resp == "network_emergency":
                # Allow only top-queue phase + current to avoid deadlock
                queues = []
                cons = traci.trafficlight.getControlledLanes(tl_id)
                for pidx, ph in enumerate(logic.phases):
                    state = ph.state
                    q_sum = 0
                    for li, lane in enumerate(cons):
                        if li < len(state) and state[li].upper() == 'G':
                            q_sum += traci.lane.getLastStepHaltingNumber(lane)
                    queues.append(q_sum)
                if queues:
                    top = int(np.argmax(queues))
                    current = traci.trafficlight.getPhase(tl_id)
                    mask = [False]*n
                    mask[top] = True
                    if current != top:
                        mask[current] = True
                    return mask
            return None
        except Exception:
            return None
    def _build_network_topology(self):
        try:
            for tl_id in self.controller.adaptive_phase_controllers.keys():
                try:
                    pos = traci.junction.getPosition(tl_id)
                    self.intersection_positions[tl_id] = pos
                except Exception:
                    self.intersection_positions[tl_id] = (0.0, 0.0)
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                for lane_id in apc.lane_ids:
                    self.lane_to_intersection[lane_id] = tl_id
            tl_ids = list(self.controller.adaptive_phase_controllers.keys())
            for i, tl1 in enumerate(tl_ids):
                for j, tl2 in enumerate(tl_ids):
                    if i != j:
                        pos1 = self.intersection_positions[tl1]
                        pos2 = self.intersection_positions[tl2]
                        dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        self.distance_matrix[(tl1, tl2)] = dist
                        if dist <= self.coordination_radius:
                            self.adjacency_matrix[tl1].add(tl2)
        except Exception as e:
            logger.error(f"[EVENT_COORDINATOR] Failed to build topology: {e}")
    def step(self, current_time: Optional[float] = None, lane_data: Optional[dict] = None):
        # PATCH: Track all emergency vehicles and update intersection states accordingly
        with self.lock:
            now = current_time if current_time is not None else self._get_sim_time()
            self.lane_data = lane_data if lane_data is not None else {}

            # --- PATCHED: Emergency vehicle continuous tracking ---
            self._continuous_emergency_vehicle_preemption(now)

            if now - self.last_detection_time >= self.detection_interval:
                self._detect_events(now)
                self.last_detection_time = now
            self._update_events(now)
            self._update_intersection_groups(now)
            sorted_groups = sorted(self.active_groups.values(), key=lambda g: g.priority_level, reverse=True)
            for group in sorted_groups:
                try:
                    strategy = group.coordination_strategy
                    if strategy == CoordinationStrategy.EMERGENCY_PREEMPTION:
                        self._execute_emergency_preemption(group, now)
                    # ...rest unchanged...
                except Exception as e:
                    logger.error(f"[COORDINATION] Failed for group {group.group_id}: {e}")
            self._cleanup_expired_items(now)
            self._maybe_escalate_network(now)
            self.network_gridlock_watchdog() 
    def _continuous_emergency_vehicle_preemption(self, now):
        """
        Continuously track all emergency vehicles, map them to intersections via cache,
        and set/queue green for the relevant approach with highest priority.
        """
        try:
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    vclass = traci.vehicle.getVehicleClass(vehicle_id)
                    if vclass not in ['emergency', 'authority']:
                        continue

                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    if not lane_id:
                        continue

                    # Use cached mapping to find intersection
                    tl_id = self.lane_to_intersection.get(lane_id)
                    if not tl_id:
                        continue

                    apc = self.controller.adaptive_phase_controllers.get(tl_id)
                    if not apc:
                        continue

                    # Find the phase that serves this lane
                    phase_idx = apc.find_phase_for_lane(lane_id)
                    if phase_idx is None:
                        continue

                    # Issue/queue emergency request for this phase
                    apc.request_phase_change(phase_idx, priority_type='emergency')

                    # Optionally, add diagnostic logging for auditing
                    log_diag(
                        "emergency_live_preemption",
                        tls_id=tl_id,
                        vehicle_id=vehicle_id,
                        lane_id=lane_id,
                        phase_idx=phase_idx,
                        sim_time=now
                    )

                except Exception as e:
                    logger.error(f"[PATCH][EMERGENCY TRACK] Vehicle {vehicle_id}: {e}")

        except Exception as e:
            logger.error(f"[PATCH][EMERGENCY TRACK] Failed: {e}")    
    def _safe_group_phase_switch(self, tl_id, target_phase, requested_duration=None, reason="group_coordination"):
        """
        Enforces strict phase-end logic and mandatory local safety validation.
        If phase has not ended or not safe, queue the request for execution at phase end.
        """
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        if not apc:
            logger.error(f"[GROUP_PHASE_SWITCH][{tl_id}] ERROR: No APC found for target {target_phase}")
            return False
        try:
            # Always resolve current phase safely
            try:
                current_phase = int(traci.trafficlight.getPhase(tl_id))
            except Exception:
                logic = self.controller._get_traffic_light_logic(tl_id)
                current_phase = min(int(target_phase), len(getattr(logic, "phases", [])) - 1) if logic else int(target_phase)

            # If not at gate: queue
            if not apc._phase_has_ended():
                apc.request_phase_change(
                    int(target_phase),
                    priority_type='group',
                    extension_duration=(float(requested_duration) if requested_duration is not None else None)
                )
                logger.info(f"[GROUP_PHASE_SWITCH][{tl_id}] Queued (gate not reached) -> {target_phase} (reason={reason})")
                return True

            # Gate passed: validate local safety (DZ + approach + min-hold)
            is_safe, why = apc._validate_phase_switch_safety(tl_id, current_phase, int(target_phase))
            if not is_safe:
                apc.request_phase_change(
                    int(target_phase),
                    priority_type='group',
                    extension_duration=(float(requested_duration) if requested_duration is not None else None)
                )
                logger.info(f"[GROUP_PHASE_SWITCH][{tl_id}] Deferred by safety: {why}")
                return True

            # Apply via APC (handles yellow/clearance)
            ok = apc.set_phase_from_API(int(target_phase), requested_duration=requested_duration, do_intergreen=True)
            logger.info(f"[GROUP_PHASE_SWITCH][{tl_id}] Applied at phase end -> ok={ok}")
            apc._log_apc_event({
                "action": "group_phase_switch",
                "from_phase": current_phase,
                "to_phase": int(target_phase),
                "requested_duration": requested_duration,
                "reason": reason,
                "applied": ok
            })
            return ok

        except Exception as e:
            logger.error(f"[GROUP_PHASE_SWITCH][{tl_id}] Exception: {e}")
            try:
                # Final fallback: queue for later safe execution
                apc.request_phase_change(int(target_phase), priority_type='group',
                                        extension_duration=(float(requested_duration) if requested_duration is not None else None))
                return True
            except Exception as e2:
                logger.error(f"[GROUP_PHASE_SWITCH][{tl_id}] Fallback failed: {e2}")
                return False
    def update_topology(self, force: bool = False):
        try:
            if force:
                self.adjacency_matrix.clear()
                self.distance_matrix.clear()
                self.intersection_positions.clear()
                self.lane_to_intersection.clear()
            self._build_network_topology()
            self._rebuild_direction_maps()
            self._log_mapping_status()
            logger.info(f"[CORRIDOR_COORDINATOR] Topology updated (force={force})")
        except Exception as e:
            logger.error(f"[CORRIDOR_COORDINATOR] update_topology failed: {e}")    
    
    def _detect_events(self, current_time: float):
        """Detect various traffic events that require coordination"""
        
        # Detect emergency vehicles
        self._detect_emergency_vehicles(current_time)
        
        # Detect congestion events
        self._detect_congestion_events(current_time)
        
        # Detect spillback conditions
        self._detect_spillback_events(current_time)
        
        # Detect gridlock conditions
        self._detect_gridlock_events(current_time)
        
        # Detect phase failures
        self._detect_phase_failures(current_time)
        
        # Detect demand surges
        self._detect_demand_surges(current_time)
    def update_topology(self, force: bool = False):
        """Rebuild network topology and compatibility maps."""
        try:
            if force:
                self.adjacency_matrix.clear()
                self.distance_matrix.clear()
                self.intersection_positions.clear()
                self.lane_to_intersection.clear()
            self._build_network_topology()
            self._rebuild_direction_maps()
            logger.info(f"[CORRIDOR_COORDINATOR] Topology updated (force={force})")
        except Exception as e:
            logger.error(f"[CORRIDOR_COORDINATOR] update_topology failed: {e}")

    def _rebuild_direction_maps(self):
        try:
            self._upstream_tls.clear()
            self._downstream_tls.clear()
            for tl_id, neighbors in self.adjacency_matrix.items():
                for nb in neighbors:
                    self._upstream_tls[tl_id].add(nb)
                    self._downstream_tls[tl_id].add(nb)
        except Exception as e:
            logger.error(f"[COORDINATOR] _rebuild_direction_maps failed: {e}")
    def detect_intersection_groups_improved(self):
        try:
            visited = set()
            groups = []
            for tl in self.controller.adaptive_phase_controllers:
                if tl in visited:
                    continue
                comp = []
                stack = [tl]
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    comp.append(cur)
                    for nb in self.adjacency_matrix.get(cur, set()):
                        if nb not in visited:
                            stack.append(nb)
                if comp:
                    groups.append(comp)
            return groups
        except Exception:
            return []
    def request_downstream_flush(self, lane_id: str) -> bool:
        """
        Ask downstream (or current) intersection to favor clearance for flows fed by lane_id.
        Compatibility stub: trigger a reasonable phase at the controlling TLS if possible.
        """
        try:
            # Find TLS controlling this lane (current)
            tl_id = self.lane_to_intersection.get(lane_id)
            apc = self.controller.adaptive_phase_controllers.get(tl_id) if tl_id else None
            if not apc:
                return False

            # Prefer a phase that serves this lane; else best phase for traffic
            phase = apc.find_phase_for_lane(lane_id)
            if phase is None:
                phase = apc.find_best_phase_for_traffic()
            if phase is None:
                return False

            # Use a short, safe duration nudge
            dur = max(apc.min_green // 2, 5)
            apc.set_phase_from_API(phase, requested_duration=dur)
            # Mark an 'active response' type for visibility
            self._active_responses[tl_id] = "clearance"
            self._response_effectiveness[tl_id] = 0.0
            logger.info(f"[DOWNSTREAM_FLUSH] Requested at {tl_id} for lane {lane_id} -> phase {phase}")
            return True
        except Exception as e:
            logger.error(f"[DOWNSTREAM_FLUSH] Failed for lane {lane_id}: {e}")
            return False

    def coordinate_congestion_response(self, cluster: List[str]):
        """
        Activate basic congestion response for a cluster of intersections.
        Compatibility stub: mark active response and let APCs adapt.
        """
        try:
            for tl_id in cluster:
                self._active_responses[tl_id] = "congestion"
                self._response_effectiveness[tl_id] = 0.0
            # Keep a record of clusters for logging
            self._congestion_clusters.append(list(cluster))
            logger.info(f"[CONGESTION] Activated response for cluster of {len(cluster)} TLS")
        except Exception as e:
            logger.error(f"[CONGESTION] coordinate_congestion_response failed: {e}")

    def _calculate_tl_congestion_severity(self, tl_id: str) -> float:
        """Compatibility: reuse internal per-intersection congestion metric."""
        try:
            return float(self._calculate_intersection_congestion(tl_id))
        except Exception:
            return 0.0

    # ======== RL agent coordination compatibility ========

    def should_allow_phase(self, tl_id: str, phase_idx: int) -> bool:
        """
        Return whether the coordinator allows this phase now.
        By default, allow all phases; you can restrict under special responses.
        """
        # Example: if metering is active, allow everything but encourage shorter greens
        return True

    def get_next_phase(self, tl_id: str) -> int:
        """
        Coordinator's suggestion for the next phase if overriding RL.
        Defaults to APC's own best phase heuristic.
        """
        try:
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if not apc:
                return 0
            best = apc.find_best_phase_for_traffic()
            if best is None:
                # fall back to current
                return int(traci.trafficlight.getPhase(tl_id))
            return int(best)
        except Exception:
            return 0

    def enforce_phase_fairness(self, tl_id: str, phase_idx: int) -> int:
        """
        Optionally rotate phases to prevent starvation; default is no change.
        """
        return int(phase_idx)

    def suggest_phase_duration(self, tl_id: str, phase_idx: int) -> Optional[float]:
        """
        Coordinator’s suggested total duration for the chosen phase.
        None means 'let APC/RL decide or keep current'.
        """
        return None

    def record_phase_activation(self, tl_id: str, phase_idx: int, duration: Optional[float]):
        """
        Bookkeeping hook called by the RL agent after a phase decision.
        """
        try:
            self._last_phase_by_tls[tl_id] = int(phase_idx)
        except Exception:
            pass
    def _detect_emergency_vehicles(self, current_time: float):
        try:
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    vehicle_class = traci.vehicle.getVehicleClass(vehicle_id)
                    if vehicle_class in ['emergency', 'authority']:
                        pos = traci.vehicle.getPosition(vehicle_id)
                        route = traci.vehicle.getRoute(vehicle_id)
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        affected_intersections = self._get_intersections_on_route(route, pos, speed)
                        if affected_intersections:
                            event_id = f"emergency_{vehicle_id}_{int(current_time)}"
                            if not any(e.metadata.get('vehicle_id') == vehicle_id
                                       for e in self.active_events.values()
                                       if e.event_type == EventType.EMERGENCY_VEHICLE):
                                event = TrafficEvent(
                                    event_id=event_id,
                                    event_type=EventType.EMERGENCY_VEHICLE,
                                    location=pos,
                                    affected_lanes=self._get_route_lanes(route),
                                    affected_intersections=affected_intersections,
                                    severity=1.0,
                                    timestamp=current_time,
                                    duration_estimate=self._estimate_route_time(route, pos, speed),
                                    metadata={
                                        'vehicle_id': vehicle_id,
                                        'vehicle_class': vehicle_class,
                                        'route': route,
                                        'estimated_arrival_times': self._calculate_arrival_times(route, pos, speed)
                                    }
                                )
                                self.active_events[event_id] = event
                                logger.warning(f"[EVENT] Emergency vehicle {vehicle_id} detected, creating coordination event")
                        else:
                            logger.warning(f"[EMERGENCY][SKIP] No intersections mapped for vehicle {vehicle_id} route {route}")
                except Exception as e:
                    logger.error(f"[EMERGENCY][ERROR] Vehicle {vehicle_id}: {e}")
                    continue
        except Exception as e:
            logger.error(f"[EVENT] Emergency detection failed: {e}") 
    def _detect_congestion_events(self, current_time: float):
        """Detect congestion events that require coordinated response"""
        try:
            congested_intersections = []
            
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                # Calculate intersection-level congestion severity
                severity = self._calculate_intersection_congestion(tl_id)
                
                # Track congestion history for trend analysis
                self.congestion_history[tl_id].append((current_time, severity))
                
                if severity > self.congestion_threshold:
                    congested_intersections.append((tl_id, severity))
            
            # Group nearby congested intersections
            if congested_intersections:
                congestion_clusters = self._cluster_congested_intersections(congested_intersections)
                
                for cluster_id, cluster_intersections in congestion_clusters.items():
                    if len(cluster_intersections) >= 2:  # Only coordinate if multiple intersections
                        event_id = f"congestion_{cluster_id}_{int(current_time)}"
                        
                        # Calculate cluster centroid
                        positions = [self.intersection_positions[tl_id] for tl_id, _ in cluster_intersections]
                        centroid = (
                            sum(p[0] for p in positions) / len(positions),
                            sum(p[1] for p in positions) / len(positions)
                        )
                        
                        max_severity = max(severity for _, severity in cluster_intersections)
                        affected_intersections = {tl_id for tl_id, _ in cluster_intersections}
                        
                        event = TrafficEvent(
                            event_id=event_id,
                            event_type=EventType.HEAVY_CONGESTION,
                            location=centroid,
                            affected_lanes=self._get_cluster_lanes(affected_intersections),
                            affected_intersections=affected_intersections,
                            severity=max_severity,
                            timestamp=current_time,
                            metadata={
                                'cluster_size': len(cluster_intersections),
                                'individual_severities': {tl_id: sev for tl_id, sev in cluster_intersections}
                            }
                        )
                        
                        self.active_events[event_id] = event
                        logger.warning(f"[EVENT] Congestion cluster detected: {len(cluster_intersections)} intersections")
                        
        except Exception as e:
            logger.error(f"[EVENT] Congestion detection failed: {e}")

    def _detect_spillback_events(self, current_time: float):
        """Detect spillback conditions between intersections"""
        try:
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                for lane_id in apc.lane_ids:
                    # Check if lane is experiencing spillback
                    if self._is_spillback_condition(lane_id):
                        # Find upstream intersection
                        upstream_intersection = self._find_upstream_intersection(lane_id)
                        
                        if upstream_intersection and upstream_intersection != tl_id:
                            event_id = f"spillback_{tl_id}_{upstream_intersection}_{int(current_time)}"
                            
                            # Check if we already have a recent spillback event for this pair
                            existing_event = None
                            for event in self.active_events.values():
                                if (event.event_type == EventType.SPILLBACK and
                                    event.metadata.get('downstream_intersection') == tl_id and
                                    event.metadata.get('upstream_intersection') == upstream_intersection):
                                    existing_event = event
                                    break
                            
                            if not existing_event:
                                pos = self.intersection_positions[tl_id]
                                event = TrafficEvent(
                                    event_id=event_id,
                                    event_type=EventType.SPILLBACK,
                                    location=pos,
                                    affected_lanes=[lane_id],
                                    affected_intersections={tl_id, upstream_intersection},
                                    severity=0.8,
                                    timestamp=current_time,
                                    metadata={
                                        'downstream_intersection': tl_id,
                                        'upstream_intersection': upstream_intersection,
                                        'spillback_lane': lane_id
                                    }
                                )
                                
                                self.active_events[event_id] = event
                                logger.warning(f"[EVENT] Spillback detected: {upstream_intersection} -> {tl_id}")
                                
        except Exception as e:
            logger.error(f"[EVENT] Spillback detection failed: {e}")

    def _detect_gridlock_events(self, current_time: float):
        """Detect gridlock conditions requiring immediate intervention"""
        try:
            gridlock_intersections = []
            
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                # Check for gridlock indicators
                total_waiting_time = 0
                vehicles_at_risk = 0
                
                for lane_id in apc.lane_ids:
                    waiting_time = traci.lane.getWaitingTime(lane_id)
                    total_waiting_time += waiting_time
                    
                    # Count vehicles near teleport threshold
                    for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
                        try:
                            vehicle_waiting = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                            if vehicle_waiting > 240:  # 4 minutes - approaching teleport
                                vehicles_at_risk += 1
                        except:
                            continue
                
                # Gridlock criteria: high total waiting time AND vehicles at risk of teleporting
                if total_waiting_time > 300 and vehicles_at_risk >= 3:
                    gridlock_intersections.append(tl_id)
            
            if gridlock_intersections:
                # Find connected components of gridlocked intersections
                gridlock_clusters = self._find_connected_components(gridlock_intersections)
                
                for cluster in gridlock_clusters:
                    if len(cluster) >= 1:
                        event_id = f"gridlock_{hash(tuple(sorted(cluster)))}_{int(current_time)}"
                        
                        # Calculate cluster centroid
                        positions = [self.intersection_positions[tl_id] for tl_id in cluster]
                        centroid = (
                            sum(p[0] for p in positions) / len(positions),
                            sum(p[1] for p in positions) / len(positions)
                        )
                        
                        event = TrafficEvent(
                            event_id=event_id,
                            event_type=EventType.GRIDLOCK,
                            location=centroid,
                            affected_lanes=self._get_cluster_lanes(set(cluster)),
                            affected_intersections=set(cluster),
                            severity=1.0,  # Gridlock is critical
                            timestamp=current_time,
                            metadata={'gridlock_cluster': cluster}
                        )
                        
                        self.active_events[event_id] = event
                        logger.error(f"[EVENT] GRIDLOCK detected at {len(cluster)} intersections: {cluster}")
                        
        except Exception as e:
            logger.error(f"[EVENT] Gridlock detection failed: {e}")

    def _detect_phase_failures(self, current_time: float):
        """Detect intersections with failing or stuck phases"""
        try:
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                # Check for phase stuck on low-demand lanes
                current_phase = traci.trafficlight.getPhase(tl_id)
                time_in_phase = current_time - apc.last_phase_switch_sim_time
                
                # If phase has been active for too long with no demand
                if time_in_phase > apc.max_green * 2:
                    phase_demand = apc._phase_green_total_queue(current_phase)
                    
                    if phase_demand == 0:
                        event_id = f"phase_failure_{tl_id}_{int(current_time)}"
                        
                        event = TrafficEvent(
                            event_id=event_id,
                            event_type=EventType.PHASE_FAILURE,
                            location=self.intersection_positions[tl_id],
                            affected_lanes=apc.lane_ids,
                            affected_intersections={tl_id},
                            severity=0.6,
                            timestamp=current_time,
                            metadata={
                                'stuck_phase': current_phase,
                                'time_stuck': time_in_phase,
                                'phase_demand': phase_demand
                            }
                        )
                        
                        self.active_events[event_id] = event
                        logger.warning(f"[EVENT] Phase failure detected at {tl_id}: stuck on phase {current_phase} for {time_in_phase:.1f}s")
                        
        except Exception as e:
            logger.error(f"[EVENT] Phase failure detection failed: {e}")

    def _detect_demand_surges(self, current_time: float):
        """Detect sudden demand surges that require coordination"""
        try:
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                # Check for rapid queue growth
                current_total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) 
                                        for lane in apc.lane_ids)
                
                # Compare with recent history
                if tl_id in self.vehicle_tracking:
                    prev_data = self.vehicle_tracking[tl_id]
                    prev_queue = prev_data.get('total_queue', 0)
                    time_diff = current_time - prev_data.get('last_update', current_time)
                    
                    if time_diff > 0:
                        queue_growth_rate = (current_total_queue - prev_queue) / time_diff
                        
                        # Surge criteria: rapid queue growth
                        if queue_growth_rate > 5.0 and current_total_queue > 20:
                            event_id = f"demand_surge_{tl_id}_{int(current_time)}"
                            
                            event = TrafficEvent(
                                event_id=event_id,
                                event_type=EventType.DEMAND_SURGE,
                                location=self.intersection_positions[tl_id],
                                affected_lanes=apc.lane_ids,
                                affected_intersections={tl_id},
                                severity=min(0.8, queue_growth_rate / 10.0),
                                timestamp=current_time,
                                metadata={
                                    'queue_growth_rate': queue_growth_rate,
                                    'current_queue': current_total_queue,
                                    'previous_queue': prev_queue
                                }
                            )
                            
                            self.active_events[event_id] = event
                            logger.warning(f"[EVENT] Demand surge detected at {tl_id}: growth rate {queue_growth_rate:.1f} veh/s")
                
                # Update tracking data
                self.vehicle_tracking[tl_id] = {
                    'total_queue': current_total_queue,
                    'last_update': current_time
                }
                
        except Exception as e:
            logger.error(f"[EVENT] Demand surge detection failed: {e}")

    def _update_events(self, current_time: float):
        """Update existing events - check if they're still active"""
        expired_events = []
        
        for event_id, event in self.active_events.items():
            # Check if event has expired
            if current_time - event.timestamp > self.event_timeout:
                expired_events.append(event_id)
                continue
            
            # Update event based on type
            if event.event_type == EventType.EMERGENCY_VEHICLE:
                self._update_emergency_event(event, current_time)
            elif event.event_type == EventType.HEAVY_CONGESTION:
                self._update_congestion_event(event, current_time)
            # Add other event type updates as needed
        
        # Remove expired events
        for event_id in expired_events:
            del self.active_events[event_id]
            logger.info(f"[EVENT] Expired event: {event_id}")

    def _update_intersection_groups(self, current_time: float):
        """Form or update intersection groups based on active events"""
        
        # Clear existing groups
        self.active_groups.clear()
        
        # Create groups based on active events
        for event in self.active_events.values():
            if not event.is_active:
                continue
                
            group_id = f"group_{event.event_type.value}_{hash(tuple(sorted(event.affected_intersections)))}"
            
            # Determine coordination strategy based on event type
            strategy = self._get_coordination_strategy(event.event_type)
            priority = self._get_event_priority(event.event_type)
            
            # Select group leader (intersection closest to event or with highest queue)
            leader = self._select_group_leader(event.affected_intersections, event)
            
            group = IntersectionGroup(
                group_id=group_id,
                members=event.affected_intersections.copy(),
                group_type=event.event_type.value,
                leader=leader,
                coordination_strategy=strategy,
                priority_level=priority,
                active_since=current_time,
                metadata={'source_event': event.event_id, 'event_severity': event.severity}
            )
            
            self.active_groups[group_id] = group

    def _execute_group_coordination(self, current_time: float):
        """Execute coordination strategies for active groups"""
        
        # Sort groups by priority (highest first)
        sorted_groups = sorted(self.active_groups.values(), 
                              key=lambda g: g.priority_level, reverse=True)
        
        for group in sorted_groups:
            try:
                if group.coordination_strategy == CoordinationStrategy.EMERGENCY_PREEMPTION:
                    self._execute_emergency_preemption(group, current_time)
                elif group.coordination_strategy == CoordinationStrategy.GREEN_WAVE:
                    self._execute_green_wave(group, current_time)
                elif group.coordination_strategy == CoordinationStrategy.SPILLBACK_PREVENTION:
                    self._execute_spillback_prevention(group, current_time)
                elif group.coordination_strategy == CoordinationStrategy.LOAD_BALANCING:
                    self._execute_load_balancing(group, current_time)
                elif group.coordination_strategy == CoordinationStrategy.METERING:
                    self._execute_metering_control(group, current_time)
                elif group.coordination_strategy == CoordinationStrategy.CLEARANCE:
                    self._execute_clearance_coordination(group, current_time)
                    
            except Exception as e:
                logger.error(f"[COORDINATION] Failed to execute {group.coordination_strategy} for group {group.group_id}: {e}")

    def _execute_emergency_preemption(self, group: IntersectionGroup, current_time: float):
        source_event = self.active_events.get(group.metadata['source_event'])
        if not source_event:
            return
        vehicle_id = source_event.metadata.get('vehicle_id')
        if not vehicle_id:
            return
        try:
            vehicle_pos = traci.vehicle.getPosition(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            speed = max(traci.vehicle.getSpeed(vehicle_id), 5.0)
            arrival_times = source_event.metadata.get('estimated_arrival_times', {})
            vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
            for tl_id in group.members:
                apc = self.controller.adaptive_phase_controllers.get(tl_id)
                if not apc:
                    continue
                if tl_id in arrival_times:
                    arrival_time = arrival_times[tl_id]
                    time_to_arrival = arrival_time - current_time
                    emergency_phase = apc.find_phase_for_lane(vehicle_lane)
                    if emergency_phase is None:
                        continue
                    # Start preemption if vehicle is approaching (within next 30 seconds)
                    if 0 < time_to_arrival < 30:
                        green_duration = max(20, min(60, time_to_arrival + 15))
                        # PATCHED: Use safe group phase switch
                        self._safe_group_phase_switch(tl_id, emergency_phase, requested_duration=green_duration, reason="emergency_preemption")
        except Exception as e:
            logger.error(f"[EMERGENCY] Preemption execution failed: {e}")

    def _execute_green_wave(self, group: IntersectionGroup, current_time: float):
        if not group.leader or len(group.members) < 2:
            return
        try:
            progression_speed = 13.89
            leader_pos = self.intersection_positions[group.leader]
            ordered_intersections = sorted(
                group.members,
                key=lambda tl_id: self._calculate_distance(leader_pos, self.intersection_positions[tl_id])
            )
            base_cycle_time = 90  # seconds
            for i, tl_id in enumerate(ordered_intersections):
                apc = self.controller.adaptive_phase_controllers.get(tl_id)
                if not apc:
                    continue
                if tl_id == group.leader:
                    offset = 0
                else:
                    distance = self._calculate_distance(leader_pos, self.intersection_positions[tl_id])
                    travel_time = distance / progression_speed
                    offset = travel_time % base_cycle_time
                # PATCH: Get best phase and use safe group phase switch
                best_phase = apc.find_best_phase_for_traffic()
                if best_phase is not None:
                    self._safe_group_phase_switch(tl_id, best_phase, requested_duration=apc.max_green, reason="green_wave")
                # Optionally, apply offset logic as before (timing_overrides)
        except Exception as e:
            logger.error(f"[GREEN_WAVE] Execution failed: {e}")

    def _execute_spillback_prevention(self, group: IntersectionGroup, current_time: float):
        source_event = self.active_events.get(group.metadata['source_event'])
        if not source_event:
            return
        downstream_intersection = source_event.metadata.get('downstream_intersection')
        upstream_intersection = source_event.metadata.get('upstream_intersection')
        if not (downstream_intersection and upstream_intersection):
            return
        try:
            upstream_apc = self.controller.adaptive_phase_controllers.get(upstream_intersection)
            if upstream_apc:
                # Metering: reduce all phase durations by 30%
                upstream_apc.phase_duration_multiplier = defaultdict(lambda: 0.7)
            downstream_apc = self.controller.adaptive_phase_controllers.get(downstream_intersection)
            if downstream_apc:
                spillback_lane = source_event.metadata.get('spillback_lane')
                if spillback_lane:
                    phase = downstream_apc.find_phase_for_lane(spillback_lane)
                    if phase is not None:
                        # PATCH: Use safe group phase switch
                        self._safe_group_phase_switch(downstream_intersection, phase, requested_duration=downstream_apc.max_green, reason="spillback_prevention")
                        logger.info(f"[SPILLBACK] Extended phase {phase} at {downstream_intersection} to clear {spillback_lane}")
        except Exception as e:
            logger.error(f"[SPILLBACK] Prevention execution failed: {e}")
            
    def _execute_load_balancing(self, group: IntersectionGroup, current_time: float):
        """Execute load balancing across group members"""
        try:
            # Calculate load (queue lengths) at each intersection
            intersection_loads = {}
            total_load = 0
            
            for tl_id in group.members:
                apc = self.controller.adaptive_phase_controllers.get(tl_id)
                if apc:
                    load = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in apc.lane_ids)
                    intersection_loads[tl_id] = load
                    total_load += load
            
            if total_load == 0:
                return
                
            # Calculate target load per intersection
            avg_load = total_load / len(group.members)
            
            # Identify overloaded and underloaded intersections
            overloaded = [(tl_id, load) for tl_id, load in intersection_loads.items() if load > avg_load * 1.2]
            underloaded = [(tl_id, load) for tl_id, load in intersection_loads.items() if load < avg_load * 0.8]
            
            # Adjust timing for load balancing
            for tl_id, load in overloaded:
                # Give longer greens to overloaded intersections
                multiplier = min(1.5, load / avg_load)
                self._apply_timing_multiplier(tl_id, multiplier, current_time)
                
            for tl_id, load in underloaded:
                # Give shorter greens to underloaded intersections
                multiplier = max(0.7, load / avg_load if avg_load > 0 else 0.7)
                self._apply_timing_multiplier(tl_id, multiplier, current_time)
                
        except Exception as e:
            logger.error(f"[LOAD_BALANCE] Execution failed: {e}")

    def _execute_metering_control(self, group: IntersectionGroup, current_time: float):
        """Execute metering control for congestion management"""
        try:
            for tl_id in group.members:
                # Apply conservative timing to limit throughput
                self._apply_metering_control(tl_id, 0.8, current_time)
                
        except Exception as e:
            logger.error(f"[METERING] Execution failed: {e}")
    def network_gridlock_watchdog(self):
        """Network-wide check for gridlock and deadlock, triggers unblock routines."""
        try:
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                for lane in apc.lane_ids:
                    waiting = traci.lane.getWaitingTime(lane)
                    if waiting > 240:  # approaching teleport threshold
                        # Only trigger if downstream is not also blocked
                        links = traci.lane.getLinks(lane)
                        blocked = False
                        for lk in links:
                            to_lane = lk[0]
                            if to_lane and traci.lane.getLastStepOccupancy(to_lane) < 0.6:
                                blocked = False
                            else:
                                blocked = True
                        if not blocked:
                            logger.warning(f"[GRIDLOCK WATCHDOG] Unblock request at {tl_id} for lane {lane}")
                            self.request_downstream_flush(lane)
        except Exception as e:
            logger.error(f"[NETWORK GRIDLOCK WATCHDOG] {e}")
    def _execute_clearance_coordination(self, group: IntersectionGroup, current_time: float):
        try:
            cycle_duration = 60
            current_cycle = int(current_time // cycle_duration) % len(group.members)
            active_intersection = list(group.members)[current_cycle]
            for tl_id in group.members:
                apc = self.controller.adaptive_phase_controllers.get(tl_id)
                if not apc:
                    continue
                if tl_id == active_intersection:
                    best_phase = apc.find_best_phase_for_traffic()
                    if best_phase is not None:
                        self._safe_group_phase_switch(tl_id, best_phase, requested_duration=apc.max_green, reason="clearance_coordination")
                        self._active_responses[tl_id] = "clearance"
                else:
                    self._active_responses[tl_id] = "metering"
                    cur = traci.trafficlight.getPhase(tl_id)
                    self._safe_group_phase_switch(tl_id, cur, requested_duration=apc.min_green, reason="clearance_coordination")
        except Exception as e:
            logger.error(f"[CLEARANCE] Execution failed: {e}")
    # ======================= Helper Methods =======================

    def _get_coordination_strategy(self, event_type: EventType) -> CoordinationStrategy:
        """Map event types to coordination strategies"""
        strategy_map = {
            EventType.EMERGENCY_VEHICLE: CoordinationStrategy.EMERGENCY_PREEMPTION,
            EventType.HEAVY_CONGESTION: CoordinationStrategy.LOAD_BALANCING,
            EventType.SPILLBACK: CoordinationStrategy.SPILLBACK_PREVENTION,
            EventType.GRIDLOCK: CoordinationStrategy.CLEARANCE,
            EventType.PHASE_FAILURE: CoordinationStrategy.ADAPTIVE_TIMING,
            EventType.DEMAND_SURGE: CoordinationStrategy.METERING,
        }
        return strategy_map.get(event_type, CoordinationStrategy.ADAPTIVE_TIMING)

    def _get_event_priority(self, event_type: EventType) -> int:
        """Get priority level for event type (1-5, 5=highest)"""
        priority_map = {
            EventType.EMERGENCY_VEHICLE: 5,
            EventType.GRIDLOCK: 4,
            EventType.SPILLBACK: 3,
            EventType.HEAVY_CONGESTION: 3,
            EventType.PHASE_FAILURE: 2,
            EventType.DEMAND_SURGE: 2,
        }
        return priority_map.get(event_type, 1)

    def _select_group_leader(self, intersections: Set[str], event: TrafficEvent) -> Optional[str]:
        """Select the leader intersection for a group"""
        if not intersections:
            return None
            
        if len(intersections) == 1:
            return list(intersections)[0]
            
        # For emergency events, choose closest to vehicle
        if event.event_type == EventType.EMERGENCY_VEHICLE:
            event_pos = event.location
            return min(intersections, 
                      key=lambda tl_id: self._calculate_distance(event_pos, self.intersection_positions[tl_id]))
        
        # For congestion events, choose intersection with highest queue
        if event.event_type == EventType.HEAVY_CONGESTION:
            max_queue = -1
            leader = None
            for tl_id in intersections:
                apc = self.controller.adaptive_phase_controllers.get(tl_id)
                if apc:
                    total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in apc.lane_ids)
                    if total_queue > max_queue:
                        max_queue = total_queue
                        leader = tl_id
            return leader
            
        # Default: choose first intersection
        return list(intersections)[0]

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _calculate_intersection_congestion(self, tl_id: str) -> float:
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        if not apc or not hasattr(self, 'lane_data'):
            return 0.0
        try:
            # Use lane_data dict
            total_queue = sum(self.lane_data.get(lane, {}).get('queue_length', 0) for lane in apc.lane_ids)
            total_waiting = sum(self.lane_data.get(lane, {}).get('waiting_time', 0) for lane in apc.lane_ids)
            avg_occupancy = np.mean([self.lane_data.get(lane, {}).get('density', 0) for lane in apc.lane_ids])
            # Normalize and combine metrics
            queue_score = min(1.0, total_queue / 50.0)
            waiting_score = min(1.0, total_waiting / 300.0)
            occupancy_score = min(1.0, avg_occupancy / 0.8)
            return (queue_score * 0.4 + waiting_score * 0.3 + occupancy_score * 0.3)
        except Exception:
            return 0.0


    def _cluster_congested_intersections(self, congested_intersections: List[Tuple[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """Cluster nearby congested intersections"""
        clusters = {}
        visited = set()
        
        for i, (tl_id, severity) in enumerate(congested_intersections):
            if tl_id in visited:
                continue
                
            cluster_id = f"cluster_{i}"
            cluster = [(tl_id, severity)]
            visited.add(tl_id)
            
            # Find nearby congested intersections
            for other_tl_id, other_severity in congested_intersections:
                if other_tl_id in visited:
                    continue
                    
                distance = self.distance_matrix.get((tl_id, other_tl_id), float('inf'))
                if distance <= self.coordination_radius:
                    cluster.append((other_tl_id, other_severity))
                    visited.add(other_tl_id)
            
            if len(cluster) >= self.min_group_size:
                clusters[cluster_id] = cluster
                
        return clusters

    def _is_spillback_condition(self, lane_id: str) -> bool:
        """Check if lane is experiencing spillback"""
        try:
            occupancy = traci.lane.getLastStepOccupancy(lane_id)
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            lane_length = traci.lane.getLength(lane_id)
            
            # Spillback criteria: high occupancy AND queue covers significant portion of lane
            queue_ratio = (queue_length * 7.5) / max(lane_length, 1)  # Assume 7.5m per vehicle
            
            return occupancy > self.spillback_threshold and queue_ratio > 0.6
            
        except Exception:
            return False

    def _find_upstream_intersection(self, lane_id: str) -> Optional[str]:
        """Find the upstream intersection that feeds into this lane"""
        try:
            edge_id = traci.lane.getEdgeID(lane_id)
            
            # Look for intersections that have outgoing lanes to this edge
            for tl_id, apc in self.controller.adaptive_phase_controllers.items():
                for controlled_lane in apc.lane_ids:
                    links = traci.lane.getLinks(controlled_lane)
                    for link in links:
                        if link and len(link) > 0:
                            target_lane = link[0]
                            if target_lane and traci.lane.getEdgeID(target_lane) == edge_id:
                                return tl_id
                                
            return None
            
        except Exception:
            return None

    def _find_connected_components(self, intersection_list: List[str]) -> List[List[str]]:
        """Find connected components among given intersections"""
        visited = set()
        components = []
        
        for tl_id in intersection_list:
            if tl_id in visited:
                continue
                
            component = []
            stack = [tl_id]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                    
                visited.add(current)
                component.append(current)
                
                # Add adjacent intersections that are also in the list
                for neighbor in self.adjacency_matrix.get(current, set()):
                    if neighbor in intersection_list and neighbor not in visited:
                        stack.append(neighbor)
            
            if component:
                components.append(component)
                
        return components

    def _get_intersections_on_route(self, route: List[str], current_pos: Tuple[float, float], speed: float) -> Set[str]:
        intersections = set()
        lookahead_distance = speed * 60  # 60 seconds lookahead

        for edge_id in route:
            lanes = self._get_lanes_for_edge(edge_id)
            for lane_id in lanes:
                tl_id = self.lane_to_intersection.get(lane_id)
                if tl_id:
                    tl_pos = self.intersection_positions[tl_id]
                    if self._calculate_distance(current_pos, tl_pos) <= lookahead_distance:
                        intersections.add(tl_id)
                else:
                    try:
                        nearest_tls = min(
                            self.intersection_positions,
                            key=lambda tid: self._calculate_distance(traci.lane.getShape(lane_id)[-1], self.intersection_positions[tid])
                        )
                        intersections.add(nearest_tls)
                    except Exception:
                        pass
        return intersections

    def _get_route_lanes(self, route: List[str]) -> List[str]:
        lanes = []
        for edge_id in route:
            lanes.extend(self._get_lanes_for_edge(edge_id))
        return lanes

    def _get_cluster_lanes(self, intersections: Set[str]) -> List[str]:
        """Get all lanes controlled by intersections in cluster"""
        lanes = []
        for tl_id in intersections:
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if apc:
                lanes.extend(apc.lane_ids)
        return lanes

    def _estimate_route_time(self, route: List[str], current_pos: Tuple[float, float], speed: float) -> float:
        """Estimate time to complete route"""
        try:
            total_distance = 0
            for edge_id in route:
                edge_length = traci.edge.getLength(edge_id)
                total_distance += edge_length
            
            return total_distance / max(speed, 1.0)
            
        except Exception:
            return 300.0  # Default 5 minutes

    def _calculate_arrival_times(self, route: List[str], current_pos: Tuple[float, float], speed: float) -> Dict[str, float]:
        """Calculate estimated arrival times at each intersection"""
        arrival_times = {}
        current_time = self._get_sim_time()
        
        try:
            distance_traveled = 0
            
            for edge_id in route:
                edge_length = traci.edge.getLength(edge_id)
                edge_lanes = self._get_lanes_for_edge(edge_id)
                
                for lane_id in edge_lanes:
                    tl_id = self.lane_to_intersection.get(lane_id)
                    if tl_id and tl_id not in arrival_times:
                        travel_time = distance_traveled / max(speed, 1.0)
                        arrival_times[tl_id] = current_time + travel_time
                
                distance_traveled += edge_length
                
        except Exception:
            pass
            
        return arrival_times

    def _activate_emergency_preemption(self, tl_id: str, vehicle_id: str, time_to_arrival: float):
        """Activate emergency preemption at intersection"""
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        if not apc:
            return
            
        try:
            # Find the phase that serves the emergency vehicle's approach
            vehicle_lane = traci.vehicle.getLaneID(vehicle_id)
            emergency_phase = apc.find_phase_for_lane(vehicle_lane)
            
            if emergency_phase is not None:
                # Calculate green duration based on time to arrival
                green_duration = max(20, min(60, time_to_arrival + 15))
                
                # Switch to emergency phase
                apc.set_phase_from_API(emergency_phase, requested_duration=green_duration)
                
                # Lock this phase temporarily
                lock_time = self._get_sim_time() + green_duration
                self.phase_locks[tl_id] = {
                    'phase': emergency_phase,
                    'unlock_time': lock_time,
                    'reason': 'emergency_preemption',
                    'vehicle_id': vehicle_id
                }
                
                logger.warning(f"[PREEMPTION] Activated at {tl_id} for vehicle {vehicle_id}, phase {emergency_phase}, duration {green_duration}s")
                
        except Exception as e:
            logger.error(f"[PREEMPTION] Activation failed at {tl_id}: {e}")

    def _apply_phase_offset(self, tl_id: str, offset: float, current_time: float):
        """Apply phase timing offset for green wave coordination"""
        # This is a simplified implementation - in practice, you'd need more sophisticated timing
        try:
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if apc:
                # Store timing override
                self.timing_overrides[tl_id] = {
                    'offset': offset,
                    'applied_at': current_time,
                    'reason': 'green_wave'
                }
                
                logger.info(f"[GREEN_WAVE] Applied offset {offset:.1f}s at {tl_id}")
                
        except Exception as e:
            logger.error(f"[GREEN_WAVE] Offset application failed at {tl_id}: {e}")

    def _apply_metering_control(self, tl_id: str, reduction_factor: float, current_time: float):
        """Apply metering control by reducing phase durations"""
        try:
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if apc:
                # Apply timing reduction
                self.timing_overrides[tl_id] = {
                    'reduction_factor': reduction_factor,
                    'applied_at': current_time,
                    'reason': 'metering'
                }
                
                # Reduce current phase duration if applicable
                current_phase = traci.trafficlight.getPhase(tl_id)
                try:
                    remaining_time = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    new_remaining = remaining_time * reduction_factor
                    if new_remaining > 5:  # Minimum 5 seconds
                        traci.trafficlight.setPhaseDuration(tl_id, new_remaining)
                except Exception:
                    pass
                
                logger.info(f"[METERING] Applied {reduction_factor:.1f}x timing reduction at {tl_id}")
                
        except Exception as e:
            logger.error(f"[METERING] Application failed at {tl_id}: {e}")

    def _apply_timing_multiplier(self, tl_id: str, multiplier: float, current_time: float):
        """Apply timing multiplier for load balancing"""
        try:
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if apc:
                self.timing_overrides[tl_id] = {
                    'timing_multiplier': multiplier,
                    'applied_at': current_time,
                    'reason': 'load_balancing'
                }
                
                logger.info(f"[LOAD_BALANCE] Applied {multiplier:.1f}x timing multiplier at {tl_id}")
                
        except Exception as e:
            logger.error(f"[LOAD_BALANCE] Timing multiplier failed at {tl_id}: {e}")

    def _update_emergency_event(self, event: TrafficEvent, current_time: float):
        """Update emergency vehicle event"""
        vehicle_id = event.metadata.get('vehicle_id')
        if not vehicle_id:
            event.is_active = False
            return
            
        try:
            # Check if vehicle still exists
            if vehicle_id not in traci.vehicle.getIDList():
                event.is_active = False
                return
                
            # Update vehicle position and affected intersections
            new_pos = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            
            # Update affected intersections
            event.location = new_pos
            event.affected_intersections = self._get_intersections_on_route(route, new_pos, speed)
            
            # Update arrival times
            event.metadata['estimated_arrival_times'] = self._calculate_arrival_times(route, new_pos, speed)
            
        except Exception:
            event.is_active = False

    def _update_congestion_event(self, event: TrafficEvent, current_time: float):
        """Update congestion event"""
        try:
            # Recalculate severity for affected intersections
            total_severity = 0
            active_intersections = set()
            
            for tl_id in event.affected_intersections:
                severity = self._calculate_intersection_congestion(tl_id)
                if severity > self.congestion_threshold:
                    active_intersections.add(tl_id)
                    total_severity += severity
            
            if active_intersections:
                event.affected_intersections = active_intersections
                event.severity = total_severity / len(active_intersections)
            else:
                event.is_active = False
                
        except Exception:
            event.is_active = False

    def _cleanup_expired_items(self, current_time: float):
        """Clean up expired phase locks and timing overrides"""
        # Clean up phase locks
        expired_locks = []
        for tl_id, lock_info in self.phase_locks.items():
            if current_time >= lock_info['unlock_time']:
                expired_locks.append(tl_id)
        
        for tl_id in expired_locks:
            del self.phase_locks[tl_id]
            logger.info(f"[CLEANUP] Released phase lock at {tl_id}")
        
        # Clean up timing overrides (expire after 5 minutes)
        expired_overrides = []
        for tl_id, override_info in self.timing_overrides.items():
            if current_time - override_info['applied_at'] > 300:
                expired_overrides.append(tl_id)
        
        for tl_id in expired_overrides:
            del self.timing_overrides[tl_id]
            logger.info(f"[CLEANUP] Expired timing override at {tl_id}")

    def _get_sim_time(self) -> float:
        try:
            return float(traci.simulation.getTime())
        except Exception:
            return time.time()

    def get_active_events(self) -> List[TrafficEvent]:
        return [event for event in self.active_events.values() if event.is_active]

    def get_active_groups(self) -> List[IntersectionGroup]:
        return list(self.active_groups.values())

    def is_intersection_locked(self, tl_id: str) -> bool:
        return tl_id in self.phase_locks

    def get_coordination_status(self, tl_id: str) -> Dict[str, Any]:
        status = {
            'has_phase_lock': tl_id in self.phase_locks,
            'has_timing_override': tl_id in self.timing_overrides,
            'active_groups': [],
            'active_events': []
        }
        for group in self.active_groups.values():
            if tl_id in group.members:
                status['active_groups'].append({
                    'group_id': group.group_id,
                    'group_type': group.group_type,
                    'coordination_strategy': group.coordination_strategy.value if group.coordination_strategy else None,
                    'is_leader': group.leader == tl_id
                })
        for event in self.active_events.values():
            if tl_id in event.affected_intersections and event.is_active:
                status['active_events'].append({
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'severity': event.severity
                })
        return status

    def force_event(self, event_type: EventType, location: Tuple[float, float],
                   affected_intersections: Set[str], severity: float = 1.0,
                   metadata: Optional[Dict] = None) -> str:
        event_id = f"forced_{event_type.value}_{int(self._get_sim_time())}"
        event = TrafficEvent(
            event_id=event_id,
            event_type=event_type,
            location=location,
            affected_lanes=[],
            affected_intersections=affected_intersections,
            severity=severity,
            timestamp=self._get_sim_time(),
            metadata=metadata or {}
        )
        self.active_events[event_id] = event
        logger.warning(f"[FORCED_EVENT] Created {event_type.value} event: {event_id}")
        return event_id

    def cancel_event(self, event_id: str) -> bool:
        if event_id in self.active_events:
            del self.active_events[event_id]
            logger.info(f"[CANCEL_EVENT] Cancelled event: {event_id}")
            return True
        return False