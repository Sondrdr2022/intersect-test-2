import time
import math
import numpy as np
from collections import defaultdict, deque
import traci
import logging

logger = logging.getLogger(__name__)

class CorridorCoordinator:
    """
    Advanced multi-intersection coordinator with improved congestion management:
    - Predictive congestion detection and prevention
    - Dynamic green wave coordination
    - Intelligent spillback prevention
    - Adaptive phase synchronization
    - Network-wide optimization
    """
    
    def __init__(self, controller,
                 spillback_queue_threshold=6.0,  # Lowered for earlier detection
                 downstream_pressure_threshold=4.0,  # More sensitive
                 wave_min_interval_s=8.0,  # Faster response
                 mask_ttl_s=5.0,
                 congestion_prediction_horizon=30.0):
        self.controller = controller
        self.spillback_queue_threshold = float(spillback_queue_threshold)
        self.downstream_pressure_threshold = float(downstream_pressure_threshold)
        self.wave_min_interval_s = float(wave_min_interval_s)
        self.mask_ttl_s = float(mask_ttl_s)
        self.congestion_prediction_horizon = congestion_prediction_horizon
        
        # Network topology
        self._lane_to_tl = {}
        self._tl_to_lanes = defaultdict(set)
        self._upstream_tls = defaultdict(set)  # TL -> upstream TLs
        self._downstream_tls = defaultdict(set)  # TL -> downstream TLs
        self._tl_distances = {}  # (tl1, tl2) -> distance
        
        # Coordination state
        self._last_topology_build = -1e9
        self._topology_ttl = 5.0
        self._last_wave_time = defaultdict(lambda: -1e9)
        self._last_masks = {}
        self._green_wave_active = {}
        self._congestion_clusters = []
        
        # Performance tracking
        self._tl_performance = defaultdict(lambda: {"throughput": 0, "delay": 0, "queue": 0})
        self._lane_flow_history = defaultdict(lambda: deque(maxlen=20))
        self._congestion_history = defaultdict(lambda: deque(maxlen=30))
        
        # Predictive models
        self._arrival_predictions = {}
        self._congestion_predictions = {}
        
    def update_topology(self, force=False):
        """Build comprehensive network topology including TL relationships"""
        now = traci.simulation.getTime()
        if not force and now - self._last_topology_build < self._topology_ttl:
            return
            
        self._lane_to_tl.clear()
        self._tl_to_lanes.clear()
        self._upstream_tls.clear()
        self._downstream_tls.clear()
        
        try:
            # Map lanes to TLs
            for tl_id in traci.trafficlight.getIDList():
                for lane in traci.trafficlight.getControlledLanes(tl_id):
                    self._lane_to_tl[lane] = tl_id
                    self._tl_to_lanes[tl_id].add(lane)
            
            # Build TL connectivity graph
            for tl_id in traci.trafficlight.getIDList():
                for lane in self._tl_to_lanes[tl_id]:
                    links = traci.lane.getLinks(lane) or []
                    for link in links:
                        to_lane = link[0] if link else None
                        if to_lane and to_lane in self._lane_to_tl:
                            downstream_tl = self._lane_to_tl[to_lane]
                            if downstream_tl != tl_id:
                                self._downstream_tls[tl_id].add(downstream_tl)
                                self._upstream_tls[downstream_tl].add(tl_id)
                                
                                # Calculate distance between TLs
                                try:
                                    dist = traci.lane.getLength(lane) + traci.lane.getLength(to_lane)
                                    self._tl_distances[(tl_id, downstream_tl)] = dist / 2
                                except:
                                    self._tl_distances[(tl_id, downstream_tl)] = 100  # default
                                    
        except Exception as e:
            logger.error(f"Topology update failed: {e}")
            
        self._last_topology_build = now
        
    def detect_congestion_clusters(self):
        """Identify groups of congested intersections that need coordinated response"""
        clusters = []
        visited = set()
        
        for tl_id in traci.trafficlight.getIDList():
            if tl_id in visited:
                continue
                
            severity = self._calculate_tl_congestion_severity(tl_id)
            if severity > 0.5:  # Congested
                cluster = self._expand_congestion_cluster(tl_id, visited)
                if len(cluster) > 1:
                    clusters.append(cluster)
                    
        self._congestion_clusters = clusters
        return clusters
        
    def _expand_congestion_cluster(self, start_tl, visited):
        """Find all connected congested intersections"""
        cluster = set()
        queue = [start_tl]
        
        while queue:
            tl_id = queue.pop(0)
            if tl_id in visited:
                continue
                
            severity = self._calculate_tl_congestion_severity(tl_id)
            if severity > 0.4:  # Include moderately congested
                cluster.add(tl_id)
                visited.add(tl_id)
                
                # Add connected TLs
                for neighbor in self._upstream_tls[tl_id] | self._downstream_tls[tl_id]:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        
        return cluster
        
    def _calculate_tl_congestion_severity(self, tl_id):
        """Calculate overall congestion severity for an intersection"""
        lanes = self._tl_to_lanes.get(tl_id, set())
        if not lanes:
            return 0.0
            
        severities = []
        for lane in lanes:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                occupancy = traci.lane.getLastStepOccupancy(lane)
                speed = traci.lane.getLastStepMeanSpeed(lane)
                max_speed = traci.lane.getMaxSpeed(lane)
                length = traci.lane.getLength(lane)
                
                queue_ratio = (queue * 7.5) / max(length, 1.0)
                speed_ratio = 1 - (speed / max(max_speed, 0.1))
                
                severity = (
                    0.5 * min(queue_ratio, 1.0) +
                    0.3 * min(occupancy, 1.0) +
                    0.2 * speed_ratio
                )
                severities.append(severity)
            except:
                continue
                
        return np.mean(severities) if severities else 0.0
        
    def coordinate_congestion_response(self, cluster):
        """Coordinate response for a cluster of congested intersections"""
        if not cluster:
            return
            
        logger.info(f"[CLUSTER COORDINATION] Managing congestion cluster: {cluster}")
        
        # 1. Identify bottleneck intersection (highest severity)
        bottleneck = max(cluster, key=self._calculate_tl_congestion_severity)
        
        # 2. Calculate optimal phase timings for the cluster
        phase_plans = self._calculate_cluster_phase_plans(cluster, bottleneck)
        
        # 3. Apply coordinated phase changes
        for tl_id, plan in phase_plans.items():
            self._apply_phase_plan(tl_id, plan)
            
        # 4. Setup upstream metering if needed
        self._setup_upstream_metering(bottleneck, cluster)
        
    def _calculate_cluster_phase_plans(self, cluster, bottleneck):
        """Calculate optimal phase plans for congestion cluster"""
        plans = {}
        
        for tl_id in cluster:
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if not apc:
                continue
                
            # Determine role in cluster
            is_bottleneck = (tl_id == bottleneck)
            is_upstream = bottleneck in self._downstream_tls.get(tl_id, set())
            is_downstream = bottleneck in self._upstream_tls.get(tl_id, set())
            
            if is_bottleneck:
                # Maximize throughput at bottleneck
                plan = self._create_bottleneck_plan(tl_id)
            elif is_upstream:
                # Meter flow to prevent spillback
                plan = self._create_metering_plan(tl_id, bottleneck)
            elif is_downstream:
                # Clear queues quickly
                plan = self._create_clearance_plan(tl_id)
            else:
                # Standard congestion response
                plan = self._create_congestion_plan(tl_id)
                
            plans[tl_id] = plan
            
        return plans
        
    def _create_bottleneck_plan(self, tl_id):
        """Create phase plan to maximize throughput at bottleneck"""
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        plan = {
            "type": "bottleneck",
            "min_green": max(20, apc.min_green),  # Longer greens
            "max_green": min(120, apc.max_green * 1.5),
            "priority_lanes": [],
            "mask": None
        }
        
        # Find lanes with highest queues
        lanes = self._tl_to_lanes[tl_id]
        queue_lanes = sorted(
            lanes,
            key=lambda l: traci.lane.getLastStepHaltingNumber(l),
            reverse=True
        )[:3]  # Top 3 congested lanes
        
        plan["priority_lanes"] = queue_lanes
        
        # Build phase mask favoring high-queue phases
        logic = self._get_current_logic(tl_id)
        if logic:
            mask = self._build_priority_mask(tl_id, queue_lanes, logic)
            plan["mask"] = mask
            
        return plan
        
    def _create_metering_plan(self, tl_id, bottleneck_tl):
        """Create metering plan to prevent overflow to bottleneck"""
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        
        # Calculate metering rate based on bottleneck capacity
        bottleneck_capacity = self._estimate_tl_capacity(bottleneck_tl)
        current_flow = self._estimate_tl_outflow(tl_id, bottleneck_tl)
        
        metering_factor = min(1.0, bottleneck_capacity / max(current_flow, 1.0))
        
        plan = {
            "type": "metering",
            "min_green": int(apc.min_green * metering_factor),
            "max_green": int(apc.max_green * metering_factor * 0.7),
            "metering_factor": metering_factor,
            "target_tl": bottleneck_tl
        }
        
        return plan
        
    def _create_clearance_plan(self, tl_id):
        """Create plan to quickly clear downstream queues"""
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        
        plan = {
            "type": "clearance",
            "min_green": max(5, apc.min_green // 2),  # Quick cycling
            "max_green": apc.max_green,
            "serve_all": True  # Ensure all movements get service
        }
        
        return plan
        
    def _create_congestion_plan(self, tl_id):
        """Standard congestion response plan"""
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        
        plan = {
            "type": "congestion",
            "min_green": apc.min_green,
            "max_green": min(90, apc.max_green * 1.2),
            "adaptive": True
        }
        
        return plan
        
    def _apply_phase_plan(self, tl_id, plan):
        """Apply phase plan to traffic light controller"""
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        if not apc:
            return
            
        # Update timing parameters
        if "min_green" in plan:
            apc.min_green = plan["min_green"]
        if "max_green" in plan:
            apc.max_green = plan["max_green"]
            
        # Apply phase mask if specified
        if "mask" in plan and plan["mask"]:
            apc.set_coordinator_mask(plan["mask"])
            self._last_masks[tl_id] = (traci.simulation.getTime(), plan["mask"])
            
        # Queue priority phase changes
        if "priority_lanes" in plan:
            for lane in plan["priority_lanes"]:
                phase_idx = self._find_phase_for_lane(tl_id, lane)
                if phase_idx is not None:
                    apc.request_phase_change(
                        phase_idx,
                        priority_type="heavy_congestion",
                        extension_duration=plan.get("max_green", apc.max_green)
                    )
                    break  # Queue one at a time
                    
        logger.info(f"[PHASE PLAN] Applied {plan['type']} plan to {tl_id}")
        
    def _setup_upstream_metering(self, bottleneck_tl, cluster):
        """Setup metering at upstream intersections"""
        upstream = self._upstream_tls.get(bottleneck_tl, set())
        
        for upstream_tl in upstream:
            if upstream_tl not in cluster:
                # Even non-congested upstream should meter
                plan = self._create_metering_plan(upstream_tl, bottleneck_tl)
                self._apply_phase_plan(upstream_tl, plan)
                
    def create_green_wave(self, arterial_tls, direction="forward"):
        """Create coordinated green wave along arterial"""
        if len(arterial_tls) < 2:
            return
            
        logger.info(f"[GREEN WAVE] Creating wave for {arterial_tls}")
        
        # Calculate offsets based on travel time
        base_speed = 13.89  # m/s (50 km/h)
        offsets = {}
        
        for i, tl_id in enumerate(arterial_tls):
            if i == 0:
                offsets[tl_id] = 0
            else:
                prev_tl = arterial_tls[i-1]
                dist = self._tl_distances.get((prev_tl, tl_id), 200)
                travel_time = dist / base_speed
                offsets[tl_id] = offsets[prev_tl] + travel_time
                
        # Synchronize phases
        current_time = traci.simulation.getTime()
        
        for tl_id, offset in offsets.items():
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if not apc:
                continue
                
            # Find arterial phase
            arterial_phase = self._find_arterial_phase(tl_id)
            if arterial_phase is None:
                continue
                
            # Calculate when to switch
            cycle_time = 90  # Standard cycle
            position_in_cycle = (current_time + offset) % cycle_time
            desired_green_start = offset % cycle_time
            
            if abs(position_in_cycle - desired_green_start) > 5:
                # Need to adjust
                apc.request_phase_change(
                    arterial_phase,
                    priority_type="heavy_congestion",
                    extension_duration=45  # Half cycle green
                )
                
        self._green_wave_active[tuple(arterial_tls)] = current_time
        
    def predict_congestion(self):
        """Predict future congestion and take preventive action"""
        predictions = {}
        
        for tl_id in traci.trafficlight.getIDList():
            lanes = self._tl_to_lanes[tl_id]
            
            for lane in lanes:
                try:
                    # Current state
                    current_queue = traci.lane.getLastStepHaltingNumber(lane)
                    current_vehicles = traci.lane.getLastStepVehicleNumber(lane)
                    
                    # Historical flow
                    flow_history = list(self._lane_flow_history[lane])
                    if len(flow_history) < 3:
                        continue
                        
                    # Simple prediction: trend analysis
                    recent_flow = np.mean(flow_history[-5:]) if len(flow_history) >= 5 else current_vehicles
                    older_flow = np.mean(flow_history[:-5]) if len(flow_history) > 5 else recent_flow
                    
                    flow_trend = recent_flow - older_flow
                    
                    # Predict queue in horizon seconds
                    predicted_arrivals = recent_flow + flow_trend * (self.congestion_prediction_horizon / 10)
                    predicted_departures = self._estimate_lane_capacity(lane) * 0.3  # Assume 30% green time
                    
                    predicted_queue = current_queue + predicted_arrivals - predicted_departures
                    
                    capacity = traci.lane.getLength(lane) / 7.5
                    congestion_risk = predicted_queue / max(capacity, 1.0)
                    
                    predictions[lane] = {
                        "current_queue": current_queue,
                        "predicted_queue": predicted_queue,
                        "congestion_risk": congestion_risk,
                        "will_congest": congestion_risk > 0.7
                    }
                    
                    # Record for history
                    self._lane_flow_history[lane].append(current_vehicles)
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {lane}: {e}")
                    
        self._congestion_predictions = predictions
        
        # Take preventive action
        self._act_on_predictions(predictions)
        
    def _act_on_predictions(self, predictions):
        """Take preventive action based on congestion predictions"""
        for lane, pred in predictions.items():
            if not pred["will_congest"]:
                continue
                
            tl_id = self._lane_to_tl.get(lane)
            if not tl_id:
                continue
                
            apc = self.controller.adaptive_phase_controllers.get(tl_id)
            if not apc:
                continue
                
            # Preemptively increase green time for congesting lane
            phase_idx = self._find_phase_for_lane(tl_id, lane)
            if phase_idx is not None:
                logger.info(f"[PREDICTIVE] Lane {lane} will congest (risk={pred['congestion_risk']:.2f}), acting preemptively")
                
                # Request extended green before congestion occurs
                extension = min(
                    apc.max_green,
                    pred["predicted_queue"] * 2  # 2 seconds per vehicle
                )
                
                apc.request_phase_change(
                    phase_idx,
                    priority_type="heavy_congestion",
                    extension_duration=extension
                )
                
    def optimize_network_flow(self):
        """Global optimization of network flow"""
        # Identify arterials
        arterials = self._identify_arterials()
        
        # Setup green waves on major arterials
        for arterial in arterials:
            if len(arterial) >= 3:
                self.create_green_wave(arterial)
                
        # Balance flow between parallel routes
        self._balance_parallel_routes()
        
    def _identify_arterials(self):
        """Identify major arterial corridors"""
        arterials = []
        visited = set()
        
        # Find chains of connected intersections with high flow
        for tl_id in traci.trafficlight.getIDList():
            if tl_id in visited:
                continue
                
            arterial = self._trace_arterial(tl_id, visited)
            if len(arterial) >= 3:
                arterials.append(arterial)
                
        return arterials
        
    def _trace_arterial(self, start_tl, visited):
        """Trace arterial from starting TL"""
        arterial = [start_tl]
        visited.add(start_tl)
        
        # Follow highest flow downstream
        current = start_tl
        while True:
            downstream = self._downstream_tls.get(current, set())
            if not downstream:
                break
                
            # Choose highest flow downstream
            best_flow = 0
            best_tl = None
            
            for d_tl in downstream:
                if d_tl in visited:
                    continue
                flow = self._estimate_tl_flow(d_tl)
                if flow > best_flow:
                    best_flow = flow
                    best_tl = d_tl
                    
            if best_tl and best_flow > 10:  # Minimum flow threshold
                arterial.append(best_tl)
                visited.add(best_tl)
                current = best_tl
            else:
                break
                
        return arterial
        
    def _balance_parallel_routes(self):
        """Balance traffic between parallel routes to prevent single-point congestion"""
        # This would require route choice modeling
        # For now, we can at least detect imbalances
        pass
        
    def _estimate_tl_capacity(self, tl_id):
        """Estimate intersection capacity"""
        lanes = self._tl_to_lanes.get(tl_id, set())
        total_capacity = 0
        
        for lane in lanes:
            capacity = self._estimate_lane_capacity(lane)
            total_capacity += capacity * 0.5  # Assume 50% effective green
            
        return total_capacity
        
    def _estimate_tl_outflow(self, from_tl, to_tl):
        """Estimate flow from one TL to another"""
        flow = 0
        from_lanes = self._tl_to_lanes.get(from_tl, set())
        
        for lane in from_lanes:
            links = traci.lane.getLinks(lane) or []
            for link in links:
                to_lane = link[0] if link else None
                if to_lane and self._lane_to_tl.get(to_lane) == to_tl:
                    flow += traci.lane.getLastStepVehicleNumber(lane) * 0.3
                    
        return flow
        
    def _estimate_tl_flow(self, tl_id):
        """Estimate total flow through intersection"""
        lanes = self._tl_to_lanes.get(tl_id, set())
        return sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
        
    def _estimate_lane_capacity(self, lane_id):
        try:
            length = float(traci.lane.getLength(lane_id))
        except:
            length = 100.0
        return max(1.0, length / 7.5)
        
    def _find_phase_for_lane(self, tl_id, lane_id):
        """Find phase that serves given lane"""
        apc = self.controller.adaptive_phase_controllers.get(tl_id)
        if apc:
            return apc.find_phase_for_lane(lane_id)
        return None
        
    def _find_arterial_phase(self, tl_id):
        """Find phase serving arterial movement"""
        # This would need to identify the main through movement
        # For now, return phase with most green lights
        logic = self._get_current_logic(tl_id)
        if not logic:
            return None
            
        best_phase = 0
        max_greens = 0
        
        for i, phase in enumerate(logic.phases):
            green_count = phase.state.count('G') + phase.state.count('g')
            if green_count > max_greens:
                max_greens = green_count
                best_phase = i
                
        return best_phase
        
    def _get_current_logic(self, tl_id):
        """Get current traffic light logic"""
        try:
            current_prog = traci.trafficlight.getProgram(tl_id)
            all_logics = traci.trafficlight.getAllProgramLogics(tl_id)
            for logic in all_logics:
                if logic.programID == current_prog:
                    return logic
            return all_logics[0] if all_logics else None
        except:
            return None
            
    def _build_priority_mask(self, tl_id, priority_lanes, logic):
        """Build phase mask prioritizing certain lanes"""
        if not logic:
            return None
            
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        phases = logic.phases
        mask = []
        
        for phase in phases:
            # Check if phase serves any priority lane
            serves_priority = False
            for i, lane in enumerate(controlled_lanes):
                if i < len(phase.state) and phase.state[i].upper() == 'G':
                    if lane in priority_lanes:
                        serves_priority = True
                        break
                        
            mask.append(serves_priority)
            
        # Ensure at least one phase is allowed
        if not any(mask):
            mask[0] = True
            
        return mask
        
    def _clear_stale_masks(self):
        """Clear expired coordination masks"""
        now = traci.simulation.getTime()
        for tl_id, (ts, mask) in list(self._last_masks.items()):
            if now - ts >= self.mask_ttl_s:
                apc = self.controller.adaptive_phase_controllers.get(tl_id)
                if apc:
                    apc.set_coordinator_mask(None)
                self._last_masks.pop(tl_id, None)
                
    def step(self, current_time=None):
        """Main coordination step"""
        # Update topology
        self.update_topology()
        
        # Predict future congestion
        self.predict_congestion()
        
        # Detect and manage congestion clusters
        clusters = self.detect_congestion_clusters()
        for cluster in clusters:
            self.coordinate_congestion_response(cluster)
            
        # Optimize network-wide flow
        if len(clusters) == 0:  # Only optimize when not managing congestion
            self.optimize_network_flow()
            
        # Clear expired masks
        self._clear_stale_masks()
        
        # Log performance metrics
        if int(current_time or 0) % 30 == 0:
            self._log_performance_metrics()
            
    def _log_performance_metrics(self):
        """Log coordination performance metrics"""
        total_queues = 0
        total_throughput = 0
        
        for tl_id in traci.trafficlight.getIDList():
            lanes = self._tl_to_lanes.get(tl_id, set())
            for lane in lanes:
                try:
                    total_queues += traci.lane.getLastStepHaltingNumber(lane)
                    total_throughput += traci.lane.getLastStepVehicleNumber(lane)
                except:
                    pass
                    
        logger.info(f"[CORRIDOR METRICS] Clusters: {len(self._congestion_clusters)}, "
                   f"Total Queue: {total_queues}, Throughput: {total_throughput}")
        
class IntelligentPhaseCoordinator:
    """
    Advanced phase coordination system that:
    - Prevents phase monopolization
    - Ensures fair service to all movements
    - Coordinates phases between intersections
    - Actively prevents starvation
    """
    
    def __init__(self, controller):
        self.controller = controller
        
        # Phase fairness tracking
        self.phase_service_history = defaultdict(lambda: deque(maxlen=100))
        self.phase_last_served = defaultdict(dict)  # {tl_id: {phase_idx: timestamp}}
        self.phase_total_green_time = defaultdict(dict)  # {tl_id: {phase_idx: total_seconds}}
        self.phase_activation_count = defaultdict(dict)  # {tl_id: {phase_idx: count}}
        
        # Network coordination
        self.coordinated_groups = []  # Groups of TLs that should coordinate
        self.phase_mappings = {}  # Maps compatible phases between TLs
        self.network_cycle_time = 90  # Standard network cycle
        self.cycle_position = {}  # Current position in cycle for each TL
        
        # Starvation prevention
        self.max_wait_threshold = 60  # Maximum wait before forcing service
        self.critical_wait_threshold = 90  # Critical starvation threshold
        self.lane_last_served = defaultdict(float)
        
        # Phase constraints
        self.min_phase_duration = 5
        self.max_phase_duration = 45
        self.max_consecutive_extensions = 3
        self.phase_extension_count = defaultdict(int)
        
        # Initialize network topology
        self._initialize_network_coordination()
        
    def _initialize_network_coordination(self):
        """Identify groups of intersections that should coordinate"""
        try:
            tl_list = traci.trafficlight.getIDList()
            
            # Build connectivity graph
            connections = defaultdict(set)
            for tl_id in tl_list:
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane in controlled_lanes:
                    # Find connected TLs
                    links = traci.lane.getLinks(lane)
                    for link in links:
                        if link and link[0]:
                            to_lane = link[0]
                            # Find which TL controls the destination lane
                            for other_tl in tl_list:
                                if other_tl != tl_id:
                                    other_controlled = traci.trafficlight.getControlledLanes(other_tl)
                                    if to_lane in other_controlled:
                                        connections[tl_id].add(other_tl)
                                        
            # Group connected intersections
            visited = set()
            for tl_id in tl_list:
                if tl_id not in visited:
                    group = self._find_connected_group(tl_id, connections, visited)
                    if len(group) > 1:
                        self.coordinated_groups.append(group)
                        
            logger.info(f"[COORDINATION] Identified {len(self.coordinated_groups)} coordination groups")
            for i, group in enumerate(self.coordinated_groups):
                logger.info(f"  Group {i}: {group}")
                
        except Exception as e:
            logger.error(f"Failed to initialize coordination: {e}")
            
    def _find_connected_group(self, start_tl, connections, visited):
        """Find all TLs connected to start_tl"""
        group = set()
        queue = [start_tl]
        
        while queue:
            tl = queue.pop(0)
            if tl in visited:
                continue
            visited.add(tl)
            group.add(tl)
            
            for connected in connections.get(tl, set()):
                if connected not in visited:
                    queue.append(connected)
                    
        return list(group)
        
    def should_allow_phase(self, tl_id, phase_idx):
        """Determine if a phase should be allowed to activate"""
        current_time = traci.simulation.getTime()
        
        # Check if phase has been monopolizing
        if self._is_phase_monopolizing(tl_id, phase_idx):
            logger.info(f"[PHASE BLOCKED] {tl_id} phase {phase_idx} is monopolizing")
            return False
            
        # Check for starving phases
        if self._has_starving_phases(tl_id, phase_idx):
            logger.info(f"[PHASE BLOCKED] {tl_id} has starving phases, blocking {phase_idx}")
            return False
            
        # Check network coordination
        if not self._is_coordinated_time(tl_id, phase_idx):
            return False
            
        return True
        
    def _is_phase_monopolizing(self, tl_id, phase_idx):
        """Check if a phase is taking too much time"""
        history = self.phase_service_history[tl_id]
        if len(history) < 10:
            return False
            
        # Count recent activations
        recent_count = sum(1 for p in list(history)[-20:] if p == phase_idx)
        
        # If this phase has been active more than 50% of recent time
        if recent_count > 10:
            return True
            
        # Check total green time in last cycle
        total_time = self.phase_total_green_time[tl_id].get(phase_idx, 0)
        if total_time > self.network_cycle_time * 0.5:
            return True
            
        return False
        
    def _has_starving_phases(self, tl_id, current_phase):
        """Check if other phases are starving"""
        current_time = traci.simulation.getTime()
        
        try:
            logic = self._get_current_logic(tl_id)
            if not logic:
                return False
                
            num_phases = len(logic.phases)
            
            for phase_idx in range(num_phases):
                if phase_idx == current_phase:
                    continue
                    
                # Skip yellow phases
                if 'y' in logic.phases[phase_idx].state:
                    continue
                    
                # Check when this phase was last served
                last_served = self.phase_last_served[tl_id].get(phase_idx, 0)
                wait_time = current_time - last_served
                
                # If any phase has waited too long, prioritize it
                if wait_time > self.critical_wait_threshold:
                    # Check if this phase has demand
                    if self._phase_has_demand(tl_id, phase_idx):
                        return True
                        
        except Exception as e:
            logger.error(f"Error checking starving phases: {e}")
            
        return False
        
    def _phase_has_demand(self, tl_id, phase_idx):
        """Check if a phase has waiting vehicles"""
        try:
            logic = self._get_current_logic(tl_id)
            if not logic or phase_idx >= len(logic.phases):
                return False
                
            phase_state = logic.phases[phase_idx].state
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for i, lane in enumerate(controlled_lanes):
                if i < len(phase_state) and phase_state[i].upper() == 'G':
                    # Check if this lane has waiting vehicles
                    queue = traci.lane.getLastStepHaltingNumber(lane)
                    if queue > 0:
                        return True
                        
        except Exception:
            pass
            
        return False
        
    def _is_coordinated_time(self, tl_id, phase_idx):
        """Check if this is the right time for this phase in coordination"""
        # Find which group this TL belongs to
        group = None
        for g in self.coordinated_groups:
            if tl_id in g:
                group = g
                break
                
        if not group:
            return True  # Not coordinated, allow any time
            
        # Simple coordination: offset phases to create green waves
        current_time = traci.simulation.getTime()
        cycle_position = current_time % self.network_cycle_time
        
        # Each TL in group gets a time window
        group_index = group.index(tl_id)
        window_size = self.network_cycle_time / len(group)
        window_start = group_index * window_size
        window_end = window_start + window_size
        
        # Allow phase if we're in the right window
        if window_start <= cycle_position <= window_end:
            return True
            
        return False
        
    def get_next_phase(self, tl_id):
        """Determine the best next phase for a traffic light"""
        current_time = traci.simulation.getTime()
        
        # Get all phases and their demand
        phase_scores = self._calculate_phase_scores(tl_id)
        
        # Sort by score but apply fairness rules
        sorted_phases = sorted(phase_scores.items(), key=lambda x: x[1], reverse=True)
        
        for phase_idx, score in sorted_phases:
            # Check if this phase should be allowed
            if self.should_allow_phase(tl_id, phase_idx):
                return phase_idx
                
        # If no phase is ideal, return the least recently used with demand
        return self._get_fairest_phase(tl_id)
        
    def _calculate_phase_scores(self, tl_id):
        """Calculate priority scores for each phase"""
        scores = {}
        current_time = traci.simulation.getTime()
        
        try:
            logic = self._get_current_logic(tl_id)
            if not logic:
                return scores
                
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for phase_idx, phase in enumerate(logic.phases):
                if 'y' in phase.state:
                    scores[phase_idx] = -1000  # Never select yellow
                    continue
                    
                score = 0
                lanes_served = []
                
                # Calculate demand score
                for i, lane in enumerate(controlled_lanes):
                    if i < len(phase.state) and phase.state[i].upper() == 'G':
                        lanes_served.append(lane)
                        
                        # Queue length
                        queue = traci.lane.getLastStepHaltingNumber(lane)
                        score += queue * 10
                        
                        # Waiting time
                        wait = traci.lane.getWaitingTime(lane)
                        score += wait * 0.5
                        
                        # Time since last served
                        last_served = self.lane_last_served.get(lane, 0)
                        time_waiting = current_time - last_served
                        if time_waiting > self.max_wait_threshold:
                            score += time_waiting * 2
                            
                # Penalize if recently served
                last_phase_serve = self.phase_last_served[tl_id].get(phase_idx, 0)
                time_since_served = current_time - last_phase_serve
                if time_since_served < 30:
                    score *= 0.5
                    
                # Penalize if no demand
                if not lanes_served or score == 0:
                    score = -100
                    
                scores[phase_idx] = score
                
        except Exception as e:
            logger.error(f"Error calculating phase scores: {e}")
            
        return scores
        
    def _get_fairest_phase(self, tl_id):
        """Get the phase that deserves service most based on fairness"""
        current_time = traci.simulation.getTime()
        best_phase = 0
        max_wait = 0
        
        try:
            logic = self._get_current_logic(tl_id)
            if not logic:
                return 0
                
            for phase_idx, phase in enumerate(logic.phases):
                if 'y' in phase.state:
                    continue
                    
                # Check if phase has demand
                if not self._phase_has_demand(tl_id, phase_idx):
                    continue
                    
                # Get time since last served
                last_served = self.phase_last_served[tl_id].get(phase_idx, 0)
                wait_time = current_time - last_served
                
                if wait_time > max_wait:
                    max_wait = wait_time
                    best_phase = phase_idx
                    
        except Exception:
            pass
            
        return best_phase
        
    def record_phase_activation(self, tl_id, phase_idx, duration):
        """Record that a phase has been activated"""
        current_time = traci.simulation.getTime()
        
        # Update history
        self.phase_service_history[tl_id].append(phase_idx)
        self.phase_last_served[tl_id][phase_idx] = current_time
        
        # Update total green time
        if phase_idx not in self.phase_total_green_time[tl_id]:
            self.phase_total_green_time[tl_id][phase_idx] = 0
        self.phase_total_green_time[tl_id][phase_idx] += duration
        
        # Update activation count
        if phase_idx not in self.phase_activation_count[tl_id]:
            self.phase_activation_count[tl_id][phase_idx] = 0
        self.phase_activation_count[tl_id][phase_idx] += 1
        
        # Update lanes served
        try:
            logic = self._get_current_logic(tl_id)
            if logic and phase_idx < len(logic.phases):
                phase_state = logic.phases[phase_idx].state
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                
                for i, lane in enumerate(controlled_lanes):
                    if i < len(phase_state) and phase_state[i].upper() == 'G':
                        self.lane_last_served[lane] = current_time
                        
        except Exception:
            pass
            
    def suggest_phase_duration(self, tl_id, phase_idx):
        """Suggest optimal duration for a phase"""
        try:
            # Base duration
            duration = self.min_phase_duration
            
            # Add time based on queue
            logic = self._get_current_logic(tl_id)
            if logic and phase_idx < len(logic.phases):
                phase_state = logic.phases[phase_idx].state
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                
                max_queue = 0
                for i, lane in enumerate(controlled_lanes):
                    if i < len(phase_state) and phase_state[i].upper() == 'G':
                        queue = traci.lane.getLastStepHaltingNumber(lane)
                        max_queue = max(max_queue, queue)
                        
                # 2 seconds per vehicle, but capped
                duration = min(self.max_phase_duration, 
                             max(self.min_phase_duration, max_queue * 2))
                
            # Check if this phase has been extended recently
            if self.phase_extension_count[tl_id] >= self.max_consecutive_extensions:
                duration = self.min_phase_duration
                self.phase_extension_count[tl_id] = 0
                
            return duration
            
        except Exception:
            return self.min_phase_duration
            
    def reset_cycle_metrics(self):
        """Reset metrics at the end of each cycle"""
        current_time = traci.simulation.getTime()
        
        # Reset total green times every cycle
        if int(current_time) % self.network_cycle_time == 0:
            for tl_id in self.phase_total_green_time:
                self.phase_total_green_time[tl_id].clear()
                
    def _get_current_logic(self, tl_id):
        """Get current traffic light logic"""
        try:
            current_prog = traci.trafficlight.getProgram(tl_id)
            all_logics = traci.trafficlight.getAllProgramLogics(tl_id)
            for logic in all_logics:
                if logic.programID == current_prog:
                    return logic
            return all_logics[0] if all_logics else None
        except:
            return None
            
    def enforce_phase_fairness(self, tl_id, requested_phase):
        """Override requested phase if fairness is violated"""
        current_time = traci.simulation.getTime()
        
        # Check for critical starvation
        for phase_idx in self.phase_last_served[tl_id]:
            if phase_idx == requested_phase:
                continue
                
            last_served = self.phase_last_served[tl_id][phase_idx]
            wait_time = current_time - last_served
            
            if wait_time > self.critical_wait_threshold:
                if self._phase_has_demand(tl_id, phase_idx):
                    logger.info(f"[FAIRNESS] Overriding phase {requested_phase} with starving phase {phase_idx}")
                    return phase_idx
                    
        return requested_phase