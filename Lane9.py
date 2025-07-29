import os, sys, traci, time, threading, warnings, argparse, logging
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
logger = logging.getLogger("Lane7Middleware")

os.environ.setdefault('SUMO_HOME', r'C:\\Program Files (x86)\\Eclipse\\Sumo')
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

class MiddlewarePhaseController:
    """
    Handles all phase/logic via remote API (Edge Function or Middleware). Only detection remains local.
    PATCH: Sync phases with SUMO after each API update to prevent phase index out-of-bounds errors.
    """
    def __init__(self, tls_id, lane_ids):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.current_phase_idx = 0
        self.phases = []
        self.phase_map = {}
        self.connection_active = True

    def check_connection(self):
        """Check if SUMO connection is still active"""
        try:
            traci.simulation.getTime()
            return True
        except (traci.TraCIException, traci.FatalTraCIError):
            self.connection_active = False
            return False

    def push_traffic_and_update_phases(self):
        if not self.check_connection():
            logger.error("SUMO connection lost, skipping traffic update")
            return
            
        traffic = []
        for lane_id in self.lane_ids:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                wait = traci.lane.getWaitingTime(lane_id)
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                traffic.append({"lane_id": lane_id, "queue": queue, "wait": wait, "speed": speed})
            except (traci.TraCIException, traci.FatalTraCIError):
                traffic.append({"lane_id": lane_id, "queue": 0, "wait": 0, "speed": 0})
        
        try:
            expected_state_length = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
            self.phases = post_traffic_to_api(self.tls_id, traffic, expected_state_length)
            self.phase_map = {p['state']: i for i, p in enumerate(self.phases)}
            
            # PATCH: Sync phases with SUMO logic if valid non-empty list
            if self.phases and len(self.phases) > 0:
                try:
                    self.sync_phases_with_sumo(self.tls_id, self.phases)
                except Exception as e:
                    logger.error(f"PATCH: Error syncing phases with SUMO: {e}")
            else:
                logger.error("PATCH: API returned 0 phases, skipping SUMO sync.")
        except (traci.TraCIException, traci.FatalTraCIError) as e:
            logger.error(f"Connection error during phase update: {e}")
            self.connection_active = False

    @staticmethod
    def sync_phases_with_sumo(tls_id, phases):
        from traci._trafficlight import Logic, Phase
        try:
            # Check connection before attempting sync
            traci.simulation.getTime()
            
            # Validate phases before creating TraCI phases
            valid_phases = []
            for p in phases:
                if 'duration' in p and 'state' in p and p['duration'] > 0:
                    # Ensure duration is reasonable (between 1 and 300 seconds)
                    duration = max(1, min(300, int(p['duration'])))
                    valid_phases.append(Phase(duration, p['state']))
            
            if not valid_phases:
                logger.error("PATCH: No valid phases to set in SUMO logic.")
                return
            
            # Get current logic to preserve parameters
            try:
                current_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
                program_id = current_logic.programID if hasattr(current_logic, 'programID') else "API-LOGIC"
            except:
                program_id = "API-LOGIC"
            
            new_logic = Logic(
                programID=program_id,
                type=0,  # fixed time
                currentPhaseIndex=0,
                phases=valid_phases,
            )
            
            traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_logic)
            logger.info(f"Successfully synced {len(valid_phases)} phases with SUMO")
            
        except (traci.TraCIException, traci.FatalTraCIError) as e:
            logger.error(f"PATCH: Failed to update SUMO phases: {e}")
            raise e
        except Exception as e:
            logger.error(f"PATCH: Unexpected error updating SUMO phases: {e}")
            raise e

    def get_phase_by_state(self, state_str):
        if not self.check_connection():
            return None
            
        idx = self.phase_map.get(state_str)
        if idx is not None:
            return idx
        try:
            return get_phase_by_state_from_api(self.tls_id, state_str)
        except Exception as e:
            logger.error(f"Error getting phase by state from API: {e}")
            return None

    def create_new_phase(self, state_str, duration):
        if not self.check_connection():
            return None
            
        try:
            new_phase = create_new_phase_in_api(self.tls_id, state_str, duration)
            if new_phase:
                self.phases.append(new_phase)
                self.phase_map[state_str] = len(self.phases) - 1
                # PATCH: sync with SUMO after new phase
                self.sync_phases_with_sumo(self.tls_id, self.phases)
                return len(self.phases) - 1
        except Exception as e:
            logger.error(f"Error creating new phase: {e}")
        return None

    def set_phase(self, phase_idx):
        if not self.check_connection():
            return False
            
        # PATCH: Clamp phase_idx to be within SUMO's allowed range
        try:
            phases_in_sumo = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0].getPhases()
            if not (0 <= phase_idx < len(phases_in_sumo)):
                logger.error(f"PATCH: Attempted to set out-of-bounds phase_idx {phase_idx}. Allowed range [0, {len(phases_in_sumo)-1}]")
                return False
                
            if 0 <= phase_idx < len(self.phases):
                phase = self.phases[phase_idx]
                try:
                    traci.trafficlight.setPhase(self.tls_id, phase_idx)
                    traci.trafficlight.setPhaseDuration(self.tls_id, phase['duration'])
                    self.current_phase_idx = phase_idx
                    return True
                except (traci.TraCIException, traci.FatalTraCIError) as e:
                    logger.error(f"Error setting phase: {e}")
                    self.connection_active = False
            return False
        except (traci.TraCIException, traci.FatalTraCIError) as e:
            logger.error(f"PATCH: set_phase fatal error: {e}")
            self.connection_active = False
            return False
        except Exception as e:
            logger.error(f"PATCH: set_phase unexpected error: {e}")
            return False

    def create_phase_state(self, green_lanes=None, yellow_lanes=None, red_lanes=None):
        if not self.check_connection():
            return None
            
        try:
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
        except (traci.TraCIException, traci.FatalTraCIError) as e:
            logger.error(f"Error creating phase state: {e}")
            self.connection_active = False
            return None

class RLAgent:
    """
    RL agent as before, but all APC logic is remote. Q-table is still local.
    """
    def __init__(self, state_size, action_size, controller, q_table_file="lane7_qtable.pkl"):
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
        try:
            with open(self.q_table_file, "wb") as f:
                import pickle
                pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Error saving Q-table: {e}")

    def load_q_table(self):
        try:
            with open(self.q_table_file, "rb") as f:
                import pickle
                self.q_table = pickle.load(f)
        except Exception:
            self.q_table = {}

class Lane9Controller:
    def __init__(self, tls_id, lane_ids):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.apc = MiddlewarePhaseController(tls_id, lane_ids)
        self.episode_reward = 0

        self.state_size = 6
        self.action_size = len(lane_ids) + 1  # +1 for all-red
        self.agent = RLAgent(self.state_size, self.action_size, self.apc)

    def get_state_vector(self):
        if not self.apc.check_connection():
            return np.zeros(self.state_size)
            
        traffic = []
        for lane_id in self.lane_ids:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                wait = traci.lane.getWaitingTime(lane_id)
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                traffic.append({"queue": queue, "wait": wait, "speed": speed})
            except (traci.TraCIException, traci.FatalTraCIError):
                traffic.append({"queue": 0, "wait": 0, "speed": 0})
                
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
        if not self.apc.check_connection():
            logger.error("SUMO connection lost, skipping step")
            return False
            
        try:
            # 1. Push traffic, update phases from API (APC logic in API)
            self.apc.push_traffic_and_update_phases()
            
            # 2. Get RL state
            state = self.get_state_vector()
            
            # 3. RL agent chooses action (phase)
            action = self.agent.get_action(state)
            
            # 4. Set phase via API (with PATCH: index validation)
            if self.apc.set_phase(action):
                # 5. Reward: fetch from API (implement if your API returns it; else use 0 or local calc)
                reward = self.get_api_reward()
                
                # 6. Next state for RL
                next_state = self.get_state_vector()
                
                # 7. RL Q-update
                self.agent.update(state, action, reward, next_state)
                self.episode_reward += reward
                return True
            else:
                logger.error("Failed to set phase")
                return False
        except Exception as e:
            logger.error(f"Error in run_step: {e}")
            return False

    def get_api_reward(self):
        # Optionally: implement fetching reward from a dedicated meta endpoint in the API
        # For now, return 0 (or parse from API if available)
        return 0

    def handle_special_cases(self):
        if not self.apc.check_connection():
            return
            
        # Real-time detection stays here!
        try:
            for lane_id in self.lane_ids:
                for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                    try:
                        if traci.vehicle.getVehicleClass(vid) in ['emergency', 'priority']:
                            self.handle_emergency_vehicle(lane_id)
                            return
                    except (traci.TraCIException, traci.FatalTraCIError):
                        continue
                        
            for lane_id in self.lane_ids:
                if self.is_left_turn_lane(lane_id):
                    if self.is_lane_blocked(lane_id):
                        self.handle_protected_left_turn(lane_id)
        except (traci.TraCIException, traci.FatalTraCIError) as e:
            logger.error(f"Error in special cases handling: {e}")
            self.apc.connection_active = False

    def is_left_turn_lane(self, lane_id):
        try:
            return any(len(link) > 6 and link[6] == 'l' for link in traci.lane.getLinks(lane_id))
        except (traci.TraCIException, traci.FatalTraCIError):
            return False

    def is_lane_blocked(self, lane_id):
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            if not vehicles: 
                return False
            return traci.vehicle.getSpeed(vehicles[0]) < 0.1
        except (traci.TraCIException, traci.FatalTraCIError):
            return False

    def handle_emergency_vehicle(self, lane_id):
        state_str = self.apc.create_phase_state(green_lanes=[lane_id])
        if state_str is None:
            return
            
        idx = self.apc.get_phase_by_state(state_str)
        if idx is None:
            idx = self.apc.create_new_phase(state_str, 60)
        if idx is not None:
            self.apc.set_phase(idx)

    def handle_protected_left_turn(self, lane_id):
        state_str = self.apc.create_phase_state(green_lanes=[lane_id])
        if state_str is None:
            return
            
        idx = self.apc.get_phase_by_state(state_str)
        if idx is None:
            idx = self.apc.create_new_phase(state_str, 30)
        if idx is not None:
            self.apc.set_phase(idx)

    def end_episode(self):
        self.agent.save_q_table()
        self.episode_reward = 0

def synchronize_phases_before_simulation(tls_id, apc):
    """Synchronize phases before simulation starts"""
    try:
        # 1. Fetch remote phases from API/Supabase
        remote_phases = get_phases_from_api(tls_id)
        
        # 2. If remote phases exist, update SUMO logic and local
        if remote_phases and len(remote_phases) > 0:
            try:
                apc.sync_phases_with_sumo(tls_id, remote_phases)
                apc.phases = remote_phases
                apc.phase_map = {p['state']: i for i, p in enumerate(remote_phases)}
                logger.info(f"Synchronized {len(remote_phases)} phases from API to SUMO before simulation start.")
            except Exception as e:
                logger.error(f"Failed to sync remote phases to SUMO: {e}")
                # Fall back to using SUMO's default phases
                return False
        else:
            # Optional: Push local SUMO phases to API if remote is empty
            try:
                sumo_phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].getPhases()
                phases_to_api = []
                
                for idx, ph in enumerate(sumo_phases):
                    phases_to_api.append({
                        "tls_id": tls_id,
                        "phase_idx": idx,
                        "state": ph.state,
                        "duration": getattr(ph, 'duration', 30)
                    })
                
                for p in phases_to_api:
                    create_new_phase_in_api(tls_id, p['state'], p['duration'])
                
                logger.info(f"Pushed {len(phases_to_api)} SUMO phases to API for bootstrap.")
            except Exception as e:
                logger.error(f"Failed to bootstrap API with SUMO phases: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error in synchronize_phases_before_simulation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Lane9 Middleware/Edge Function RL Controller")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Max simulation steps')
    parser.add_argument('--episodes', type=int, default=1, help='Episodes')
    args = parser.parse_args()
    
    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary), '-c', args.sumo, '--start', '--quit-on-end']

    for episode in range(args.episodes):
        logger.info(f"Starting episode {episode + 1}/{args.episodes}")
        
        try:
            traci.start(sumo_cmd)
            
            # Get traffic light and lane information
            tls_list = traci.trafficlight.getIDList()
            if not tls_list:
                logger.error("No traffic lights found in simulation")
                traci.close()
                continue
                
            tls_id = tls_list[0]
            lane_ids = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
            
            if not lane_ids:
                logger.error("No lanes found in simulation")
                traci.close()
                continue
            
            controller = Lane9Controller(tls_id, lane_ids)
            
            # PATCH: Synchronize phases before running simulation
            if not synchronize_phases_before_simulation(tls_id, controller.apc):
                logger.warning("Phase synchronization failed, continuing with default phases")
            
            step = 0
            successful_steps = 0
            
            while True:
                try:
                    # Check if simulation should continue
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        logger.info("No more vehicles expected, ending simulation")
                        break
                        
                    if args.max_steps and step >= args.max_steps:
                        logger.info(f"Reached max steps limit: {args.max_steps}")
                        break
                    
                    # Run controller step
                    if controller.run_step():
                        successful_steps += 1
                    
                    # Handle special cases
                    controller.handle_special_cases()
                    
                    # Advance simulation
                    traci.simulationStep()
                    step += 1
                    
                    # Log progress every 100 steps
                    if step % 100 == 0:
                        logger.info(f"Step {step}, successful controller steps: {successful_steps}")
                
                except (traci.TraCIException, traci.FatalTraCIError) as e:
                    logger.error(f"SUMO connection error at step {step}: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error at step {step}: {e}")
                    break
            
            logger.info(f"Episode {episode + 1} completed. Total steps: {step}, successful controller steps: {successful_steps}")
            controller.end_episode()
            
        except Exception as e:
            logger.error(f"Error in episode {episode + 1}: {e}")
        finally:
            try:
                traci.close()
            except:
                pass
        
        # Wait between episodes
        if episode < args.episodes - 1:
            logger.info("Waiting 2 seconds before next episode...")
            time.sleep(2)

if __name__ == "__main__":
    main()