import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import traci
import numpy as np
import time
import threading
import argparse
import json
import asyncio
import websockets

WS_URL = "ws://localhost:8000/apc/ws"

def gather_lane_data(lane_ids):
    lanes = []
    for lid in lane_ids:
        queue = traci.lane.getLastStepHaltingNumber(lid)
        waiting_time = traci.lane.getWaitingTime(lid)
        mean_speed = traci.lane.getLastStepMeanSpeed(lid)
        length = max(1.0, traci.lane.getLength(lid))
        density = traci.lane.getLastStepVehicleNumber(lid) / length
        vids = list(traci.lane.getLastStepVehicleIDs(lid))
        vclasses = [traci.vehicle.getVehicleClass(vid) for vid in vids]
        links = traci.lane.getLinks(lid)
        is_left = any((len(link) > 6 and link[6] == 'l') or (len(link) > 3 and link[3] == 'l') for link in links)
        is_right = any((len(link) > 6 and link[6] == 'r') or (len(link) > 3 and link[3] == 'r') for link in links)
        lanes.append({
            "id": lid,
            "queue": queue,
            "wait": waiting_time,
            "speed": mean_speed,
            "density": density,
            "vehicle_ids": vids,
            "vehicle_classes": vclasses,
            "is_left": is_left,
            "is_right": is_right
        })
    return lanes

def gather_phase_data(tls_id):
    phases = []
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    for idx, phase in enumerate(logic.phases):
        phases.append({
            "index": idx,
            "state": phase.state,
            "duration": float(getattr(phase, "duration", 10))
        })
    return phases

def apply_commands(tls_id, commands):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = list(logic.phases)
    for cmd in commands:
        if cmd["action"] == "add_phase":
            state = cmd["state"]
            if any(phase.state == state for phase in phases): continue
            from traci._trafficlight import Phase, Logic
            phases.append(Phase(cmd["duration"], state))
            traci.trafficlight.setCompleteRedYellowGreenDefinition(
                tls_id,
                Logic(logic.programID, logic.type, len(phases) - 1, phases)
            )
        elif cmd["action"] == "switch_phase" and cmd["phase_index"] is not None:
            traci.trafficlight.setPhase(tls_id, cmd["phase_index"])
        elif cmd["action"] == "set_duration" and cmd["phase_index"] is not None and cmd["duration"] is not None:
            traci.trafficlight.setPhase(tls_id, cmd["phase_index"])
            traci.trafficlight.setPhaseDuration(tls_id, cmd["duration"])

async def main_loop():
    lane_ids = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
    tls_id = traci.trafficlight.getIDList()[0]
    async with websockets.connect(WS_URL, max_queue=None) as websocket:
        while traci.simulation.getMinExpectedNumber() > 0:
            lanes = gather_lane_data(lane_ids)
            phases = gather_phase_data(tls_id)
            req = {
                "intersection": {
                    "tls_id": tls_id,
                    "current_phase": traci.trafficlight.getPhase(tls_id),
                    "phases": phases,
                    "lanes": lanes,
                    "sim_time": traci.simulation.getTime()
                }
            }
            await websocket.send(json.dumps(req))
            data = await websocket.recv()
            data = json.loads(data)
            apply_commands(tls_id, data["commands"])
            traci.simulationStep()

def main():
    parser = argparse.ArgumentParser(description="Run SUMO RL simulation via API controller (WebSocket)")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    args = parser.parse_args()
    start_universal_simulation(args.sumo, args.gui, args.max_steps, args.episodes)

def start_universal_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1):
    def simulation_loop():
        for episode in range(episodes):
            sumo_binary = "sumo-gui" if use_gui else "sumo"
            sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary), '-c', sumocfg_path, '--start', '--quit-on-end']
            traci.start(sumo_cmd)
            asyncio.run(main_loop())
            traci.close()
            if episode < episodes - 1: time.sleep(2)
    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.start()
    sim_thread.join()

if __name__ == "__main__":
    main()