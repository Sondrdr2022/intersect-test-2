import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import traci
import numpy as np
import time
import argparse
import json
import asyncio
import websockets
import multiprocessing
from supabase import create_client

WS_URL = "ws://localhost:8000/apc/ws"
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://zckiwulodojgcfwyjrcx.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
def is_valid_phase_state(state, controlled_lanes):
    return isinstance(state, str) and len(state) == len(controlled_lanes)
def save_phase_to_supabase(tls_id, phase_idx, state, duration, controlled_lanes):
    if not is_valid_phase_state(state, controlled_lanes):
        print(f"[PHASE-PERSIST] Not saving: state length {len(state)} != lanes {len(controlled_lanes)}")
        return
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
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }).execute()
    except Exception as e:
        print(f"[PHASE-PERSIST] Save error: {e}")

def load_phases_from_supabase(tls_id, controlled_lanes):
    try:
        tls_id = str(tls_id)
        resp = supabase.table("apc_phases").select("*").eq("tls_id", tls_id).execute()
        result = {}
        for row in resp.data or []:
            if not is_valid_phase_state(row["state"], controlled_lanes):
                print(f"[PHASE-LOAD] Skipping phase {row.get('phase_idx','?')}: state length {len(row['state'])} != {len(controlled_lanes)}")
                continue
            result[int(row["phase_idx"])] = dict(
                state=row["state"], duration=float(row["duration"])
            )
        return result
    except Exception as e:
        print(f"[PHASE-PERSIST] Load error: {e}")
    return {}

def create_valid_phase_state(green_lanes, controlled_lanes):
    return ''.join(['G' if lane in green_lanes else 'r' for lane in controlled_lanes])

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
def clean_supabase_phases(tls_id, expected_lane_count):
    resp = supabase.table("apc_phases").select("*").eq("tls_id", tls_id).execute()
    for row in resp.data:
        if len(row["state"]) != expected_lane_count:
            supabase.table("apc_phases").delete().eq("tls_id", tls_id).eq("phase_idx", row["phase_idx"]).execute()
            print(f"[PHASE-CLEAN] Removed phase {row['phase_idx']} with state length {len(row['state'])}")
def apply_commands(tls_id, commands):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    phases = list(logic.phases)
    persisted = load_phases_from_supabase(tls_id, controlled_lanes)
    for cmd in commands:
        if cmd["action"] == "add_phase":
            state = cmd["state"]
            duration = cmd.get("duration", 10)
            # Ensure phase state length matches controlled lanes and correct mapping
            if len(state) != len(controlled_lanes):
                print(f"[WARNING] Phase state length mismatch: state='{state}', controlled_lanes={controlled_lanes}")
                if len(state) > len(controlled_lanes):
                    state = state[:len(controlled_lanes)]  # Truncate
                else:
                    state = state.ljust(len(controlled_lanes), 'r')
            if any(phase.state == state for phase in phases):
                continue
            from traci._trafficlight import Phase, Logic
            phases.append(Phase(duration, state))
            # Only use valid phases
            valid_phases = [p for p in phases if len(p.state) == len(controlled_lanes)]
            traci.trafficlight.setProgramLogic(
                tls_id, Logic("SUPABASE-OVERRIDE", 0, 0, valid_phases)
            )
            save_phase_to_supabase(tls_id, len(valid_phases) - 1, state, duration)
        elif cmd["action"] == "switch_phase" and cmd["phase_index"] is not None:
            idx = cmd["phase_index"]
            # Use persisted duration if available
            if idx in persisted:
                traci.trafficlight.setPhase(tls_id, idx)
                traci.trafficlight.setPhaseDuration(tls_id, persisted[idx]['duration'])
            else:
                traci.trafficlight.setPhase(tls_id, idx)
        elif cmd["action"] == "set_duration" and cmd["phase_index"] is not None and cmd["duration"] is not None:
            idx = cmd["phase_index"]
            duration = cmd["duration"]
            traci.trafficlight.setPhase(tls_id, idx)
            traci.trafficlight.setPhaseDuration(tls_id, duration)
            if idx < len(phases) and len(phases[idx].state) == len(controlled_lanes):
                save_phase_to_supabase(tls_id, idx, phases[idx].state, duration)

async def main_loop(display_queue=None):
    lane_ids = [lid for lid in traci.lane.getIDList() if not lid.startswith(":")]
    tls_id = traci.trafficlight.getIDList()[0]
    
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    clean_supabase_phases(tls_id, len(controlled_lanes))    # Preload phases from Supabase if available!
    persisted = load_phases_from_supabase(tls_id, controlled_lanes)
    if persisted:
        from traci._trafficlight import Phase, Logic
        phases = []
        for i in sorted(persisted):
            state = persisted[i]['state']
            duration = persisted[i]['duration']
            if len(state) == len(controlled_lanes):  # Only load valid phases
                phases.append(Phase(duration, state))
            else:
                print(f"[PHASE-LOAD] Skipping phase {i} due to state length mismatch: {len(state)} vs {len(controlled_lanes)}")
        if phases:
            traci.trafficlight.setProgramLogic(
                tls_id, Logic("SUPABASE-OVERRIDE", 0, 0, phases)
            )
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
            # Display code unchanged...
            if display_queue is not None:
                table_data = []
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                for phase in phases:
                    if len(phase["state"]) != len(controlled_lanes):
                        print(f"ERROR: Phase state length {len(phase['state'])} does not match number of controlled lanes {len(controlled_lanes)}")
                        # Fix or skip this phase
                unique_lanes = sorted(set(controlled_lanes), key=controlled_lanes.index)
                current_phase = traci.trafficlight.getPhase(tls_id)
                now = traci.simulation.getTime()
                next_switch = traci.trafficlight.getNextSwitch(tls_id)
                phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
                logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                phases_list = logic.phases
                phase_state = phases_list[current_phase].state if current_phase < len(phases_list) else "-"
                for lane in unique_lanes:
                    orig_idx = controlled_lanes.index(lane)
                    color = phase_state[orig_idx] if orig_idx < len(phase_state) else "-"
                    if color.upper() == 'G':
                        lane_time_left = max(0, round(next_switch - now, 2))
                    else:
                        t = next_switch
                        found = False
                        for offset in range(1, len(phases_list) + 1):
                            phase_idx = (current_phase + offset) % len(phases_list)
                            phase = phases_list[phase_idx]
                            if orig_idx < len(phase.state) and phase.state[orig_idx].upper() == 'G':
                                lane_time_left = max(0, round(t - now, 2))
                                found = True
                                break
                            t += phase.duration
                        if not found:
                            lane_time_left = "-"
                    table_data.append((
                        tls_id,
                        lane,
                        lane_time_left,
                        round(phase_duration,2) if isinstance(phase_duration, (float,int)) else phase_duration,
                        current_phase,
                        phase_state
                    ))
                try:
                    if display_queue.empty():
                        display_queue.put_nowait(table_data)
                except Exception:
                    pass
            traci.simulationStep()

def display_proc_func(display_queue):
    import tkinter as tk
    from tkinter import ttk

    class SimplePhaseTable:
        def __init__(self, queue):
            self.queue = queue
            self.root = tk.Tk()
            self.root.title("Phase Time")
            self.root.minsize(800, 200)
            self.tree = ttk.Treeview(
                self.root,
                columns=("tl_id", "lane", "time left", "phase duration", "phase index", "phase state"),
                show="headings"
            )
            for col in ["tl_id", "lane", "time left", "phase duration", "phase index", "phase state"]:
                self.tree.heading(col, text=col.replace("_", " ").title())
                self.tree.column(col, minwidth=80, width=120, anchor="center")
            self.tree.pack(fill=tk.BOTH, expand=True)
            self.no_data_label = tk.Label(self.root, text="Waiting for traffic light data...", font=("Arial", 16))
            self.no_data_label.pack(pady=20)
            self.root.after(500, self.update_table)

        def update_table(self):
            try:
                while not self.queue.empty():
                    table_data = self.queue.get_nowait()
                    for row in self.tree.get_children():
                        self.tree.delete(row)
                    for row in table_data:
                        self.tree.insert("", "end", values=row)
                    self.no_data_label.pack_forget()
            except Exception:
                pass
            self.root.after(500, self.update_table)

        def start(self):
            self.root.mainloop()

    SimplePhaseTable(display_queue).start()

def start_universal_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1):
    display_queue = multiprocessing.Queue(maxsize=1)
    display_proc = multiprocessing.Process(target=display_proc_func, args=(display_queue,))
    display_proc.start()

    def simulation_loop():
        for episode in range(episodes):
            sumo_binary = "sumo-gui" if use_gui else "sumo"
            sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary), '-c', sumocfg_path, '--start', '--quit-on-end']
            traci.start(sumo_cmd)
            asyncio.run(main_loop(display_queue=display_queue))
            traci.close()
            if episode < episodes - 1: time.sleep(2)

    try:
        simulation_loop()
    finally:
        display_proc.terminate()
        display_proc.join()

def main():
    parser = argparse.ArgumentParser(description="Run SUMO RL simulation via API controller (WebSocket)")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    args = parser.parse_args()
    start_universal_simulation(args.sumo, args.gui, args.max_steps, args.episodes)

if __name__ == "__main__":
    main()