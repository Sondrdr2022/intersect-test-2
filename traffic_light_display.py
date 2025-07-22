import tkinter as tk
from tkinter import ttk
import traci

class TrafficLightPhaseDisplay:
    def __init__(self, event_log, poll_interval=500):
        self.root = tk.Tk()
        self.root.title("Phase Time")
        self.root.minsize(1200, 200)

        self.tree = ttk.Treeview(
            self.root,
            columns=(
                "tl_id",
                "lane",
                "time left",
                "phase duration",
                "extended time",
                "event type",
                "phase index",
                "phase state",
                "action taken",
                "protected left",
                "blocked"
            ),
            show="headings"
        )
        for col in [
            "tl_id", "lane", "time left", "phase duration", "extended time", "event type", "phase index",
            "phase state", "action taken", "protected left", "blocked"
        ]:
            self.tree.heading(col, text=col.replace("_", " ").title())
            self.tree.column(col, minwidth=80, width=120, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.poll_interval = poll_interval
        self.running = False
        self.ready = True
        self.event_log = event_log
        self.no_data_label = tk.Label(self.root, text="Waiting for traffic light data...", font=("Arial", 16))
        self.no_data_label.pack(pady=20)

        self.phase_idx = None
        self.duration = None
        self.current_time = None
        self.next_switch_time = None
        self.elapsed = None
        self.remaining = None
        self.extended_time = None

    def update_table(self):
        if not getattr(self, "ready", True):
            self.root.after(self.poll_interval, self.update_table)
            return
        try:
            for row in self.tree.get_children():
                self.tree.delete(row)
            tls_list = []
            try:
                tls_list = traci.trafficlight.getIDList()
            except Exception:
                pass
            if not tls_list:
                self.no_data_label.config(text="No traffic lights found in the simulation. Waiting for data...")
                self.no_data_label.lift()
                self.root.after(self.poll_interval, self.update_table)
                return
            else:
                self.no_data_label.pack_forget()
            inserted = False
            for tl_id in tls_list:
                try:
                    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    phases = logic.getPhases()
                    if not phases:
                        continue
                    now = traci.simulation.getTime()
                    next_switch = traci.trafficlight.getNextSwitch(tl_id)
                    phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
                    curr_phase_record = None
                    for p in reversed(self.event_log):
                        if p.get("tls_id") == tl_id and p.get("phase_idx") == current_phase:
                            curr_phase_record = p
                            break
                    # PATCH: Prefer event_type and action_taken fields
                    event_type = (curr_phase_record.get("event_type") if curr_phase_record and "event_type" in curr_phase_record
                                else curr_phase_record.get("action") if curr_phase_record and "action" in curr_phase_record
                                else self.get_event_type_for(tl_id, current_phase))
                    action_taken = (curr_phase_record.get("action_taken") if curr_phase_record and "action_taken" in curr_phase_record
                                    else event_type)
                    phase_state = (curr_phase_record.get("state") if curr_phase_record and "state" in curr_phase_record
                                else (phases[current_phase].state if current_phase < len(phases) else "-"))
                    protected_left = (curr_phase_record.get("protected_left", False) if curr_phase_record
                                    else self.is_protected_left(tl_id, current_phase))
                    blocked = (curr_phase_record.get("blocked", False) if curr_phase_record
                            else self.is_blocked(tl_id, current_phase))
                    extended_time = curr_phase_record.get("extended_time", 0) if curr_phase_record else 0

                    # For each lane, show its own row
                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    unique_lanes = sorted(set(controlled_lanes), key=controlled_lanes.index)
                    for lane in unique_lanes:
                        orig_idx = controlled_lanes.index(lane)
                        color = phase_state[orig_idx] if orig_idx < len(phase_state) else "-"
                        if color.upper() == 'G':
                            lane_time_left = max(0, round(next_switch - now, 2))
                        else:
                            t = next_switch
                            found = False
                            for offset in range(1, len(phases) + 1):
                                phase_idx = (current_phase + offset) % len(phases)
                                phase = phases[phase_idx]
                                if orig_idx < len(phase.state) and phase.state[orig_idx].upper() == 'G':
                                    lane_time_left = max(0, round(t - now, 2))
                                    found = True
                                    break
                                t += phase.duration
                            if not found:
                                lane_time_left = "-"
                        self.tree.insert(
                            "", "end",
                            values=(
                                tl_id,
                                lane,
                                lane_time_left,
                                round(phase_duration, 2) if isinstance(phase_duration, (float, int)) else phase_duration,
                                round(extended_time, 2) if isinstance(extended_time, (float, int)) else extended_time,
                                event_type,
                                current_phase,
                                phase_state,
                                action_taken,
                                "Yes" if protected_left else "No",
                                "Yes" if blocked else "No"
                            )
                        )
                        inserted = True
                except Exception as e:
                    print(f"[TrafficLightPhaseDisplay ERROR]: Could not update for {tl_id}: {e}")
            if not inserted:
                self.no_data_label.config(text="Waiting for traffic light data...")
                self.no_data_label.pack(pady=20)
            else:
                self.no_data_label.pack_forget()
        except Exception as e:
            print("[TrafficLightPhaseDisplay ERROR]:", e)
        if self.running:
            self.root.after(self.poll_interval, self.update_table)
    def update_phase_duration(self, phase_idx, duration, current_time, next_switch_time, extended_time=0):
        self.phase_idx = phase_idx
        self.duration = duration
        self.current_time = current_time
        self.next_switch_time = next_switch_time
        self.elapsed = max(0, current_time - (next_switch_time - duration))
        self.remaining = max(0, next_switch_time - current_time)
        self.extended_time = extended_time  # NEW
        self.redraw()

    def redraw(self):
        self.update_table()

    def get_event_type_for(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("action", "Unknown")
        return "Unknown"

    def get_action_taken_for(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("action_taken", event.get("action", "Unknown"))
        return "Unknown"

    def is_protected_left(self, tl_id, phase_index):
        # Try event log first
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                # If log says it's protected left, trust it
                if event.get("protected_left", False):
                    return True
                # If action seems to be protected left, trust it
                if str(event.get("action", "")).lower().startswith("add_true_protected_left"):
                    return True
        # Fallback: infer from phase state (only one green, and that lane is a left turn)
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            phases = logic.getPhases()
            if phase_index < len(phases):
                phase_state = phases[phase_index].state
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                green_indices = [i for i, c in enumerate(phase_state) if c.upper() == 'G']
                if len(green_indices) == 1:
                    lane_id = controlled_lanes[green_indices[0]]
                    links = traci.lane.getLinks(lane_id)
                    is_left = any(len(link) > 6 and link[6] == 'l' for link in links)
                    return is_left
        except Exception:
            pass
        return False

    def is_blocked(self, tl_id, phase_index):
        # Try event log first
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                if event.get("blocked", False):
                    return True
        # Fallback: check if any green lane has a stopped vehicle for >5s
        try:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            phases = logic.getPhases()
            if phase_index < len(phases):
                phase_state = phases[phase_index].state
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                for idx, lane_id in enumerate(controlled_lanes):
                    if phase_state[idx].upper() == 'G':
                        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                        if vehicles:
                            front_vehicle = vehicles[0]
                            speed = traci.vehicle.getSpeed(front_vehicle)
                            stopped_time = traci.vehicle.getAccumulatedWaitingTime(front_vehicle)
                            if speed < 0.2 and stopped_time > 5:
                                return True
        except Exception:
            pass
        return False
    def start(self):
        self.running = True
        self.update_table()
        self.root.mainloop()

    def stop(self):
        self.running = False
        self.root.quit()