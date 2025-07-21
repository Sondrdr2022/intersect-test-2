import tkinter as tk
from tkinter import ttk
import traci

class TrafficLightPhaseDisplay:
    def __init__(self, event_log, poll_interval=1000):
        self.root = tk.Tk()
        self.root.title("Phase Time")
        self.tree = ttk.Treeview(
            self.root,
            columns=(
                "tl_id",
                "time left",
                "phase duration",
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
            "tl_id", "time left", "phase duration", "event type", "phase index",
            "phase state", "action taken", "protected left", "blocked"
        ]:
            self.tree.heading(col, text=col.replace("_", " ").title())
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.poll_interval = poll_interval
        self.running = False
        self.ready = True
        self.event_log = event_log

        # For direct phase update support:
        self.phase_idx = None
        self.duration = None
        self.current_time = None
        self.next_switch_time = None
        self.elapsed = None
        self.remaining = None

    def update_table(self):
        if not getattr(self, "ready", True):
            self.root.after(self.poll_interval, self.update_table)
            return
        try:
            for row in self.tree.get_children():
                self.tree.delete(row)
            # Defensive: Only draw if there is at least one traffic light
            tls_list = []
            try:
                tls_list = traci.trafficlight.getIDList()
            except Exception:
                pass
            if not tls_list:
                return  # No traffic lights yet in simulation
            for tl_id in tls_list:
                try:
                    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    phases = logic.getPhases()
                    if not phases:
                        continue
                    # "Time left" and "phase duration"
                    now = traci.simulation.getTime()
                    next_switch = traci.trafficlight.getNextSwitch(tl_id)
                    time_left = max(0, next_switch - now)
                    phase_duration = traci.trafficlight.getPhaseDuration(tl_id)

                    # Find latest event for current phase for this tl_id
                    curr_phase_record = None
                    for p in reversed(self.event_log):
                        if p.get("tls_id") == tl_id and p.get("phase_idx") == current_phase:
                            curr_phase_record = p
                            break

                    # Fill in event/log data as before
                    event_type = (curr_phase_record.get("action") if curr_phase_record and "action" in curr_phase_record
                                  else self.get_event_type_for(tl_id, current_phase))
                    action_taken = (curr_phase_record.get("action_taken", event_type) if curr_phase_record
                                    else self.get_action_taken_for(tl_id, current_phase))
                    phase_state = (curr_phase_record.get("state") if curr_phase_record and "state" in curr_phase_record
                                   else (phases[current_phase].state if current_phase < len(phases) else "-"))
                    protected_left = (curr_phase_record.get("protected_left", False) if curr_phase_record
                                      else self.is_protected_left(tl_id, current_phase))
                    blocked = (curr_phase_record.get("blocked", False) if curr_phase_record
                               else self.is_blocked(tl_id, current_phase))

                    if event_type is None or event_type == "":
                        event_type = "No Event"
                    if action_taken is None or action_taken == "":
                        action_taken = event_type
                    if phase_state is None or phase_state == "":
                        phase_state = phases[current_phase].state if current_phase < len(phases) else "-"

                    self.tree.insert(
                        "", "end",
                        values=(
                            tl_id,
                            round(time_left, 2) if isinstance(time_left, (float, int)) else time_left,
                            round(phase_duration, 2) if isinstance(phase_duration, (float, int)) else phase_duration,
                            event_type,
                            current_phase,
                            phase_state,
                            action_taken,
                            "Yes" if protected_left else "No",
                            "Yes" if blocked else "No"
                        )
                    )
                except Exception as e:
                    print(f"[TrafficLightPhaseDisplay ERROR]: Could not update for {tl_id}: {e}")
        except Exception as e:
            print("[TrafficLightPhaseDisplay ERROR]:", e)
        if self.running:
            self.root.after(self.poll_interval, self.update_table)

    def update_phase_duration(self, phase_idx, duration, current_time, next_switch_time):
        self.phase_idx = phase_idx
        self.duration = duration
        self.current_time = current_time
        self.next_switch_time = next_switch_time
        self.elapsed = max(0, current_time - (next_switch_time - duration))
        self.remaining = max(0, next_switch_time - current_time)
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
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("protected_left", False) or event.get("action", "").startswith("add_true_protected_left")
        return False

    def is_blocked(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("blocked", False)
        return False

    def start(self):
        self.running = True
        self.update_table()
        self.root.mainloop()

    def stop(self):
        self.running = False
        self.root.quit()