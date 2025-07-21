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
                "current time",
                "next",
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
            "tl_id", "current time", "next", "event type", "phase index",
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
            for tl_id in traci.trafficlight.getIDList():
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
                current_phase = traci.trafficlight.getPhase(tl_id)
                phases = logic.getPhases()
                curr_phase_record = None
                next_phase_record = None
                next_phase = (current_phase + 1) % len(phases)

                # Find latest event for current and next phase
                for p in reversed(self.event_log):
                    if p.get("tls_id") == tl_id and p.get("phase_idx") == current_phase:
                        curr_phase_record = p
                        break
                for p in reversed(self.event_log):
                    if p.get("tls_id") == tl_id and p.get("phase_idx") == next_phase:
                        next_phase_record = p
                        break

                # --- PHASE TIME CORRECTION LOGIC ---
                # Always get the actual live phase duration and time left
                # Even if event_log has a different value!
                try:
                    sim_time = traci.simulation.getTime()
                    next_switch_time = traci.trafficlight.getNextSwitch(tl_id)
                    phase_duration_live = traci.trafficlight.getPhaseDuration(tl_id)
                    time_remaining_live = max(0, next_switch_time - sim_time)
                    time_elapsed_live = max(0, phase_duration_live - time_remaining_live)
                except Exception:
                    # Defensive fallback
                    phase_duration_live = "-"
                    time_remaining_live = "-"
                    time_elapsed_live = "-"

                # For current phase, always prefer live SUMO value
                curr_time = time_elapsed_live if isinstance(time_elapsed_live, (int, float)) else phase_duration_live
                # For next phase, prefer event log, otherwise SUMO phase definition
                if next_phase_record and "duration" in next_phase_record:
                    next_time = next_phase_record["duration"]
                else:
                    try:
                        next_time = phases[next_phase].duration if next_phase < len(phases) else "-"
                    except Exception:
                        next_time = "-"

                # Try to fill in missing/correct data
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

                # Fallbacks for display
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
                        round(curr_time, 2) if isinstance(curr_time, (float, int)) else curr_time,
                        round(next_time, 2) if isinstance(next_time, (float, int)) else next_time,
                        event_type,
                        current_phase,
                        phase_state,
                        action_taken,
                        "Yes" if protected_left else "No",
                        "Yes" if blocked else "No"
                    )
                )
        except Exception as e:
            print("[TrafficLightPhaseDisplay ERROR]:", e)
        if self.running:
            self.root.after(self.poll_interval, self.update_table)    
    
    def get_event_type_for(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("action", "Unknown")
        return "Unknown"

    def update_phase_duration(self, phase_idx, duration, current_time, next_switch_time):
        """
        Update the phase time display with the latest timing info.
        """
        self.phase_idx = phase_idx
        self.duration = duration
        self.current_time = current_time
        self.next_switch_time = next_switch_time

        # Calculate elapsed and remaining time for display
        self.elapsed = max(0, current_time - (next_switch_time - duration))
        self.remaining = max(0, next_switch_time - current_time)

        self.redraw()

    def redraw(self):
        """Force the table to update with the new values."""
        self.update_table()

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