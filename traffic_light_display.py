import tkinter as tk
from tkinter import ttk
import traci

class TrafficLightPhaseDisplay:
    def __init__(self,event_log, poll_interval=1000):
        self.root = tk.Tk()
        self.root.title("Phase Time")
        # Add new columns for analytics/log info
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
        self.tree.heading("tl_id", text="tl_id")
        self.tree.heading("current time", text="current time")
        self.tree.heading("next", text="next")
        self.tree.heading("event type", text="Event Type")
        self.tree.heading("phase index", text="Phase Index")
        self.tree.heading("phase state", text="Phase State")
        self.tree.heading("action taken", text="Action Taken")
        self.tree.heading("protected left", text="Protected Left Turn")
        self.tree.heading("blocked", text="Is Blocked")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.poll_interval = poll_interval
        self.running = False
        self.ready = True  # Set to True when SUMO/TraCI is connected
        self.event_log = event_log
        

    def update_table(self):
        if not getattr(self, "ready", True):
            self.root.after(self.poll_interval, self.update_table)
            return
        try:
            # Remove old rows
            for row in self.tree.get_children():
                self.tree.delete(row)
            # Example: You need to replace the following with your actual log/event extraction logic
            for tl_id in traci.trafficlight.getIDList():
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
                current_phase = traci.trafficlight.getPhase(tl_id)
                phases = logic.getPhases()
                curr_time = phases[current_phase].duration if current_phase < len(phases) else "-"
                next_phase = (current_phase + 1) % len(phases)
                next_time = phases[next_phase].duration if next_phase < len(phases) else "-"
                
                # --- YOU MUST SET THESE FROM YOUR EVENT LOGGING SYSTEM ---
                # For demo, defaults are shown here. Replace with actual values!
                event_type = self.get_event_type_for(tl_id, current_phase)
                action_taken = self.get_action_taken_for(tl_id, current_phase)
                phase_state = phases[current_phase].state if current_phase < len(phases) else "-"
                protected_left = self.is_protected_left(tl_id, current_phase)
                blocked = self.is_blocked(tl_id, current_phase)
                # --------------------------------------------------------
                self.tree.insert(
                    "", "end",
                    values=(
                        tl_id,
                        curr_time,
                        next_time,
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

    def get_action_taken_for(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                # You can customize this logic based on your event structure
                return event.get("action_taken", event.get("action", "Unknown"))
        return "Unknown"

    def is_protected_left(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                # Heuristic: check action or a key
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