import os
import sys
import traci
import sumolib
import tkinter as tk
from tkinter import ttk
import datetime


class SmartIntersectionTrafficDisplay:
    def __init__(self, event_log, controller=None, poll_interval=500):
        """
        event_log: list the controller appends to (phase events with base/extended)
        controller: optional UniversalSmartTrafficController; if provided we can
                    compute elapsed/extended from APC state (more accurate).
        poll_interval: ms between UI refreshes.
        """
        self.root = tk.Tk()
        self.root.title("Smart Intersection Traffic Light Display")
        self.root.minsize(1300, 300)

        # Optional controller reference (when run from Lane7a)
        self.controller = controller

        # Columns to display
        self.columns = (
            "tl_id", "lane", "time_left", "base_duration", "extended_time",
            "phase_index", "phase_state", "queue_length", "waiting_time", "mean_speed"
        )

        self.tree = ttk.Treeview(self.root, columns=self.columns, show="headings")

        column_config = {
            "tl_id": ("TL ID", 60),
            "lane": ("Lane", 100),
            "time_left": ("Time Left", 80),
            "base_duration": ("Base Duration", 100),
            "extended_time": ("Extended Time", 100),
            "phase_index": ("Phase Index", 80),
            "phase_state": ("Phase State", 220),
            "queue_length": ("Queue Length", 100),
            "waiting_time": ("Waiting Time", 100),
            "mean_speed": ("Mean Speed", 100),
        }

        for col in self.columns:
            heading, width = column_config[col]
            self.tree.heading(col, text=heading)
            self.tree.column(col, minwidth=60, width=width, anchor="center")

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(self.root, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.poll_interval = poll_interval
        self.running = False
        self.event_log = event_log
        self.no_data_label = tk.Label(self.root, text="Waiting for traffic light data...", font=("Arial", 16))

    def get_phase_data_from_events(self, tl_id, current_phase):
        """
        Read base_duration and extended_time for (tl_id, current_phase) from event_log.
        Handles both 'tl_id' and legacy 'tls_id' keys.
        Returns (base_duration, extended_time) or (None, None) if not found.
        """
        base_duration = None
        extended_time = None
        for rec in reversed(self.event_log):
            rec_id = rec.get("tl_id", rec.get("tls_id"))
            if rec_id == tl_id and rec.get("phase_idx") == current_phase:
                base_duration = rec.get("base_duration", rec.get("duration"))
                extended_time = rec.get("extended_time")
                break
        return base_duration, extended_time

    def _format_value(self, val):
        if val is None:
            return "-"
        if isinstance(val, (int, float)):
            return round(val, 1)
        return val

    def _safe_traci(self, fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    def _compute_timing(self, tl_id, current_phase, now):
        """
        Compute time_left, base_duration, extended_time.
        Prefers using the controller's APC (accurate elapsed) if available,
        else falls back to last event record, else SUMO static duration.
        """
        next_switch = self._safe_traci(lambda: traci.trafficlight.getNextSwitch(tl_id), now)
        time_left = max(0.0, (next_switch or now) - now)

        # Try controller APC first (accurate)
        base_duration = None
        extended_time = None
        apc = None
        if self.controller is not None:
            apc = getattr(self.controller, "adaptive_phase_controllers", {}).get(tl_id)

        if apc is not None:
            try:
                pr = apc.load_phase_from_supabase(current_phase) or {}
                base_duration = float(pr.get("base_duration", pr.get("duration", 0.0)))
            except Exception:
                base_duration = None
            try:
                elapsed = max(0.0, now - float(getattr(apc, "last_phase_switch_sim_time", now)))
                total_now = elapsed + time_left
                if isinstance(base_duration, (int, float)) and base_duration > 0:
                    extended_time = max(0.0, total_now - base_duration)
            except Exception:
                pass

        # Fallback to event log
        if base_duration is None or extended_time is None:
            b2, e2 = self.get_phase_data_from_events(tl_id, current_phase)
            if base_duration is None:
                base_duration = b2
            if extended_time is None:
                extended_time = e2

        # Last resort: use SUMO static phase duration and compute extended ≈ total - base
        if base_duration is None:
            logic = self._safe_traci(lambda: traci.trafficlight.getAllProgramLogics(tl_id)[0], None)
            phases = getattr(logic, "phases", [])
            if 0 <= current_phase < len(phases):
                try:
                    base_duration = float(phases[current_phase].duration)
                except Exception:
                    base_duration = 0.0
            else:
                base_duration = 0.0

        if extended_time is None and isinstance(base_duration, (int, float)) and base_duration >= 0:
            # Estimate extended_time using total_now if we can recover elapsed from APC; else keep 0
            if apc is not None:
                try:
                    elapsed = max(0.0, now - float(getattr(apc, "last_phase_switch_sim_time", now)))
                    total_now = elapsed + time_left
                    extended_time = max(0.0, total_now - base_duration)
                except Exception:
                    extended_time = 0.0
            else:
                extended_time = 0.0

        return time_left, base_duration, extended_time

    def update_table(self):
        try:
            # Clear rows
            for row in self.tree.get_children():
                self.tree.delete(row)

            # If TraCI not running, wait
            if not self._safe_traci(lambda: traci.isLoaded(), False):
                self.show_no_data_message("SUMO not connected. Waiting for data...")
                if self.running:
                    self.root.after(self.poll_interval, self.update_table)
                return

            tls_list = self._safe_traci(lambda: traci.trafficlight.getIDList(), []) or []
            if not tls_list:
                self.show_no_data_message("No traffic lights found in the simulation. Waiting for data...")
                if self.running:
                    self.root.after(self.poll_interval, self.update_table)
                return
            else:
                self.hide_no_data_message()

            inserted_any = False
            now = self._safe_traci(lambda: traci.simulation.getTime(), 0.0)

            for tl_id in tls_list:
                try:
                    logic = self._safe_traci(lambda: traci.trafficlight.getAllProgramLogics(tl_id)[0], None)
                    phases = getattr(logic, "phases", [])
                    current_phase = self._safe_traci(lambda: traci.trafficlight.getPhase(tl_id), 0)
                    phase_state = phases[current_phase].state if (phases and 0 <= current_phase < len(phases)) else "-"

                    # Compute timing
                    time_left, base_duration, extended_time = self._compute_timing(tl_id, current_phase, now)

                    # Lanes
                    controlled_lanes = self._safe_traci(lambda: traci.trafficlight.getControlledLanes(tl_id), []) or []
                    unique_lanes = sorted(set(controlled_lanes), key=controlled_lanes.index)

                    for lane in unique_lanes:
                        # Per-lane stats
                        queue_length = self._safe_traci(lambda: traci.lane.getLastStepHaltingNumber(lane), 0)
                        waiting_time = self._safe_traci(lambda: traci.lane.getWaitingTime(lane), 0.0)
                        mean_speed = self._safe_traci(lambda: traci.lane.getLastStepMeanSpeed(lane), 0.0)

                        self.tree.insert(
                            "",
                            "end",
                            values=(
                                tl_id,
                                lane,
                                self._format_value(time_left),
                                self._format_value(base_duration),
                                self._format_value(extended_time),
                                current_phase,
                                phase_state,
                                queue_length,
                                self._format_value(waiting_time),
                                self._format_value(mean_speed),
                            ),
                        )
                        inserted_any = True

                except Exception as e:
                    print(f"[SmartIntersectionTrafficDisplay ERROR]: Could not update for {tl_id}: {e}")

            if not inserted_any:
                self.show_no_data_message("Waiting for traffic light data...")
            else:
                self.hide_no_data_message()

        except Exception as e:
            print("[SmartIntersectionTrafficDisplay ERROR]:", e)

        if self.running:
            self.root.after(self.poll_interval, self.update_table)

    def show_no_data_message(self, message):
        self.no_data_label.config(text=message)
        self.no_data_label.grid(row=2, column=0, columnspan=2, pady=20)

    def hide_no_data_message(self):
        self.no_data_label.grid_remove()

    def start(self):
        self.running = True
        self.update_table()
        self.root.mainloop()

    def stop(self):
        self.running = False
        try:
            self.root.quit()
        except Exception:
            pass


def run_simulation_step(display, event_log):
    """
    Standalone helper when running the display by itself.
    """
    try:
        if traci.isLoaded():
            traci.simulationStep()
            current_step = traci.simulation.getTime()

            # Keep last 1000 entries
            if len(event_log) > 1000:
                event_log.pop(0)

            for tl_id in traci.trafficlight.getIDList():
                current_phase = traci.trafficlight.getPhase(tl_id)
                phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
                next_switch = traci.trafficlight.getNextSwitch(tl_id)
                current_time = traci.simulation.getTime()
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                phases = logic.getPhases()
                phase_state = phases[current_phase].state if current_phase < len(phases) else "-"

                # Defaults
                base_duration = phase_duration
                extended_time = 0

                # Try event_log last entry for this (tl, phase)
                for rec in reversed(event_log):
                    rec_id = rec.get("tl_id", rec.get("tls_id"))
                    if rec_id == tl_id and rec.get("phase_idx") == current_phase:
                        base_duration = rec.get("base_duration", rec.get("duration", phase_duration))
                        extended_time = rec.get("extended_time", 0)
                        break

                event_log.append(
                    {
                        "tls_id": tl_id,
                        "phase_idx": current_phase,
                        "state": phase_state,
                        "duration": phase_duration,
                        "base_duration": base_duration,
                        "current_time": current_time,
                        "next_switch_time": next_switch,
                        "extended_time": extended_time,
                    }
                )

            # Schedule next step or stop if sim ended
            if traci.simulation.getMinExpectedNumber() > 0:
                display.root.after(50, run_simulation_step, display, event_log)
            else:
                display.stop()
        else:
            display.stop()
    except traci.exceptions.FatalTraCIError:
        print("[TraCI ERROR]: SUMO-GUI disconnected or closed.")
        display.stop()
    except Exception as e:
        print(f"[Simulation ERROR]: {e}")
        display.stop()


def run_sumo_with_display(sumocfg_file, event_log=None):
    """Run the SUMO GUI and this display standalone."""
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    sumo_binary = sumolib.checkBinary("sumo-gui")
    if not os.path.exists(sumocfg_file):
        sys.exit(f"Error: SUMO configuration file '{sumocfg_file}' not found")
    if event_log is None:
        event_log = []
    try:
        traci.start([sumo_binary, "-c", sumocfg_file, "--start", "--quit-on-end"])
        display = SmartIntersectionTrafficDisplay(event_log=event_log, controller=None, poll_interval=500)
        display.root.after(50, run_simulation_step, display, event_log)
        display.start()
    except Exception as e:
        print(f"[SUMO Simulation ERROR]: {e}")
    finally:
        if traci.isLoaded():
            traci.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sumocfg_file = sys.argv[1]
    else:
        print("Usage: python traffic_light_display.py <path_to_sumocfg_file>")
        print("Or import this module to use with your controller")
        sys.exit(1)
    run_sumo_with_display(sumocfg_file)