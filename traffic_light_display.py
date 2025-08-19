import os
import sys
import traci
import sumolib
import tkinter as tk
from tkinter import ttk
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import tkinter.messagebox


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
        self.root.minsize(1300, 350)

        # Optional controller reference (when run from Lane7a)
        self.controller = controller

        # Historical data storage for graph
        self.history_data = {
            'time': [],
            'queue_length': [],
            'waiting_time': [],
            'mean_speed': [],
            'base_duration': [],  # Added for extension graph
            'extended_time': [],  # Added for extension graph
            'total_duration': []  # Added for extension graph
        }
        
        # Store per traffic light data
        self.tl_history_data = defaultdict(lambda: {
            'time': [],
            'queue_length': [],
            'waiting_time': [],
            'mean_speed': [],
            'base_duration': [],  # Added for per-TL extension graph
            'extended_time': [],  # Added for per-TL extension graph
            'total_duration': []  # Added for per-TL extension graph
        })

        # Columns to display
        self.columns = (
            "tl_id", "lane", "time_left", "base_duration", "extended_time",
            "phase_index", "phase_state", "queue_length", "waiting_time", "mean_speed"
        )

        # Create main frame for layout
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create button frame
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Add graph button
        self.graph_button = tk.Button(
            button_frame, 
            text="Show Analysis Graph", 
            command=self.show_analysis_graph,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        self.graph_button.pack(side=tk.LEFT, padx=5)
        
        # Add per-TL graph button
        self.tl_graph_button = tk.Button(
            button_frame,
            text="Show Per-TL Graphs",
            command=self.show_per_tl_graphs,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        self.tl_graph_button.pack(side=tk.LEFT, padx=5)
        
        # Add extension time graph button
        self.extension_graph_button = tk.Button(
            button_frame,
            text="Show Extension Time Graph",
            command=self.show_extension_time_graph,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        self.extension_graph_button.pack(side=tk.LEFT, padx=5)
        
        # Add per-TL extension graph button
        self.tl_extension_graph_button = tk.Button(
            button_frame,
            text="Show Per-TL Extension Graphs",
            command=self.show_per_tl_extension_graphs,
            bg="#9C27B0",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        self.tl_extension_graph_button.pack(side=tk.LEFT, padx=5)
        
        # Add simulation time label
        self.time_label = tk.Label(button_frame, text="Simulation Time: 0.0s", font=("Arial", 10))
        self.time_label.pack(side=tk.RIGHT, padx=5)

        self.tree = ttk.Treeview(main_frame, columns=self.columns, show="headings")

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
        v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.tree.grid(row=1, column=0, sticky="nsew")
        v_scrollbar.grid(row=1, column=1, sticky="ns")
        h_scrollbar.grid(row=2, column=0, sticky="ew")

        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.poll_interval = poll_interval
        self.running = False
        self.event_log = event_log
        self.no_data_label = tk.Label(main_frame, text="Waiting for traffic light data...", font=("Arial", 16))

    def show_extension_time_graph(self):
        """Create and display the extension time graph showing base duration and extensions."""
        if not self.history_data['time']:
            tk.messagebox.showwarning("No Data", "No simulation data available yet. Please run the simulation first.")
            return
        
        # Create new window for graph
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Phase Extension Analysis - Overall System")
        graph_window.geometry("1200x700")
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Traffic Light Phase Extension Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Extension time over time with base duration line
        if self.history_data['extended_time']:
            # Extension time
            ax1.plot(self.history_data['time'], self.history_data['extended_time'], 
                    'r-', linewidth=2, label='Extension Time', alpha=0.7)
            
            # Average base duration as flat line
            if self.history_data['base_duration']:
                avg_base = sum(self.history_data['base_duration']) / len(self.history_data['base_duration'])
                ax1.axhline(y=0, color='g', linestyle='--', linewidth=2, 
                           label='Zero Extension (Base Only)', alpha=0.5)
                
            ax1.fill_between(self.history_data['time'], 0, self.history_data['extended_time'],
                            where=[x >= 0 for x in self.history_data['extended_time']],
                            color='red', alpha=0.3, label='Extended Time')
            ax1.fill_between(self.history_data['time'], 0, self.history_data['extended_time'],
                            where=[x < 0 for x in self.history_data['extended_time']],
                            color='blue', alpha=0.3, label='Reduced Time')
            
            ax1.set_ylabel('Extension Time\n(seconds)', fontsize=10, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            ax1.set_facecolor('#f5f5f5')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Total duration vs base duration
        if self.history_data['total_duration'] and self.history_data['base_duration']:
            ax2.plot(self.history_data['time'], self.history_data['total_duration'], 
                    'b-', linewidth=2, label='Total Duration', alpha=0.7)
            ax2.plot(self.history_data['time'], self.history_data['base_duration'], 
                    'g--', linewidth=2, label='Base Duration', alpha=0.7)
            
            # Fill area between base and total
            ax2.fill_between(self.history_data['time'], 
                           self.history_data['base_duration'],
                           self.history_data['total_duration'],
                           where=[t >= b for t, b in zip(self.history_data['total_duration'], 
                                                        self.history_data['base_duration'])],
                           color='red', alpha=0.2, label='Extension Area')
            ax2.fill_between(self.history_data['time'], 
                           self.history_data['base_duration'],
                           self.history_data['total_duration'],
                           where=[t < b for t, b in zip(self.history_data['total_duration'], 
                                                       self.history_data['base_duration'])],
                           color='blue', alpha=0.2, label='Reduction Area')
            
            ax2.set_ylabel('Duration\n(seconds)', fontsize=10, fontweight='bold')
            ax2.set_xlabel('Simulation Time (seconds)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            ax2.set_facecolor('#f0f8ff')
        
        # Add statistics
        self.add_extension_statistics_to_graph(fig)
        
        plt.tight_layout()
        
        # Embed matplotlib figure in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Add toolbar for navigation
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def show_per_tl_extension_graphs(self):
        """Show extension time graphs for each traffic light separately."""
        if not self.tl_history_data:
            tk.messagebox.showwarning("No Data", "No traffic light data available yet.")
            return
        
        # Create new window
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Per Traffic Light Extension Analysis")
        graph_window.geometry("1400x800")
        
        # Create scrollable frame
        canvas = tk.Canvas(graph_window)
        scrollbar = ttk.Scrollbar(graph_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create graphs for each traffic light
        num_tls = len(self.tl_history_data)
        if num_tls > 0:
            fig = plt.figure(figsize=(14, 3 * num_tls))
            
            for idx, (tl_id, data) in enumerate(self.tl_history_data.items(), 1):
                if data['time'] and data.get('extended_time'):  # Only plot if extension data exists
                    # Create 2 subplots for each TL
                    ax1 = plt.subplot(num_tls, 2, (idx-1)*2 + 1)
                    ax2 = plt.subplot(num_tls, 2, (idx-1)*2 + 2)
                    
                    # Extension Time
                    ax1.plot(data['time'], data['extended_time'], 'r-', linewidth=1.5)
                    ax1.axhline(y=0, color='g', linestyle='--', linewidth=1, alpha=0.5)
                    ax1.fill_between(data['time'], 0, data['extended_time'],
                                    where=[x >= 0 for x in data['extended_time']],
                                    color='red', alpha=0.3)
                    ax1.fill_between(data['time'], 0, data['extended_time'],
                                    where=[x < 0 for x in data['extended_time']],
                                    color='blue', alpha=0.3)
                    ax1.set_title(f'{tl_id} - Extension Time', fontsize=10, fontweight='bold')
                    ax1.set_ylabel('Seconds')
                    ax1.grid(True, alpha=0.3)
                    
                    # Total vs Base Duration
                    if data.get('total_duration') and data.get('base_duration'):
                        ax2.plot(data['time'], data['total_duration'], 'b-', linewidth=1.5, label='Total')
                        ax2.plot(data['time'], data['base_duration'], 'g--', linewidth=1.5, label='Base')
                        ax2.set_title(f'{tl_id} - Duration Comparison', fontsize=10, fontweight='bold')
                        ax2.set_ylabel('Seconds')
                        ax2.legend(loc='upper right', fontsize=8)
                        ax2.grid(True, alpha=0.3)
                    
                    if idx == num_tls:  # Add x-label only to bottom row
                        ax1.set_xlabel('Time (s)')
                        ax2.set_xlabel('Time (s)')
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
            canvas_fig.draw()
            canvas_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def add_extension_statistics_to_graph(self, fig):
        """Add statistics text box to the extension graph."""
        if self.history_data['extended_time']:
            # Calculate statistics
            extensions = self.history_data['extended_time']
            positive_ext = [e for e in extensions if e > 0]
            negative_ext = [e for e in extensions if e < 0]
            
            avg_extension = sum(extensions) / len(extensions) if extensions else 0
            max_extension = max(extensions) if extensions else 0
            min_extension = min(extensions) if extensions else 0
            
            # Percentage of time extended vs reduced
            pct_extended = (len(positive_ext) / len(extensions) * 100) if extensions else 0
            pct_reduced = (len(negative_ext) / len(extensions) * 100) if extensions else 0
            pct_baseline = 100 - pct_extended - pct_reduced
            
            # Average base and total durations
            avg_base = sum(self.history_data['base_duration']) / len(self.history_data['base_duration']) \
                      if self.history_data['base_duration'] else 0
            avg_total = sum(self.history_data['total_duration']) / len(self.history_data['total_duration']) \
                       if self.history_data['total_duration'] else 0
            
            # Create statistics text
            stats_text = (
                f"📊 EXTENSION STATISTICS\n"
                f"─────────────────────\n"
                f"Extension Time:\n"
                f"  • Average: {avg_extension:+.1f} seconds\n"
                f"  • Maximum: {max_extension:+.1f} seconds\n"
                f"  • Minimum: {min_extension:+.1f} seconds\n\n"
                f"Phase Adjustments:\n"
                f"  • Extended: {pct_extended:.1f}%\n"
                f"  • Reduced: {pct_reduced:.1f}%\n"
                f"  • Baseline: {pct_baseline:.1f}%\n\n"
                f"Duration Averages:\n"
                f"  • Base: {avg_base:.1f} seconds\n"
                f"  • Total: {avg_total:.1f} seconds\n"
                f"  • Difference: {avg_total - avg_base:+.1f} seconds"
            )
            
            # Add text box
            fig.text(0.98, 0.5, stats_text, transform=fig.transFigure,
                    fontsize=9, verticalalignment='center',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def show_analysis_graph(self):
        """Create and display the analysis graph with queue length, waiting time, and mean speed."""
        if not self.history_data['time']:
            tk.messagebox.showwarning("No Data", "No simulation data available yet. Please run the simulation first.")
            return
        
        # Create new window for graph
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Traffic Analysis Graph - Overall System Performance")
        graph_window.geometry("1200x700")
        
        # Create matplotlib figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Traffic Light System Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot Queue Length
        ax1.plot(self.history_data['time'], self.history_data['queue_length'], 
                'b-', linewidth=2, label='Queue Length')
        ax1.set_ylabel('Queue Length\n(vehicles)', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_facecolor('#f0f8ff')
        
        # Plot Waiting Time
        ax2.plot(self.history_data['time'], self.history_data['waiting_time'], 
                'r-', linewidth=2, label='Waiting Time')
        ax2.set_ylabel('Waiting Time\n(seconds)', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_facecolor('#fff0f5')
        
        # Plot Mean Speed
        ax3.plot(self.history_data['time'], self.history_data['mean_speed'], 
                'g-', linewidth=2, label='Mean Speed')
        ax3.set_ylabel('Mean Speed\n(m/s)', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Simulation Time (seconds)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.set_facecolor('#f0fff0')
        
        # Add statistics
        self.add_statistics_to_graph(fig)
        
        plt.tight_layout()
        
        # Embed matplotlib figure in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Add toolbar for navigation
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def show_per_tl_graphs(self):
        """Show graphs for each traffic light separately."""
        if not self.tl_history_data:
            tk.messagebox.showwarning("No Data", "No traffic light data available yet.")
            return
        
        # Create new window
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Per Traffic Light Analysis")
        graph_window.geometry("1400x800")
        
        # Create scrollable frame
        canvas = tk.Canvas(graph_window)
        scrollbar = ttk.Scrollbar(graph_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create graphs for each traffic light
        num_tls = len(self.tl_history_data)
        if num_tls > 0:
            fig = plt.figure(figsize=(14, 4 * num_tls))
            
            for idx, (tl_id, data) in enumerate(self.tl_history_data.items(), 1):
                if data['time']:  # Only plot if data exists
                    # Create 3 subplots for each TL
                    ax1 = plt.subplot(num_tls, 3, (idx-1)*3 + 1)
                    ax2 = plt.subplot(num_tls, 3, (idx-1)*3 + 2)
                    ax3 = plt.subplot(num_tls, 3, (idx-1)*3 + 3)
                    
                    # Queue Length
                    ax1.plot(data['time'], data['queue_length'], 'b-', linewidth=1.5)
                    ax1.set_title(f'{tl_id} - Queue Length', fontsize=10, fontweight='bold')
                    ax1.set_ylabel('Vehicles')
                    ax1.grid(True, alpha=0.3)
                    
                    # Waiting Time
                    ax2.plot(data['time'], data['waiting_time'], 'r-', linewidth=1.5)
                    ax2.set_title(f'{tl_id} - Waiting Time', fontsize=10, fontweight='bold')
                    ax2.set_ylabel('Seconds')
                    ax2.grid(True, alpha=0.3)
                    
                    # Mean Speed
                    ax3.plot(data['time'], data['mean_speed'], 'g-', linewidth=1.5)
                    ax3.set_title(f'{tl_id} - Mean Speed', fontsize=10, fontweight='bold')
                    ax3.set_ylabel('m/s')
                    ax3.grid(True, alpha=0.3)
                    
                    if idx == num_tls:  # Add x-label only to bottom row
                        ax1.set_xlabel('Time (s)')
                        ax2.set_xlabel('Time (s)')
                        ax3.set_xlabel('Time (s)')
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
            canvas_fig.draw()
            canvas_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def add_statistics_to_graph(self, fig):
        """Add statistics text box to the graph."""
        if self.history_data['queue_length']:
            # Calculate statistics
            avg_queue = sum(self.history_data['queue_length']) / len(self.history_data['queue_length'])
            max_queue = max(self.history_data['queue_length'])
            min_queue = min(self.history_data['queue_length'])
            
            avg_wait = sum(self.history_data['waiting_time']) / len(self.history_data['waiting_time'])
            max_wait = max(self.history_data['waiting_time'])
            
            avg_speed = sum(self.history_data['mean_speed']) / len(self.history_data['mean_speed'])
            min_speed = min(self.history_data['mean_speed'])
            max_speed = max(self.history_data['mean_speed'])
            
            # Create statistics text
            stats_text = (
                f"📊 STATISTICS\n"
                f"─────────────\n"
                f"Queue Length:\n"
                f"  • Avg: {avg_queue:.1f} vehicles\n"
                f"  • Max: {max_queue:.1f} vehicles\n"
                f"  • Min: {min_queue:.1f} vehicles\n\n"
                f"Waiting Time:\n"
                f"  • Avg: {avg_wait:.1f} seconds\n"
                f"  • Max: {max_wait:.1f} seconds\n\n"
                f"Mean Speed:\n"
                f"  • Avg: {avg_speed:.1f} m/s\n"
                f"  • Min: {min_speed:.1f} m/s\n"
                f"  • Max: {max_speed:.1f} m/s"
            )
            
            # Add text box
            fig.text(0.98, 0.5, stats_text, transform=fig.transFigure,
                    fontsize=9, verticalalignment='center',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def collect_historical_data(self, sim_time, total_queue, total_wait, avg_speed,
                               avg_base_duration=None, avg_extended_time=None, avg_total_duration=None):
        """Collect data points for historical graph including extension data."""
        self.history_data['time'].append(sim_time)
        self.history_data['queue_length'].append(total_queue)
        self.history_data['waiting_time'].append(total_wait)
        self.history_data['mean_speed'].append(avg_speed)
        
        # Add extension data if provided
        if avg_base_duration is not None:
            self.history_data['base_duration'].append(avg_base_duration)
        if avg_extended_time is not None:
            self.history_data['extended_time'].append(avg_extended_time)
        if avg_total_duration is not None:
            self.history_data['total_duration'].append(avg_total_duration)
        
        # Limit data points to prevent memory issues (keep last 10000 points)
        max_points = 10000
        if len(self.history_data['time']) > max_points:
            for key in self.history_data:
                self.history_data[key] = self.history_data[key][-max_points:]

    def collect_tl_historical_data(self, tl_id, sim_time, queue, wait, speed,
                                   base_duration=None, extended_time=None, total_duration=None):
        """Collect per-traffic-light historical data including extension data."""
        self.tl_history_data[tl_id]['time'].append(sim_time)
        self.tl_history_data[tl_id]['queue_length'].append(queue)
        self.tl_history_data[tl_id]['waiting_time'].append(wait)
        self.tl_history_data[tl_id]['mean_speed'].append(speed)
        
        # Add extension data if provided
        if base_duration is not None:
            self.tl_history_data[tl_id]['base_duration'].append(base_duration)
        if extended_time is not None:
            self.tl_history_data[tl_id]['extended_time'].append(extended_time)
        if total_duration is not None:
            self.tl_history_data[tl_id]['total_duration'].append(total_duration)
        
        # Limit data points
        max_points = 5000
        if len(self.tl_history_data[tl_id]['time']) > max_points:
            for key in self.tl_history_data[tl_id]:
                self.tl_history_data[tl_id][key] = self.tl_history_data[tl_id][key][-max_points:]

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

        PATCH: Show negative extended_time for shortened phases.
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
                    # PATCH: Allow negative extended_time (decreased time)
                    extended_time = total_now - base_duration
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

        # PATCH: Allow negative extended_time (decreased time)
        if extended_time is None and isinstance(base_duration, (int, float)) and base_duration >= 0:
            if apc is not None:
                try:
                    elapsed = max(0.0, now - float(getattr(apc, "last_phase_switch_sim_time", now)))
                    total_now = elapsed + time_left
                    extended_time = total_now - base_duration
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
            
            # Update time label
            self.time_label.config(text=f"Simulation Time: {now:.1f}s")
            
            # Variables for overall system metrics
            total_queue_sum = 0
            total_wait_sum = 0
            total_speed_sum = 0
            total_base_duration = 0
            total_extended_time = 0
            total_duration_sum = 0
            total_count = 0

            for tl_id in tls_list:
                try:
                    logic = self._safe_traci(lambda: traci.trafficlight.getAllProgramLogics(tl_id)[0], None)
                    phases = getattr(logic, "phases", [])
                    current_phase = self._safe_traci(lambda: traci.trafficlight.getPhase(tl_id), 0)
                    phase_state = phases[current_phase].state if (phases and 0 <= current_phase < len(phases)) else "-"

                    # Compute timing
                    time_left, base_duration, extended_time = self._compute_timing(tl_id, current_phase, now)

                    # Lanes, but only aggregate/summarize
                    controlled_lanes = self._safe_traci(lambda: traci.trafficlight.getControlledLanes(tl_id), []) or []
                    unique_lanes = sorted(set(controlled_lanes), key=controlled_lanes.index)

                    lane_summary = ", ".join(unique_lanes) if unique_lanes else "-"
                    lane_count = len(unique_lanes)
                    lane_display = lane_summary  # or str(lane_count)

                    # Aggregate stats
                    queue_lengths = [self._safe_traci(lambda lid=lid: traci.lane.getLastStepHaltingNumber(lid), 0) for lid in unique_lanes] if unique_lanes else [0]
                    waiting_times = [self._safe_traci(lambda lid=lid: traci.lane.getWaitingTime(lid), 0.0) for lid in unique_lanes] if unique_lanes else [0.0]
                    mean_speeds = [self._safe_traci(lambda lid=lid: traci.lane.getLastStepMeanSpeed(lid), 0.0) for lid in unique_lanes] if unique_lanes else [0.0]

                    queue_length = sum(queue_lengths) / lane_count if lane_count else 0
                    waiting_time = sum(waiting_times) / lane_count if lane_count else 0
                    mean_speed = sum(mean_speeds) / lane_count if lane_count else 0
                    
                    # Calculate total duration
                    total_duration = (base_duration + extended_time) if base_duration is not None and extended_time is not None else base_duration
                    
                    # Collect per-TL historical data with extension info
                    self.collect_tl_historical_data(
                        tl_id, now, queue_length, waiting_time, mean_speed,
                        base_duration, extended_time, total_duration
                    )
                    
                    # Add to totals for overall metrics
                    total_queue_sum += queue_length
                    total_wait_sum += waiting_time
                    total_speed_sum += mean_speed
                    if base_duration is not None:
                        total_base_duration += base_duration
                    if extended_time is not None:
                        total_extended_time += extended_time
                    if total_duration is not None:
                        total_duration_sum += total_duration
                    total_count += 1

                    self.tree.insert(
                        "",
                        "end",
                        values=(
                            tl_id,
                            lane_display,
                            self._format_value(time_left),
                            self._format_value(base_duration),
                            self._format_value(extended_time),
                            current_phase,
                            phase_state,
                            self._format_value(queue_length),
                            self._format_value(waiting_time),
                            self._format_value(mean_speed),
                        ),
                    )
                    inserted_any = True

                except Exception as e:
                    print(f"[SmartIntersectionTrafficDisplay ERROR]: Could not update for {tl_id}: {e}")

            # Collect overall system historical data with extension info
            if total_count > 0:
                avg_queue = total_queue_sum
                avg_wait = total_wait_sum / total_count
                avg_speed = total_speed_sum / total_count
                avg_base = total_base_duration / total_count if total_base_duration > 0 else 30  # Default base
                avg_extended = total_extended_time / total_count
                avg_total = total_duration_sum / total_count if total_duration_sum > 0 else avg_base
                
                self.collect_historical_data(
                    now, avg_queue, avg_wait, avg_speed,
                    avg_base, avg_extended, avg_total
                )

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
        self.no_data_label.grid(row=3, column=0, columnspan=2, pady=20)

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


# Standalone functions remain the same
def run_simulation_step(display, event_log):
    """Standalone helper when running the display by itself."""
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

                # PATCH: Allow negative extended_time (decreased time)
                elapsed = max(0.0, current_time - rec.get("last_phase_switch_sim_time", current_time)) if 'last_phase_switch_sim_time' in rec else 0.0
                total_now = elapsed + max(0.0, (next_switch or current_time) - current_time)
                if isinstance(base_duration, (int, float)):
                    extended_time = total_now - base_duration

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
    # Import tkinter.messagebox here to avoid import errors
    import tkinter.messagebox
    
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