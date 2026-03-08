# visualizer.py

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

class DroneVisualizer:
    """Visualize drone detection and localization results with long audio support"""

    def __init__(self, config):
        self.config = config
        self.fig = None
        self.ax = None

    def create_environment_map(self, true_position=None, search_area=(-2, 3, -1, 2)):
        """Create the base environment map with microphone positions"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        x_min, x_max, y_min, y_max = search_area
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (meters)')
        self.ax.set_ylabel('Y Position (meters)')
        self.ax.set_title('Drone Localization System', fontsize=16, fontweight='bold')
        self._draw_microphones()
        if true_position is not None:
            self._draw_true_position(true_position)
        self.ax.text(0.02, 0.98,
                     'Coordinate System:\n• Mic 1: Origin (0,0)\n• X: Right direction\n• Y: Forward direction',
                     transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        return self.fig, self.ax

    def _draw_microphones(self):
        mic_positions = self.config.MIC_POSITIONS
        colors = ['red', 'blue', 'green']
        labels = ['Mic 1 (Ref)', 'Mic 2', 'Mic 3']
        for i, (pos, color, label) in enumerate(zip(mic_positions, colors, labels)):
            circle = Circle(pos, radius=0.05, color=color, alpha=0.8, zorder=5)
            self.ax.add_patch(circle)
            self.ax.text(pos[0], pos[1] + 0.08, label,
                         ha='center', va='bottom', fontweight='bold', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
            if i > 0:
                self.ax.plot([mic_positions[0][0], pos[0]], [mic_positions[0][1], pos[1]],
                             'k--', alpha=0.3, linewidth=1)

    def _draw_true_position(self, true_position):
        self.ax.plot(true_position[0], true_position[1], 'g*',
                     markersize=15, markeredgecolor='black', markeredgewidth=1,
                     label='True Position', zorder=6)
        self.ax.text(true_position[0], true_position[1] + 0.15,
                     f'True: ({true_position[0]:.2f}, {true_position[1]:.2f})',
                     ha='center', va='bottom', fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    def add_estimated_position(self, estimated_position, confidence=None, error=None):
        if self.ax is None:
            self.create_environment_map()
        self.ax.plot(estimated_position[0], estimated_position[1], 'ro',
                     markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                     label='Estimated Position', zorder=7)
        if confidence is not None:
            confidence_radius = 0.1 + (confidence * 0.3)
            self.ax.add_patch(plt.Circle(estimated_position, confidence_radius,
                                        fill=False, linestyle='--',
                                        edgecolor='red', alpha=0.7, linewidth=2))
        if error is not None and error > 0:
            true_pos = None
            for text in self.ax.texts:
                if 'True:' in text.get_text():
                    import re
                    match = re.search(r'\(([-\d.]+),\s*([-\d.]+)\)', text.get_text())
                    if match:
                        true_pos = [float(match.group(1)), float(match.group(2))]
                        break
            if true_pos:
                self.ax.arrow(estimated_position[0], estimated_position[1],
                              true_pos[0] - estimated_position[0],
                              true_pos[1] - estimated_position[1],
                              head_width=0.05, head_length=0.1,
                              fc='red', ec='red', alpha=0.6, linestyle=':')
        label_text = f'Est: ({estimated_position[0]:.2f}, {estimated_position[1]:.2f})'
        if confidence is not None:
            label_text += f'\nConf: {confidence:.1%}'
        if error is not None:
            label_text += f'\nError: {error:.3f}m'
        self.ax.text(estimated_position[0], estimated_position[1] - 0.2,
                     label_text, ha='center', va='top', fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

    def add_localization_info(self, tdoas=None, probability=None, filename=None, segments_info=None):
        if self.ax is None:
            self.create_environment_map()
        info_text = "Localization Info:\n"
        if filename:
            info_text += f"File: {filename}\n"
        if probability is not None:
            status = "DRONE" if probability >= 0.5 else "NO DRONE"
            info_text += f"Status: {status}\nConfidence: {probability:.1%}\n"
        if segments_info:
            info_text += f"Segments: {segments_info['detected']}/{segments_info['total']} detected\n"
            info_text += f"Max confidence: {segments_info['max_confidence']:.1%}"
        if tdoas is not None:
            info_text += f"TDOAs: [{tdoas[0]:.6f}, {tdoas[1]:.6f}] s"
        self.ax.text(0.02, 0.02, info_text, transform=self.ax.transAxes,
                     fontsize=9, verticalalignment='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    def create_segment_analysis_plot(self, segments, filename=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        times = [seg['start_time'] for seg in segments]
        probabilities = [seg['probability'] for seg in segments]
        detected = [seg['detected'] for seg in segments]
        colors = ['red' if det else 'blue' for det in detected]
        bars = ax.bar(times, probabilities, width=2.8, alpha=0.7, color=colors, edgecolor='black')
        ax.axhline(y=0.75, color='orange', linestyle='--', alpha=0.8, label='Detection Threshold (0.75)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Drone Probability')
        ax.set_title(f'Segment Analysis - {filename if filename else "Long Audio File"}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        for bar, prob in zip(bars, probabilities):
            if bar.get_height() > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        detected_count = sum(detected)
        total_segments = len(segments)
        max_prob = max(probabilities)
        summary_text = f"Detection Summary:\nDetected: {detected_count}/{total_segments} segments\nMax confidence: {max_prob:.1%}"
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        plt.tight_layout()
        return fig

    # def show(self):
    #     if self.ax is None:
    #         print("No data to visualize. Call create_environment_map() first.")
    #         return
    #     handles, labels = self.ax.get_legend_handles_labels()
    #     if handles:
    #         self.ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    #     plt.tight_layout()
    #     plt.show()
    def show(self):
        if self.ax is None:
            print("No data to visualize. Call create_environment_map() first.")
            return
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
        # Adjust figure margins to prevent clipping of annotations
        self.fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
        plt.show()

