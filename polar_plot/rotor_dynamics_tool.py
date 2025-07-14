import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class RotorDynamicsAnalyzer:
    """
    A comprehensive tool for gas turbine rotor dynamics analysis
    """
    
    def __init__(self):
        self.data = None
        self.engine_phases = {
            'startup': (0, 0.2),
            'idle': (0.2, 0.3),
            'acceleration': (0.3, 0.7),
            'full_load': (0.7, 0.9),
            'shutdown': (0.9, 1.0)
        }
        
    def generate_synthetic_data(self, n_rows=1000, n_cols=30):
        """Generate synthetic gas turbine data with realistic parameters"""
        
        # Time series (normalized 0-1 for engine cycle)
        time_norm = np.linspace(0, 1, n_rows)
        
        # Engine speed profile (realistic startup to shutdown)
        speed_profile = np.zeros_like(time_norm)
        for i, t in enumerate(time_norm):
            if t < 0.2:  # Startup
                speed_profile[i] = 3000 * (t / 0.2) ** 2
            elif t < 0.3:  # Idle
                speed_profile[i] = 3000 + 500 * np.sin(10 * np.pi * (t - 0.2))
            elif t < 0.7:  # Acceleration
                speed_profile[i] = 3000 + 7000 * ((t - 0.3) / 0.4) ** 1.5
            elif t < 0.9:  # Full load
                speed_profile[i] = 10000 + 200 * np.sin(5 * np.pi * (t - 0.7))
            else:  # Shutdown
                speed_profile[i] = 10000 * (1 - (t - 0.9) / 0.1) ** 2
        
        # Add noise to speed
        speed_profile += np.random.normal(0, 50, len(speed_profile))
        speed_profile = np.maximum(speed_profile, 0)
        
        # Generate bearing vibration data
        bearing_data = {}
        
        # Multiple bearing locations
        bearing_locations = ['bearing_1', 'bearing_2', 'bearing_3', 'bearing_4']
        
        for bearing in bearing_locations:
            # Vibration amplitude (increases with speed, with resonances)
            base_amplitude = 0.1 + 0.3 * (speed_profile / 10000) ** 2
            
            # Add critical speed resonances
            critical_speeds = [3500, 7500]
            for cs in critical_speeds:
                resonance = 2.0 * np.exp(-((speed_profile - cs) / 500) ** 2)
                base_amplitude += resonance
            
            # Add random variations
            amplitude = base_amplitude + np.random.normal(0, 0.1, len(speed_profile))
            amplitude = np.maximum(amplitude, 0.01)  # Minimum amplitude
            
            # Phase data (related to imbalance)
            phase_trend = 2 * np.pi * np.cumsum(np.random.normal(0, 0.01, len(speed_profile)))
            phase_imbalance = np.pi * np.sin(2 * np.pi * speed_profile / 5000)
            phase = (phase_trend + phase_imbalance) % (2 * np.pi)
            
            bearing_data[f'{bearing}_amplitude'] = amplitude
            bearing_data[f'{bearing}_phase'] = phase
            bearing_data[f'{bearing}_phase_deg'] = np.degrees(phase)
        
        # Additional parameters
        data = {
            'time': time_norm,
            'engine_speed_rpm': speed_profile,
            'fuel_flow': 100 + 400 * (speed_profile / 10000) ** 1.5 + np.random.normal(0, 20, len(speed_profile)),
            'exhaust_temp': 400 + 600 * (speed_profile / 10000) ** 1.2 + np.random.normal(0, 30, len(speed_profile)),
            'compressor_pressure': 1 + 15 * (speed_profile / 10000) ** 2 + np.random.normal(0, 0.5, len(speed_profile)),
            'oil_pressure': 2 + 8 * (speed_profile / 10000) + np.random.normal(0, 0.3, len(speed_profile)),
            'oil_temperature': 50 + 100 * (speed_profile / 10000) ** 1.5 + np.random.normal(0, 5, len(speed_profile)),
        }
        
        # Add bearing data
        data.update(bearing_data)
        
        # Add more synthetic parameters to reach 30 columns
        additional_params = [
            'turbine_inlet_temp', 'turbine_outlet_temp', 'combustor_pressure',
            'air_flow_rate', 'power_output', 'efficiency', 'nox_emissions',
            'co_emissions', 'thrust', 'specific_fuel_consumption'
        ]
        
        for param in additional_params:
            if param in ['turbine_inlet_temp', 'turbine_outlet_temp']:
                data[param] = 800 + 400 * (speed_profile / 10000) ** 1.3 + np.random.normal(0, 40, len(speed_profile))
            elif param in ['nox_emissions', 'co_emissions']:
                data[param] = 10 + 50 * (speed_profile / 10000) ** 2 + np.random.normal(0, 5, len(speed_profile))
            else:
                data[param] = np.random.normal(100, 20, len(speed_profile)) * (speed_profile / 10000)
        
        # Create DataFrame
        self.data = pd.DataFrame(data)
        
        # Ensure we have exactly 30 columns
        while len(self.data.columns) < n_cols:
            col_name = f'parameter_{len(self.data.columns)}'
            self.data[col_name] = np.random.normal(0, 1, len(self.data))
        
        return self.data
    
    def rdp_algorithm(self, points, epsilon):
        """
        Ramer-Douglas-Peucker algorithm for data reduction
        """
        if len(points) < 3:
            return points.tolist() if hasattr(points, 'tolist') else list(points)
        
        # Find the point with maximum distance
        dmax = 0
        index = 0
        
        for i in range(1, len(points) - 1):
            d = self.perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            rec_results1 = self.rdp_algorithm(points[:index + 1], epsilon)
            rec_results2 = self.rdp_algorithm(points[index:], epsilon)
            
            # Ensure we're concatenating lists, not numpy arrays
            result1 = list(rec_results1[:-1])
            result2 = list(rec_results2)
            return result1 + result2
        else:
            return [points[0], points[-1]]
    
    def perpendicular_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
    
    def reduce_data_rdp(self, x_data, y_data, epsilon=0.1):
        """Apply RDP algorithm to reduce data points"""
        points = np.column_stack((x_data, y_data))
        reduced_points = self.rdp_algorithm(points, epsilon)
        reduced_points = np.array(reduced_points)
        
        return reduced_points[:, 0], reduced_points[:, 1]
    
    def create_polar_plot(self, bearing_name='bearing_1', title_suffix=""):
        """Create polar plot for dynamic trim balance analysis with start, stop, and full load labels only"""
        
        if self.data is None:
            raise ValueError("No data available. Please generate or load data first.")
        
        amp_col = f'{bearing_name}_amplitude'
        phase_col = f'{bearing_name}_phase'
        
        if amp_col not in self.data.columns or phase_col not in self.data.columns:
            raise ValueError(f"Bearing data for {bearing_name} not found in dataset")
        
        # Get amplitude and phase data
        amplitude = self.data[amp_col].values
        phase = self.data[phase_col].values
        speed = self.data['engine_speed_rpm'].values
        
        # Identify key operating conditions
        # Start: lowest 10% of speeds
        # Stop: similar to start (engine shutdown)
        # Full Load: highest 10% of speeds
        speed_10th = np.percentile(speed, 10)
        speed_90th = np.percentile(speed, 90)
        
        start_mask = speed <= speed_10th
        full_load_mask = speed >= speed_90th
        stop_mask = speed <= speed_10th  # Assuming stop is similar to start conditions
        
        # Get representative points for each condition
        start_idx = np.where(start_mask)[0]
        full_load_idx = np.where(full_load_mask)[0]
        stop_idx = np.where(stop_mask)[0]
        
        # Select representative points (median values for each condition)
        if len(start_idx) > 0:
            start_point_idx = start_idx[len(start_idx)//4]  # Early in start sequence
            start_amp, start_phase = amplitude[start_point_idx], phase[start_point_idx]
        else:
            start_amp, start_phase = amplitude[0], phase[0]
        
        if len(full_load_idx) > 0:
            full_load_point_idx = full_load_idx[len(full_load_idx)//2]  # Middle of full load
            full_load_amp, full_load_phase = amplitude[full_load_point_idx], phase[full_load_point_idx]
        else:
            full_load_amp, full_load_phase = amplitude[len(amplitude)//2], phase[len(phase)//2]
        
        if len(stop_idx) > 0:
            stop_point_idx = stop_idx[-len(stop_idx)//4]  # Late in stop sequence
            stop_amp, stop_phase = amplitude[stop_point_idx], phase[stop_point_idx]
        else:
            stop_amp, stop_phase = amplitude[-1], phase[-1]
        
        # Create polar plot
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Get speed values for each point
        start_speed = speed[start_point_idx] if len(start_idx) > 0 else speed[0]
        full_load_speed = speed[full_load_point_idx] if len(full_load_idx) > 0 else speed[len(speed)//2]
        stop_speed = speed[stop_point_idx] if len(stop_idx) > 0 else speed[-1]
        
        # Create speed trend line connecting the three points
        # Order points by operational sequence: START -> FULL LOAD -> STOP
        trend_phases = [start_phase, full_load_phase, stop_phase]
        trend_amps = [start_amp, full_load_amp, stop_amp]
        trend_speeds = [start_speed, full_load_speed, stop_speed]
        
        # Plot speed trend line connecting the points
        ax.plot(trend_phases, trend_amps, color='purple', linewidth=4, alpha=0.7, 
                linestyle='--', marker='none', label='Speed Trend Path', zorder=3)
        
        # Plot only the three key points
        ax.scatter(start_phase, start_amp, c='green', s=150, marker='o', 
                  edgecolor='black', linewidth=2, label='Start', zorder=5)
        ax.scatter(full_load_phase, full_load_amp, c='red', s=150, marker='s', 
                  edgecolor='black', linewidth=2, label='Full Load', zorder=5)
        ax.scatter(stop_phase, stop_amp, c='blue', s=150, marker='^', 
                  edgecolor='black', linewidth=2, label='Stop', zorder=5)
        
        # Add text labels for each point with speed information
        ax.annotate(f'START\n{start_speed:.0f} RPM', xy=(start_phase, start_amp), 
                   xytext=(start_phase, start_amp * 1.4),
                   fontsize=11, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
        
        ax.annotate(f'FULL LOAD\n{full_load_speed:.0f} RPM', xy=(full_load_phase, full_load_amp), 
                   xytext=(full_load_phase, full_load_amp * 1.4),
                   fontsize=11, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcoral', alpha=0.8))
        
        ax.annotate(f'STOP\n{stop_speed:.0f} RPM', xy=(stop_phase, stop_amp), 
                   xytext=(stop_phase, stop_amp * 1.4),
                   fontsize=11, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
        
        # Add speed progression arrows between points
        # Arrow from START to FULL LOAD
        ax.annotate('', xy=(full_load_phase, full_load_amp), xytext=(start_phase, start_amp),
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2, alpha=0.6))
        
        # Arrow from FULL LOAD to STOP  
        ax.annotate('', xy=(stop_phase, stop_amp), xytext=(full_load_phase, full_load_amp),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=2, alpha=0.6))
        
        # Add phase annotations for reference
        phase_labels = {
            0: '0°', np.pi/2: '90°', np.pi: '180°', 3*np.pi/2: '270°'
        }
        
        max_amp = max(max(amplitude), start_amp, full_load_amp, stop_amp)
        for angle, label in phase_labels.items():
            ax.text(angle, max_amp * 1.5, label, fontsize=10, ha='center', alpha=0.6)
        
        # Customize plot
        ax.set_title(f'Dynamic Trim Balance with Speed Trend - {bearing_name.replace("_", " ").title()}{title_suffix}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max_amp * 1.6)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=11)
        
        # Add detailed speed progression info
        info_text = f"""Speed Progression:
START: {start_speed:.0f} RPM → FULL LOAD: {full_load_speed:.0f} RPM → STOP: {stop_speed:.0f} RPM
Total Range: {speed.min():.0f} - {speed.max():.0f} RPM"""
        plt.figtext(0.02, 0.02, info_text, fontsize=10, style='italic',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_speed_trend_plot(self):
        """Create engine speed trend plot with 6 key labeled points only"""
        
        if self.data is None:
            raise ValueError("No data available. Please generate or load data first.")
        
        time_data = self.data['time'].values
        speed_data = self.data['engine_speed_rpm'].values
        
        # Identify 6 key points: Start, Stop, Full Load + 3 equally spaced
        n_points = len(time_data)
        
        # Key indices for 6 points
        start_idx = 0                                    # Start
        stop_idx = n_points - 1                         # Stop
        full_load_idx = int(0.7 * n_points)            # Full load around 70% through cycle
        
        # 3 equally spaced points between start and full load
        mid_idx_1 = int(0.2 * n_points)                # 20% - Ramp up
        mid_idx_2 = int(0.45 * n_points)               # 45% - Mid acceleration  
        mid_idx_3 = int(0.85 * n_points)               # 85% - Wind down
        
        # Extract the 6 key points
        key_indices = [start_idx, mid_idx_1, mid_idx_2, full_load_idx, mid_idx_3, stop_idx]
        key_times = [time_data[i] for i in key_indices]
        key_speeds = [speed_data[i] for i in key_indices]
        
        # Define labels and colors for each point
        labels = ['START', 'RAMP UP', 'ACCELERATION', 'FULL LOAD', 'WIND DOWN', 'STOP']
        colors = ['green', 'orange', 'blue', 'red', 'purple', 'gray']
        markers = ['o', 's', '^', 'D', 'v', 'X']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot trend line connecting the 6 points
        ax.plot(key_times, key_speeds, color='navy', linewidth=3, alpha=0.7, 
                linestyle='-', label='Speed Trend Line')
        
        # Plot the 6 key points
        for i, (time, speed, label, color, marker) in enumerate(zip(key_times, key_speeds, labels, colors, markers)):
            ax.scatter(time, speed, s=200, c=color, marker=marker, 
                      edgecolor='black', linewidth=2, zorder=5, alpha=0.9)
            
            # Add labels with background boxes
            offset_y = max(speed_data) * 0.08 if i % 2 == 0 else -max(speed_data) * 0.08
            ax.annotate(label, xy=(time, speed), xytext=(time, speed + offset_y),
                       fontsize=11, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.3, edgecolor='black'),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1))
        
        # Add speed and time info for each point
        info_text = "Key Operating Points:\n"
        for i, (time, speed, label) in enumerate(zip(key_times, key_speeds, labels)):
            info_text += f"{label}: {speed:.0f} RPM @ t={time:.2f}\n"
        
        # Add info box
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='lightgray', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Normalized Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Engine Speed (RPM)', fontsize=14, fontweight='bold')
        ax.set_title('Gas Turbine Engine Speed Trend - Key Operating Points', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(speed_data) * 1.1)
        
        # Create legend for markers
        legend_elements = [plt.Line2D([0], [0], marker=markers[i], color=colors[i], 
                                     markersize=10, linestyle='None', label=labels[i])
                          for i in range(6)]
        legend_elements.append(plt.Line2D([0], [0], color='navy', linewidth=3, 
                                         label='Speed Trend Line'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 0.85), 
                 fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_analysis(self):
        """Create a comprehensive analysis with multiple plots"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # Speed trend plot
        ax1 = plt.subplot(2, 3, (1, 3))
        time_data = self.data['time'].values
        speed_data = self.data['engine_speed_rpm'].values
        time_reduced, speed_reduced = self.reduce_data_rdp(time_data, speed_data, epsilon=50)
        
        ax1.plot(time_data, speed_data, color='lightblue', alpha=0.6, linewidth=1)
        ax1.plot(time_reduced, speed_reduced, color='navy', linewidth=3)
        ax1.set_title('Engine Speed Trend', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Normalized Time')
        ax1.set_ylabel('Engine Speed (RPM)')
        ax1.grid(True, alpha=0.3)
        
        # Add phase regions
        colors = ['lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightgray']
        phase_names = ['Startup', 'Idle', 'Acceleration', 'Full Load', 'Shutdown']
        
        for i, (phase, (start, end)) in enumerate(self.engine_phases.items()):
            ax1.axvspan(start, end, alpha=0.3, color=colors[i])
            mid_point = (start + end) / 2
            max_speed = max(speed_data)
            ax1.text(mid_point, max_speed * 0.9, phase_names[i], 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Polar plots for different bearings
        bearing_positions = [(2, 3, 4), (2, 3, 5), (2, 3, 6)]
        bearings = ['bearing_1', 'bearing_2', 'bearing_3']
        
        for i, (bearing, pos) in enumerate(zip(bearings, bearing_positions)):
            ax = plt.subplot(*pos, projection='polar')
            
            amp_col = f'{bearing}_amplitude'
            phase_col = f'{bearing}_phase'
            
            amplitude = self.data[amp_col].values
            phase = self.data[phase_col].values
            speed = self.data['engine_speed_rpm'].values
            
            # Apply RDP
            x_cart = amplitude * np.cos(phase)
            y_cart = amplitude * np.sin(phase)
            x_reduced, y_reduced = self.reduce_data_rdp(x_cart, y_cart, epsilon=0.05)
            amp_reduced = np.sqrt(x_reduced**2 + y_reduced**2)
            phase_reduced = np.arctan2(y_reduced, x_reduced)
            
            # Plot
            scatter = ax.scatter(phase, amplitude, c=speed, cmap='viridis', alpha=0.3, s=15)
            ax.scatter(phase_reduced, amp_reduced, c='red', s=40, alpha=0.8, marker='o')
            ax.plot(phase_reduced, amp_reduced, 'r-', linewidth=2, alpha=0.7)
            
            ax.set_title(f'{bearing.replace("_", " ").title()} Balance', fontsize=12, fontweight='bold')
            ax.set_theta_zero_location('E')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_to_excel(self, filename='gas_turbine_data.xlsx'):
        """Save the generated data to Excel file"""
        if self.data is None:
            raise ValueError("No data to save. Please generate data first.")
        
        self.data.to_excel(filename, index=False)
        print(f"Data saved to {filename}")
    
    def load_from_excel(self, filename):
        """Load data from Excel file"""
        try:
            self.data = pd.read_excel(filename)
            print(f"Data loaded from {filename}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_summary_statistics(self):
        """Get summary statistics for the dataset"""
        if self.data is None:
            raise ValueError("No data available. Please generate or load data first.")
        
        return self.data.describe()

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = RotorDynamicsAnalyzer()
    
    # Generate synthetic data
    print("Generating synthetic gas turbine data...")
    data = analyzer.generate_synthetic_data(n_rows=1000, n_cols=30)
    print(f"Generated data shape: {data.shape}")
    
    # Save to Excel
    analyzer.save_to_excel('gas_turbine_rotor_data.xlsx')
    
    # Create plots
    print("\nCreating rotor dynamics analysis plots...")
    
    # Individual polar plot
    fig1 = analyzer.create_polar_plot('bearing_1', " - Detailed Analysis")
    plt.show()
    
    # Speed trend plot
    fig2 = analyzer.create_speed_trend_plot()
    plt.show()
    
    # Comprehensive analysis
    fig3 = analyzer.create_comprehensive_analysis()
    plt.show()
    
    # Display summary statistics
    print("\nDataset Summary Statistics:")
    print(analyzer.get_summary_statistics())
    
    print("\nRotor Dynamics Analysis Complete!")
    print("Key Features:")
    print("- RDP algorithm applied for data reduction")
    print("- Polar plots for dynamic trim balance")
    print("- Engine speed trend with phase labels")
    print("- Synthetic data with realistic turbine parameters")
    print("- Excel export/import functionality")