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
            return points
        
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
            
            return rec_results1[:-1] + rec_results2
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
        """Create polar plot for dynamic trim balance analysis"""
        
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
        
        # Apply RDP algorithm for data reduction
        # Convert to cartesian for RDP
        x_cart = amplitude * np.cos(phase)
        y_cart = amplitude * np.sin(phase)
        
        # Reduce data
        x_reduced, y_reduced = self.reduce_data_rdp(x_cart, y_cart, epsilon=0.05)
        
        # Convert back to polar
        amp_reduced = np.sqrt(x_reduced**2 + y_reduced**2)
        phase_reduced = np.arctan2(y_reduced, x_reduced)
        
        # Create polar plot
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Plot full data (light)
        scatter = ax.scatter(phase, amplitude, c=speed, cmap='viridis', alpha=0.3, s=20, label='All Data')
        
        # Plot reduced data (prominent)
        ax.scatter(phase_reduced, amp_reduced, c='red', s=60, alpha=0.8, marker='o', 
                  edgecolor='black', linewidth=1, label='Reduced Data (RDP)')
        
        # Connect reduced points
        ax.plot(phase_reduced, amp_reduced, 'r-', linewidth=2, alpha=0.7)
        
        # Add phase annotations for key engine states
        phase_labels = {
            0: '0°', np.pi/2: '90°', np.pi: '180°', 3*np.pi/2: '270°'
        }
        
        for angle, label in phase_labels.items():
            ax.text(angle, ax.get_ylim()[1] * 1.1, label, fontsize=12, ha='center')
        
        # Customize plot
        ax.set_title(f'Dynamic Trim Balance - {bearing_name.replace("_", " ").title()}{title_suffix}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(amplitude) * 1.2)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for speed
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('Engine Speed (RPM)', fontsize=12)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        return fig
    
    def create_speed_trend_plot(self):
        """Create engine speed trend plot with phase labels"""
        
        if self.data is None:
            raise ValueError("No data available. Please generate or load data first.")
        
        # Apply RDP to speed data
        time_data = self.data['time'].values
        speed_data = self.data['engine_speed_rpm'].values
        
        # Reduce data for cleaner plot
        time_reduced, speed_reduced = self.reduce_data_rdp(time_data, speed_data, epsilon=50)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot original data (light)
        ax.plot(time_data, speed_data, color='lightblue', alpha=0.6, linewidth=1, label='Original Data')
        
        # Plot reduced data (prominent)
        ax.plot(time_reduced, speed_reduced, color='navy', linewidth=3, label='Reduced Data (RDP)')
        
        # Add phase regions
        colors = ['lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightgray']
        phase_names = ['Startup', 'Idle', 'Acceleration', 'Full Load', 'Shutdown']
        
        for i, (phase, (start, end)) in enumerate(self.engine_phases.items()):
            ax.axvspan(start, end, alpha=0.3, color=colors[i], label=phase_names[i])
            
            # Add phase labels
            mid_point = (start + end) / 2
            max_speed = max(speed_data)
            ax.text(mid_point, max_speed * 0.9, phase_names[i], 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Normalized Time', fontsize=14)
        ax.set_ylabel('Engine Speed (RPM)', fontsize=14)
        ax.set_title('Gas Turbine Engine Speed Trend Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # Add annotations for key points
        key_points = [
            (0.0, 0, 'Engine Start'),
            (0.7, max(speed_data), 'Full Load'),
            (1.0, 0, 'Shutdown')
        ]
        
        for x, y, label in key_points:
            if x <= max(time_data):
                # Find closest point in reduced data
                idx = np.argmin(np.abs(time_reduced - x))
                ax.annotate(label, (time_reduced[idx], speed_reduced[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
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