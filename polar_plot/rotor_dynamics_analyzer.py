import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class RotorDynamicsAnalyzer:
    """
    A comprehensive tool for rotor dynamics analysis of gas turbine data.
    
    Features:
    - Excel file reading and processing
    - Synthetic data generation
    - Polar plots for dynamic trim balance
    - RDP algorithm for data reduction
    - Engine speed trend visualization
    """
    
    def __init__(self):
        self.data = None
        self.reduced_data = None
        self.engine_phases = {
            'startup': (0, 20),
            'idle': (20, 30),
            'part_load': (30, 70),
            'full_load': (70, 95),
            'shutdown': (95, 100)
        }
        
    def generate_synthetic_data(self, n_rows=1000, n_cols=30, save_to_excel=True):
        """
        Generate synthetic gas turbine data with realistic parameters.
        
        Parameters:
        -----------
        n_rows : int
            Number of data points (default: 1000)
        n_cols : int
            Number of parameters (default: 30)
        save_to_excel : bool
            Whether to save data to Excel file
            
        Returns:
        --------
        pandas.DataFrame
            Generated synthetic data
        """
        print(f"Generating synthetic gas turbine data ({n_rows} rows, {n_cols} columns)...")
        
        # Time vector
        time = np.linspace(0, 3600, n_rows)  # 1 hour operation in seconds
        
        # Engine speed profile (0 to full load and shutdown)
        engine_speed = self._generate_engine_speed_profile(time)
        
        # Bearing vibration data
        bearing_1_amplitude = self._generate_bearing_vibration(engine_speed, base_amplitude=0.5)
        bearing_1_phase = self._generate_phase_data(engine_speed, base_phase=45)
        bearing_2_amplitude = self._generate_bearing_vibration(engine_speed, base_amplitude=0.7)
        bearing_2_phase = self._generate_phase_data(engine_speed, base_phase=120)
        
        # Temperature data
        turbine_temp = 300 + (engine_speed / 100) * 800 + np.random.normal(0, 20, n_rows)
        compressor_temp = 15 + (engine_speed / 100) * 400 + np.random.normal(0, 15, n_rows)
        
        # Pressure data
        compressor_pressure = 1 + (engine_speed / 100) * 15 + np.random.normal(0, 0.5, n_rows)
        
        # Additional synthetic parameters
        fuel_flow = (engine_speed / 100) * 2000 + np.random.normal(0, 50, n_rows)
        air_flow = (engine_speed / 100) * 50000 + np.random.normal(0, 1000, n_rows)
        
        # Create DataFrame
        data_dict = {
            'Time_s': time,
            'Engine_Speed_RPM': engine_speed,
            'Bearing_1_Amplitude_mil': bearing_1_amplitude,
            'Bearing_1_Phase_deg': bearing_1_phase,
            'Bearing_2_Amplitude_mil': bearing_2_amplitude,
            'Bearing_2_Phase_deg': bearing_2_phase,
            'Turbine_Temp_C': turbine_temp,
            'Compressor_Temp_C': compressor_temp,
            'Compressor_Pressure_bar': compressor_pressure,
            'Fuel_Flow_kg_h': fuel_flow,
            'Air_Flow_kg_s': air_flow,
        }
        
        # Add more synthetic parameters to reach n_cols
        for i in range(11, n_cols):
            param_name = f'Parameter_{i+1}'
            # Generate correlated data with engine speed
            correlation_factor = np.random.uniform(0.3, 0.9)
            noise_factor = np.random.uniform(0.1, 0.3)
            base_value = np.random.uniform(10, 1000)
            
            param_data = (base_value + 
                         correlation_factor * (engine_speed / 100) * base_value + 
                         np.random.normal(0, noise_factor * base_value, n_rows))
            data_dict[param_name] = param_data
            
        self.data = pd.DataFrame(data_dict)
        
        if save_to_excel:
            filename = 'synthetic_gas_turbine_data.xlsx'
            self.data.to_excel(filename, index=False)
            print(f"Synthetic data saved to {filename}")
            
        print("Synthetic data generation completed!")
        return self.data
    
    def _generate_engine_speed_profile(self, time):
        """Generate realistic engine speed profile for startup to shutdown."""
        n_points = len(time)
        speed_profile = np.zeros(n_points)
        
        # Startup phase (0-20%)
        startup_end = int(0.2 * n_points)
        speed_profile[:startup_end] = 3000 * (1 - np.exp(-4 * np.linspace(0, 1, startup_end)))
        
        # Acceleration to part load (20-30%)
        accel_start = startup_end
        accel_end = int(0.3 * n_points)
        speed_profile[accel_start:accel_end] = np.linspace(3000, 7000, accel_end - accel_start)
        
        # Part load operation (30-70%)
        part_load_start = accel_end
        part_load_end = int(0.7 * n_points)
        speed_profile[part_load_start:part_load_end] = 7000 + 1000 * np.sin(
            2 * np.pi * 3 * np.linspace(0, 1, part_load_end - part_load_start)
        )
        
        # Full load (70-95%)
        full_load_start = part_load_end
        full_load_end = int(0.95 * n_points)
        speed_profile[full_load_start:full_load_end] = np.linspace(8000, 12000, full_load_end - full_load_start)
        
        # Shutdown (95-100%)
        shutdown_start = full_load_end
        speed_profile[shutdown_start:] = 12000 * np.exp(-8 * np.linspace(0, 1, n_points - shutdown_start))
        
        # Add some noise
        speed_profile += np.random.normal(0, 50, n_points)
        speed_profile = np.maximum(speed_profile, 0)  # Ensure non-negative
        
        return speed_profile
    
    def _generate_bearing_vibration(self, engine_speed, base_amplitude=0.5):
        """Generate bearing vibration amplitude data."""
        # Base vibration increases with speed
        vibration = base_amplitude * (1 + engine_speed / 10000)
        
        # Add critical speed resonances
        critical_speeds = [3000, 6000, 9000]
        for critical_speed in critical_speeds:
            resonance_factor = 3 * np.exp(-((engine_speed - critical_speed) / 500) ** 2)
            vibration += resonance_factor
            
        # Add noise
        vibration += np.random.normal(0, 0.1, len(engine_speed))
        vibration = np.maximum(vibration, 0.1)  # Minimum vibration level
        
        return vibration
    
    def _generate_phase_data(self, engine_speed, base_phase=0):
        """Generate bearing vibration phase data."""
        # Phase changes with speed and has some random walk
        phase = base_phase + (engine_speed / 10000) * 180
        phase += np.cumsum(np.random.normal(0, 2, len(engine_speed)))
        phase = phase % 360  # Keep within 0-360 degrees
        
        return phase
    
    def read_excel_data(self, filepath):
        """
        Read gas turbine data from Excel file.
        
        Parameters:
        -----------
        filepath : str
            Path to Excel file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        try:
            print(f"Reading data from {filepath}...")
            self.data = pd.read_excel(filepath)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
    
    def rdp_algorithm(self, points, epsilon):
        """
        Ramer-Douglas-Peucker algorithm for data reduction.
        
        Parameters:
        -----------
        points : array-like
            Array of points [[x1, y1], [x2, y2], ...]
        epsilon : float
            Distance threshold for point elimination
            
        Returns:
        --------
        numpy.ndarray
            Reduced set of points
        """
        def perpendicular_distance(point, line_start, line_end):
            """Calculate perpendicular distance from point to line."""
            if np.array_equal(line_start, line_end):
                return np.linalg.norm(point - line_start)
            
            return abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        
        def rdp_recursive(points, epsilon):
            """Recursive RDP implementation."""
            if len(points) < 3:
                return points
            
            # Find the point with maximum distance from line between first and last points
            line_start, line_end = points[0], points[-1]
            max_distance = 0
            max_index = 0
            
            for i in range(1, len(points) - 1):
                distance = perpendicular_distance(points[i], line_start, line_end)
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
            
            # If max distance is greater than epsilon, recursively simplify
            if max_distance > epsilon:
                # Recursive call on both segments
                left_points = rdp_recursive(points[:max_index + 1], epsilon)
                right_points = rdp_recursive(points[max_index:], epsilon)
                
                # Combine results (remove duplicate middle point)
                return np.vstack([left_points[:-1], right_points])
            else:
                # All points can be approximated by a straight line
                return np.array([points[0], points[-1]])
        
        points = np.array(points)
        return rdp_recursive(points, epsilon)
    
    def reduce_data(self, x_col, y_col, epsilon=None):
        """
        Reduce data using RDP algorithm.
        
        Parameters:
        -----------
        x_col : str
            Column name for x-axis data
        y_col : str
            Column name for y-axis data
        epsilon : float
            Distance threshold (auto-calculated if None)
            
        Returns:
        --------
        numpy.ndarray
            Reduced data points
        """
        if self.data is None:
            print("No data available. Please load or generate data first.")
            return None
            
        x_data = self.data[x_col].values
        y_data = self.data[y_col].values
        
        # Normalize data for RDP algorithm
        x_norm = (x_data - x_data.min()) / (x_data.max() - x_data.min())
        y_norm = (y_data - y_data.min()) / (y_data.max() - y_data.min())
        
        points = np.column_stack([x_norm, y_norm])
        
        # Auto-calculate epsilon if not provided
        if epsilon is None:
            epsilon = 0.005  # 0.5% of normalized range
            
        reduced_points = self.rdp_algorithm(points, epsilon)
        
        # Denormalize
        reduced_x = reduced_points[:, 0] * (x_data.max() - x_data.min()) + x_data.min()
        reduced_y = reduced_points[:, 1] * (y_data.max() - y_data.min()) + y_data.min()
        
        self.reduced_data = np.column_stack([reduced_x, reduced_y])
        
        print(f"Data reduced from {len(self.data)} to {len(reduced_points)} points "
              f"({len(reduced_points)/len(self.data)*100:.1f}% reduction)")
        
        return self.reduced_data
    
    def create_polar_plot(self, amplitude_col, phase_col, speed_col, 
                         title="Dynamic Trim Balance Analysis", 
                         reduce_data=True, epsilon=None):
        """
        Create polar plot for dynamic trim balance analysis.
        
        Parameters:
        -----------
        amplitude_col : str
            Column name for vibration amplitude
        phase_col : str
            Column name for vibration phase
        speed_col : str
            Column name for engine speed
        title : str
            Plot title
        reduce_data : bool
            Whether to apply RDP data reduction
        epsilon : float
            RDP threshold (auto-calculated if None)
        """
        if self.data is None:
            print("No data available. Please load or generate data first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get data
        amplitude = self.data[amplitude_col].values
        phase = self.data[phase_col].values
        speed = self.data[speed_col].values
        
        # Convert phase to radians
        phase_rad = np.deg2rad(phase)
        
        # Apply data reduction if requested
        if reduce_data:
            # Reduce amplitude vs speed
            reduced_amp_speed = self.reduce_data(speed_col, amplitude_col, epsilon)
            # Reduce phase vs speed  
            reduced_phase_speed = self.reduce_data(speed_col, phase_col, epsilon)
            
            # Interpolate reduced data
            speed_reduced = reduced_amp_speed[:, 0]
            amplitude_reduced = reduced_amp_speed[:, 1]
            phase_reduced_interp = np.interp(speed_reduced, reduced_phase_speed[:, 0], reduced_phase_speed[:, 1])
            phase_rad_reduced = np.deg2rad(phase_reduced_interp)
        else:
            speed_reduced = speed
            amplitude_reduced = amplitude
            phase_rad_reduced = phase_rad
        
        # Polar plot
        ax1 = plt.subplot(121, projection='polar')
        
        # Create colormap based on engine speed
        colors = plt.cm.viridis(speed_reduced / speed_reduced.max())
        
        # Plot data points
        scatter = ax1.scatter(phase_rad_reduced, amplitude_reduced, 
                            c=speed_reduced, cmap='viridis', 
                            s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Plot trajectory line
        ax1.plot(phase_rad_reduced, amplitude_reduced, 'k-', alpha=0.3, linewidth=1)
        
        # Add speed labels at key points
        self._add_speed_labels(ax1, phase_rad_reduced, amplitude_reduced, speed_reduced)
        
        # Customize polar plot
        ax1.set_title(f'{title}\nPolar Plot', pad=20, fontsize=14, fontweight='bold')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rlabel_position(45)
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8, pad=0.1)
        cbar.set_label('Engine Speed (RPM)', fontsize=12)
        
        # Cartesian plot: Speed vs Amplitude
        ax2.plot(speed, amplitude, 'b-', alpha=0.3, linewidth=1, label='Original Data')
        if reduce_data:
            ax2.plot(speed_reduced, amplitude_reduced, 'ro-', markersize=4, 
                    linewidth=2, label='Reduced Data')
        
        ax2.set_xlabel('Engine Speed (RPM)', fontsize=12)
        ax2.set_ylabel(f'{amplitude_col} (mil)', fontsize=12)
        ax2.set_title('Engine Speed vs Vibration Amplitude', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add phase annotations
        self._add_phase_annotations(ax2, speed_reduced, amplitude_reduced)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        self._print_analysis_summary(amplitude, phase, speed, amplitude_reduced if reduce_data else amplitude)
    
    def _add_speed_labels(self, ax, phase_rad, amplitude, speed):
        """Add speed labels at key operational points."""
        # Find indices for key operational phases
        key_points = []
        
        for phase_name, (start_pct, end_pct) in self.engine_phases.items():
            # Find speed range for this phase
            speed_range = (speed.max() - speed.min())
            phase_start_speed = speed.min() + (start_pct / 100) * speed_range
            phase_end_speed = speed.min() + (end_pct / 100) * speed_range
            
            # Find data points in this speed range
            mask = (speed >= phase_start_speed) & (speed <= phase_end_speed)
            if np.any(mask):
                # Get middle point of this phase
                mid_idx = np.where(mask)[0][len(np.where(mask)[0]) // 2]
                key_points.append((mid_idx, phase_name))
        
        # Add labels
        for idx, phase_name in key_points:
            if idx < len(phase_rad):
                ax.annotate(f'{phase_name.title()}\n{speed[idx]:.0f} RPM',
                           xy=(phase_rad[idx], amplitude[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                           fontsize=9, ha='left')
    
    def _add_phase_annotations(self, ax, speed, amplitude):
        """Add operational phase annotations to cartesian plot."""
        speed_range = speed.max() - speed.min()
        
        for phase_name, (start_pct, end_pct) in self.engine_phases.items():
            start_speed = speed.min() + (start_pct / 100) * speed_range
            end_speed = speed.min() + (end_pct / 100) * speed_range
            
            ax.axvspan(start_speed, end_speed, alpha=0.1, 
                      label=f'{phase_name.title()} Phase')
            
            # Add text label
            mid_speed = (start_speed + end_speed) / 2
            ax.text(mid_speed, ax.get_ylim()[1] * 0.9, phase_name.title(),
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _print_analysis_summary(self, amplitude_orig, phase_orig, speed, amplitude_reduced):
        """Print analysis summary statistics."""
        print("\n" + "="*60)
        print("ROTOR DYNAMICS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Data Points - Original: {len(amplitude_orig)}, Reduced: {len(amplitude_reduced)}")
        print(f"Speed Range: {speed.min():.0f} - {speed.max():.0f} RPM")
        print(f"Amplitude Range: {amplitude_orig.min():.2f} - {amplitude_orig.max():.2f} mil")
        print(f"Phase Range: {phase_orig.min():.1f} - {phase_orig.max():.1f} degrees")
        
        # Find critical speeds (local maxima in amplitude)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(amplitude_orig, height=np.mean(amplitude_orig) + np.std(amplitude_orig))
        
        if len(peaks) > 0:
            print(f"\nCritical Speeds Detected:")
            for i, peak_idx in enumerate(peaks[:5]):  # Show top 5
                print(f"  {i+1}. {speed[peak_idx]:.0f} RPM - Amplitude: {amplitude_orig[peak_idx]:.2f} mil")
        
        print("="*60)

# Example usage and testing functions
def main():
    """Main function to demonstrate the rotor dynamics analyzer."""
    print("Gas Turbine Rotor Dynamics Analyzer")
    print("="*50)
    
    # Initialize analyzer
    analyzer = RotorDynamicsAnalyzer()
    
    # Generate synthetic data
    data = analyzer.generate_synthetic_data(n_rows=1000, n_cols=30)
    
    # Create polar plots for bearing analysis
    print("\nCreating polar plots for bearing analysis...")
    
    # Bearing 1 analysis
    analyzer.create_polar_plot(
        amplitude_col='Bearing_1_Amplitude_mil',
        phase_col='Bearing_1_Phase_deg', 
        speed_col='Engine_Speed_RPM',
        title="Bearing 1 Dynamic Trim Balance",
        reduce_data=True
    )
    
    # Bearing 2 analysis
    analyzer.create_polar_plot(
        amplitude_col='Bearing_2_Amplitude_mil',
        phase_col='Bearing_2_Phase_deg',
        speed_col='Engine_Speed_RPM', 
        title="Bearing 2 Dynamic Trim Balance",
        reduce_data=True
    )
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 