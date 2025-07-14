#!/usr/bin/env python3
"""
Demo script for Gas Turbine Rotor Dynamics Analyzer

This script demonstrates the key features of the analyzer including:
- Synthetic data generation
- Polar plots for dynamic trim balance
- Data reduction using RDP algorithm
- Multiple bearing analysis
"""

from rotor_dynamics_analyzer import RotorDynamicsAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

def demo_basic_usage():
    """Demonstrate basic usage of the analyzer."""
    print("="*60)
    print("DEMO 1: Basic Usage - Synthetic Data Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = RotorDynamicsAnalyzer()
    
    # Generate synthetic data
    print("Generating synthetic gas turbine data...")
    data = analyzer.generate_synthetic_data(n_rows=1000, n_cols=30, save_to_excel=True)
    
    # Display data info
    print(f"\nGenerated data shape: {data.shape}")
    print(f"Columns: {list(data.columns[:10])}...")  # Show first 10 columns
    
    # Create polar plot for Bearing 1
    print("\nCreating polar plot for Bearing 1...")
    analyzer.create_polar_plot(
        amplitude_col='Bearing_1_Amplitude_mil',
        phase_col='Bearing_1_Phase_deg',
        speed_col='Engine_Speed_RPM',
        title="Bearing 1 Dynamic Trim Balance",
        reduce_data=True
    )

def demo_excel_reading():
    """Demonstrate Excel file reading capability."""
    print("\n" + "="*60)
    print("DEMO 2: Excel File Reading")
    print("="*60)
    
    analyzer = RotorDynamicsAnalyzer()
    
    # Try to read the generated Excel file
    try:
        data = analyzer.read_excel_data('synthetic_gas_turbine_data.xlsx')
        if data is not None:
            print("Successfully loaded data from Excel file!")
            
            # Create analysis for Bearing 2
            analyzer.create_polar_plot(
                amplitude_col='Bearing_2_Amplitude_mil',
                phase_col='Bearing_2_Phase_deg',
                speed_col='Engine_Speed_RPM',
                title="Bearing 2 Dynamic Trim Balance (from Excel)",
                reduce_data=True,
                epsilon=0.01  # Custom RDP threshold
            )
    except Exception as e:
        print(f"Excel reading demo skipped: {e}")

def demo_data_reduction():
    """Demonstrate RDP data reduction with different epsilon values."""
    print("\n" + "="*60)
    print("DEMO 3: RDP Data Reduction Comparison")
    print("="*60)
    
    analyzer = RotorDynamicsAnalyzer()
    analyzer.generate_synthetic_data(n_rows=500, n_cols=15, save_to_excel=False)
    
    # Test different epsilon values
    epsilon_values = [0.001, 0.005, 0.01, 0.02]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, epsilon in enumerate(epsilon_values):
        # Reduce data with specific epsilon
        reduced_data = analyzer.reduce_data('Engine_Speed_RPM', 'Bearing_1_Amplitude_mil', epsilon=epsilon)
        
        # Plot comparison
        ax = axes[i]
        original_speed = analyzer.data['Engine_Speed_RPM'].values
        original_amplitude = analyzer.data['Bearing_1_Amplitude_mil'].values
        
        ax.plot(original_speed, original_amplitude, 'b-', alpha=0.3, linewidth=1, label='Original')
        ax.plot(reduced_data[:, 0], reduced_data[:, 1], 'ro-', markersize=3, 
                linewidth=2, label=f'Reduced (ε={epsilon})')
        
        ax.set_xlabel('Engine Speed (RPM)')
        ax.set_ylabel('Amplitude (mil)')
        ax.set_title(f'RDP Reduction: ε={epsilon}\n({len(reduced_data)} points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('RDP Algorithm Comparison - Different Epsilon Values', fontsize=16, y=1.02)
    plt.show()

def demo_multi_bearing_analysis():
    """Demonstrate analysis of multiple bearings simultaneously."""
    print("\n" + "="*60)
    print("DEMO 4: Multi-Bearing Comparative Analysis")
    print("="*60)
    
    analyzer = RotorDynamicsAnalyzer()
    analyzer.generate_synthetic_data(n_rows=800, n_cols=25, save_to_excel=False)
    
    # Create comparative polar plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(projection='polar'))
    
    bearings = [
        ('Bearing_1_Amplitude_mil', 'Bearing_1_Phase_deg', 'Bearing 1'),
        ('Bearing_2_Amplitude_mil', 'Bearing_2_Phase_deg', 'Bearing 2')
    ]
    
    for i, (amp_col, phase_col, bearing_name) in enumerate(bearings):
        # Get data
        amplitude = analyzer.data[amp_col].values
        phase = analyzer.data[phase_col].values
        speed = analyzer.data['Engine_Speed_RPM'].values
        
        # Reduce data
        reduced_amp_speed = analyzer.reduce_data('Engine_Speed_RPM', amp_col, epsilon=0.008)
        reduced_phase_speed = analyzer.reduce_data('Engine_Speed_RPM', phase_col, epsilon=0.008)
        
        # Plot on polar axes
        ax_polar = axes[i, 0]
        phase_rad = np.deg2rad(np.interp(reduced_amp_speed[:, 0], reduced_phase_speed[:, 0], reduced_phase_speed[:, 1]))
        
        scatter = ax_polar.scatter(phase_rad, reduced_amp_speed[:, 1], 
                                 c=reduced_amp_speed[:, 0], cmap='viridis', 
                                 s=40, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax_polar.plot(phase_rad, reduced_amp_speed[:, 1], 'k-', alpha=0.4, linewidth=1)
        
        ax_polar.set_title(f'{bearing_name} - Polar Plot', pad=20, fontsize=12, fontweight='bold')
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.grid(True, alpha=0.3)
        
        # Cartesian plot
        ax_cart = axes[i, 1]
        ax_cart.plot(speed, amplitude, 'b-', alpha=0.3, linewidth=1, label='Original')
        ax_cart.plot(reduced_amp_speed[:, 0], reduced_amp_speed[:, 1], 'ro-', 
                    markersize=3, linewidth=2, label='Reduced')
        
        ax_cart.set_xlabel('Engine Speed (RPM)')
        ax_cart.set_ylabel('Amplitude (mil)')
        ax_cart.set_title(f'{bearing_name} - Speed vs Amplitude')
        ax_cart.legend()
        ax_cart.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_custom_analysis():
    """Demonstrate custom analysis with user-defined parameters."""
    print("\n" + "="*60)
    print("DEMO 5: Custom Analysis Parameters")
    print("="*60)
    
    analyzer = RotorDynamicsAnalyzer()
    
    # Generate data with custom parameters
    print("Generating custom synthetic data...")
    data = analyzer.generate_synthetic_data(n_rows=1200, n_cols=35, save_to_excel=False)
    
    # Perform analysis with custom settings
    print("Performing custom polar plot analysis...")
    analyzer.create_polar_plot(
        amplitude_col='Bearing_1_Amplitude_mil',
        phase_col='Bearing_1_Phase_deg',
        speed_col='Engine_Speed_RPM',
        title="Custom Analysis - High Resolution Data",
        reduce_data=True,
        epsilon=0.003  # More aggressive reduction
    )
    
    # Show data statistics
    print("\nCustom Data Statistics:")
    print("-" * 30)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    stats = data[numeric_cols].describe()
    print(stats[['Engine_Speed_RPM', 'Bearing_1_Amplitude_mil', 'Bearing_2_Amplitude_mil']])

def main():
    """Run all demo functions."""
    print("Gas Turbine Rotor Dynamics Analyzer - Demo Suite")
    print("="*80)
    
    try:
        # Run demos
        demo_basic_usage()
        demo_excel_reading()
        demo_data_reduction()
        demo_multi_bearing_analysis()
        demo_custom_analysis()
        
        print("\n" + "="*80)
        print("Demo completed successfully!")
        print("Check the generated files:")
        print("- synthetic_gas_turbine_data.xlsx: Sample data file")
        print("- All polar plots and analysis results displayed above")
        print("="*80)
        
    except Exception as e:
        print(f"Demo encountered an error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    import numpy as np  # Import needed for the demo
    main() 