#!/usr/bin/env python3
"""
Quick Start Example for Gas Turbine Rotor Dynamics Analyzer

This script demonstrates the most common use cases:
1. Generate synthetic data
2. Create polar plots for bearing analysis
3. Analyze multiple bearings
"""

from rotor_dynamics_analyzer import RotorDynamicsAnalyzer
import matplotlib.pyplot as plt

def quick_example():
    """Run a quick example analysis."""
    print("Gas Turbine Rotor Dynamics Analyzer - Quick Start")
    print("="*60)
    
    # Step 1: Initialize the analyzer
    analyzer = RotorDynamicsAnalyzer()
    
    # Step 2: Generate synthetic data
    print("Generating synthetic gas turbine data...")
    data = analyzer.generate_synthetic_data(
        n_rows=1000,     # 1000 data points
        n_cols=30,       # 30 parameters
        save_to_excel=True  # Save to Excel file
    )
    
    print(f"Generated data with {data.shape[0]} rows and {data.shape[1]} columns")
    print(f"Speed range: {data['Engine_Speed_RPM'].min():.0f} - {data['Engine_Speed_RPM'].max():.0f} RPM")
    
    # Step 3: Create polar plot for Bearing 1
    print("\nCreating polar plot for Bearing 1...")
    analyzer.create_polar_plot(
        amplitude_col='Bearing_1_Amplitude_mil',
        phase_col='Bearing_1_Phase_deg',
        speed_col='Engine_Speed_RPM',
        title="Bearing 1 - Dynamic Trim Balance Analysis",
        reduce_data=True
    )
    
    # Step 4: Create polar plot for Bearing 2
    print("\nCreating polar plot for Bearing 2...")
    analyzer.create_polar_plot(
        amplitude_col='Bearing_2_Amplitude_mil',
        phase_col='Bearing_2_Phase_deg',
        speed_col='Engine_Speed_RPM',
        title="Bearing 2 - Dynamic Trim Balance Analysis",
        reduce_data=True
    )
    
    print("\nQuick start example completed!")
    print("Check the generated files:")
    print("- synthetic_gas_turbine_data.xlsx: Your synthetic data")
    print("- Polar plots should be displayed showing vibration analysis")

def analyze_your_data():
    """Example of how to analyze your own data."""
    print("\n" + "="*60)
    print("To analyze your own Excel data, use this pattern:")
    print("="*60)
    
    code_example = '''
# Initialize analyzer
analyzer = RotorDynamicsAnalyzer()

# Load your Excel file
data = analyzer.read_excel_data('your_data.xlsx')

# Create polar plot (adjust column names to match your data)
analyzer.create_polar_plot(
    amplitude_col='Your_Amplitude_Column',
    phase_col='Your_Phase_Column',
    speed_col='Your_Speed_Column',
    title="Your Analysis Title",
    reduce_data=True,
    epsilon=0.005  # Adjust for more/less data reduction
)
'''
    print(code_example)

if __name__ == "__main__":
    quick_example()
    analyze_your_data() 