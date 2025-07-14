# Gas Turbine Rotor Dynamics Analyzer

A comprehensive Python tool for analyzing gas turbine rotor dynamics data with focus on bearing vibration analysis, dynamic trim balance, and operational phase tracking.

## Features

- **Excel Data Processing**: Read and analyze gas turbine data from Excel files
- **Synthetic Data Generation**: Create realistic synthetic gas turbine data for testing and demonstration
- **Polar Plots**: Generate polar plots for dynamic trim balance analysis
- **Data Reduction**: Implement Ramer-Douglas-Peucker (RDP) algorithm to reduce data density for cleaner visualizations
- **Engine Phase Tracking**: Automatically identify and label operational phases (startup, idle, part load, full load, shutdown)
- **Multi-Bearing Analysis**: Analyze multiple bearing positions simultaneously
- **Critical Speed Detection**: Automatically detect critical speeds from vibration data

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.9.0
- openpyxl >= 3.0.9
- xlsxwriter >= 3.0.0

## Quick Start

### Basic Usage

```python
from rotor_dynamics_analyzer import RotorDynamicsAnalyzer

# Initialize the analyzer
analyzer = RotorDynamicsAnalyzer()

# Generate synthetic data (1000 rows, 30 columns)
data = analyzer.generate_synthetic_data(n_rows=1000, n_cols=30)

# Create polar plot for bearing analysis
analyzer.create_polar_plot(
    amplitude_col='Bearing_1_Amplitude_mil',
    phase_col='Bearing_1_Phase_deg',
    speed_col='Engine_Speed_RPM',
    title="Bearing 1 Dynamic Trim Balance",
    reduce_data=True
)
```

### Reading Excel Data

```python
# Read your own Excel file
data = analyzer.read_excel_data('your_gas_turbine_data.xlsx')

# Analyze the data
analyzer.create_polar_plot(
    amplitude_col='your_amplitude_column',
    phase_col='your_phase_column',
    speed_col='your_speed_column'
)
```

## Data Format

The tool expects Excel files with the following types of columns:

### Required Columns
- **Engine Speed**: RPM values (e.g., 'Engine_Speed_RPM')
- **Bearing Amplitude**: Vibration amplitude in mils (e.g., 'Bearing_1_Amplitude_mil')
- **Bearing Phase**: Phase angle in degrees (e.g., 'Bearing_1_Phase_deg')

### Optional Columns
- Time series data
- Temperature measurements
- Pressure readings
- Flow rates
- Any other engine parameters

### Example Data Structure
```
| Time_s | Engine_Speed_RPM | Bearing_1_Amplitude_mil | Bearing_1_Phase_deg | Turbine_Temp_C | ... |
|--------|------------------|-------------------------|---------------------|----------------|-----|
| 0.0    | 0.0              | 0.12                    | 45.2               | 295.3          | ... |
| 3.6    | 245.8            | 0.15                    | 47.1               | 298.7          | ... |
| 7.2    | 512.3            | 0.18                    | 52.3               | 305.2          | ... |
```

## Features in Detail

### 1. Synthetic Data Generation

The tool can generate realistic synthetic gas turbine data including:

- **Engine Speed Profile**: Realistic startup, acceleration, steady-state, and shutdown phases
- **Bearing Vibration**: Amplitude and phase data with critical speed resonances
- **Temperature Data**: Turbine and compressor temperatures correlated with engine speed
- **Pressure Data**: Compressor discharge pressure
- **Flow Data**: Fuel flow and air flow measurements
- **Additional Parameters**: Up to 30 customizable parameters

```python
# Generate custom synthetic data
data = analyzer.generate_synthetic_data(
    n_rows=1500,      # Number of data points
    n_cols=35,        # Number of columns
    save_to_excel=True # Save to Excel file
)
```

### 2. Polar Plot Analysis

Create comprehensive polar plots showing:

- **Vibration Vectors**: Amplitude and phase in polar coordinates
- **Speed Color Coding**: Points colored by engine speed
- **Trajectory Lines**: Showing vibration evolution during operation
- **Phase Labels**: Automatic labeling of operational phases
- **Critical Speed Detection**: Identification of resonance points

```python
analyzer.create_polar_plot(
    amplitude_col='Bearing_1_Amplitude_mil',
    phase_col='Bearing_1_Phase_deg',
    speed_col='Engine_Speed_RPM',
    title="Dynamic Trim Balance Analysis",
    reduce_data=True,    # Apply RDP data reduction
    epsilon=0.005        # Custom reduction threshold
)
```

### 3. Data Reduction (RDP Algorithm)

The Ramer-Douglas-Peucker algorithm reduces data density while preserving important features:

- **Automatic Threshold**: Intelligent epsilon selection
- **Custom Thresholds**: User-defined reduction levels
- **Preserves Trends**: Maintains critical speed characteristics
- **Cleaner Plots**: Reduces visual clutter in polar plots

```python
# Manual data reduction
reduced_data = analyzer.reduce_data(
    x_col='Engine_Speed_RPM',
    y_col='Bearing_1_Amplitude_mil',
    epsilon=0.01  # Reduction threshold
)
```

### 4. Engine Operational Phases

Automatic identification and labeling of:

- **Startup** (0-20% of speed range)
- **Idle** (20-30% of speed range)
- **Part Load** (30-70% of speed range)
- **Full Load** (70-95% of speed range)
- **Shutdown** (95-100% of speed range)

### 5. Critical Speed Detection

Automatic detection of critical speeds using peak finding algorithms:

- Identifies resonance points in amplitude data
- Reports critical speeds with corresponding amplitudes
- Useful for balancing and rotor dynamics analysis

## Running the Demo

To see all features in action:

```bash
python demo.py
```

Or run the main analyzer:

```bash
python rotor_dynamics_analyzer.py
```

The demo includes:

1. **Basic Usage**: Synthetic data generation and basic polar plots
2. **Excel Reading**: Loading data from Excel files
3. **Data Reduction**: Comparison of different RDP thresholds
4. **Multi-Bearing Analysis**: Simultaneous analysis of multiple bearings
5. **Custom Analysis**: Advanced parameters and settings

## Output Files

The tool generates:

- **Excel Files**: `synthetic_gas_turbine_data.xlsx` (if using synthetic data)
- **Polar Plots**: Interactive matplotlib figures
- **Analysis Reports**: Console output with statistics and critical speeds

## Advanced Usage

### Custom Engine Phases

```python
# Modify engine phase definitions
analyzer.engine_phases = {
    'startup': (0, 15),
    'idle': (15, 25),
    'part_load': (25, 75),
    'full_load': (75, 100),
    'shutdown': (95, 100)
}
```

### Multiple Bearing Analysis

```python
# Analyze multiple bearings
bearings = ['Bearing_1', 'Bearing_2', 'Bearing_3']
for bearing in bearings:
    analyzer.create_polar_plot(
        amplitude_col=f'{bearing}_Amplitude_mil',
        phase_col=f'{bearing}_Phase_deg',
        speed_col='Engine_Speed_RPM',
        title=f'{bearing} Analysis'
    )
```

### Custom RDP Thresholds

```python
# Test different reduction levels
epsilons = [0.001, 0.005, 0.01, 0.02]
for eps in epsilons:
    analyzer.create_polar_plot(
        # ... columns ...
        reduce_data=True,
        epsilon=eps
    )
```

## Applications

This tool is useful for:

- **Rotor Dynamics Engineers**: Analyzing bearing vibration patterns
- **Gas Turbine Maintenance**: Identifying critical speeds and imbalance
- **Research and Development**: Studying rotor behavior during operation
- **Training and Education**: Understanding rotor dynamics concepts
- **Data Visualization**: Creating publication-quality polar plots

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Excel File Errors**: Check that Excel file has proper column names
3. **Empty Plots**: Verify data contains the specified column names
4. **Performance Issues**: Reduce data size or increase RDP epsilon value

### Performance Tips

- Use data reduction for large datasets (>1000 points)
- Adjust epsilon values based on data density
- Consider using fewer columns for faster processing

## Contributing

Feel free to contribute by:

- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new analysis methods

## License

This project is open-source and available under the MIT License. 