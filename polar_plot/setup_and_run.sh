#!/bin/bash

echo "🚀 Gas Turbine Rotor Dynamics Analyzer - Setup & Run"
echo "======================================================="

# Check if virtual environment exists
if [ ! -d "notebook_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv notebook_env
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "Please ensure Python 3 is installed"
        exit 1
    fi
    echo "✅ Virtual environment created successfully"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source notebook_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements.txt

# Install Jupyter
echo "📓 Installing Jupyter notebook..."
pip install jupyter notebook ipywidgets

# Test the setup
echo "🧪 Testing the setup..."
python test_notebook.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "To run the Jupyter notebook:"
    echo "1. Activate the virtual environment:"
    echo "   source notebook_env/bin/activate"
    echo ""
    echo "2. Start Jupyter notebook:"
    echo "   jupyter notebook rotor_dynamics_analysis.ipynb"
    echo ""
    echo "Or run the demo scripts:"
    echo "   python quick_start.py"
    echo "   python demo.py"
    echo ""
    echo "📂 Files created:"
    echo "   • rotor_dynamics_analysis.ipynb - Main notebook"
    echo "   • synthetic_gas_turbine_data.xlsx - Sample data"
    echo "   • rotor_dynamics_analyzer.py - Core analyzer"
    echo "   • test_plot.png - Test visualization"
    echo ""
else
    echo "❌ Setup test failed. Please check the error messages above."
    exit 1
fi 