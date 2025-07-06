#!/bin/bash

# AI Curriculum Planner Streamlit App Launcher

echo "🎓 Starting AI Curriculum Planner Streamlit App..."
echo "==============================================="

# Check if we're in the correct directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the app directory."
    exit 1
fi

# Check if virtual environment should be activated
if [ -f "../venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Install requirements if needed
echo "📋 Checking requirements..."
pip install -r requirements.txt

# Generate data if needed
echo "🔧 Checking for data files..."
if [ ! -d "../data" ]; then
    echo "📊 Generating sample data..."
    cd ..
    python main.py --generate-data --num-students 50
    cd app
fi

# Launch Streamlit app
echo "🚀 Launching Streamlit app..."
echo ""
echo "The app will open in your default web browser."
echo "If it doesn't open automatically, visit: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app."
echo ""

streamlit run main.py --server.port 8501 --server.address localhost
