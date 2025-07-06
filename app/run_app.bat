@echo off
echo 🎓 Starting AI Curriculum Planner Streamlit App...
echo ===============================================

REM Check if we're in the correct directory
if not exist "main.py" (
    echo ❌ Error: main.py not found. Please run this script from the app directory.
    pause
    exit /b 1
)

REM Check if virtual environment should be activated
if exist "..\venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call ..\venv\Scripts\activate.bat
)

REM Install requirements if needed
echo 📋 Checking requirements...
pip install -r requirements.txt

REM Generate data if needed
echo 🔧 Checking for data files...
if not exist "..\data" (
    echo 📊 Generating sample data...
    cd ..
    python main.py --generate-data --num-students 50
    cd app
)

REM Launch Streamlit app
echo 🚀 Launching Streamlit app...
echo.
echo The app will open in your default web browser.
echo If it doesn't open automatically, visit: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app.
echo.

streamlit run main.py --server.port 8501 --server.address localhost

pause
