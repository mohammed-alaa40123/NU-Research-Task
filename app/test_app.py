"""
Simple test script to verify Streamlit app components
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test core dependencies
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
        
        import pandas as pd
        import numpy as np
        print("✅ Pandas/NumPy imported successfully")
        
        import networkx as nx
        print("✅ NetworkX imported successfully")
        
        # Test source modules
        from src.curriculum_graph import create_sample_curriculum
        from src.student_simulation import create_student_cohort
        from src.constraints import AcademicConstraints
        print("✅ Source modules imported successfully")
        
        # Test page modules
        from pages2 import home, curriculum_graph, student_analysis
        print("✅ Page modules imported successfully")
        
        return True
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_generation():
    """Test if data can be generated successfully"""
    try:
        print("\nTesting data generation...")
        
        from src.curriculum_graph import create_sample_curriculum
        from src.student_simulation import create_student_cohort
        
        # Create curriculum
        curriculum = create_sample_curriculum()
        print(f"✅ Curriculum created with {len(curriculum.graph.nodes())} courses")
        
        # Create students
        students = create_student_cohort(curriculum, 10)
        print(f"✅ Student cohort created with {len(students)} students")
        
        return True
    
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_visualizations():
    """Test if visualizations can be created"""
    try:
        print("\nTesting visualizations...")
        
        import plotly.graph_objects as go
        
        # Create a simple test plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines'))
        fig.update_layout(title="Test Plot")
        
        print("✅ Plotly visualization created successfully")
        
        return True
    
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 AI Curriculum Planner - App Test Suite")
    print("==========================================")
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_data_generation()
    all_tests_passed &= test_visualizations()
    
    print("\n" + "="*50)
    
    if all_tests_passed:
        print("🎉 All tests passed! The app should work correctly.")
        print("\nTo launch the app, run:")
        print("  streamlit run main.py")
        print("or")
        print("  ./run_app.sh  (Linux/macOS)")
        print("  run_app.bat   (Windows)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure you're in the correct directory")
        print("3. Check Python version (3.8+ required)")
    
    print("\n🎓 Happy exploring!")
