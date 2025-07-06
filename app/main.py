"""
AI Curriculum Planner - Streamlit Application
Main entry point for the multi-page Streamlit app
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages2 import (
    home,
    curriculum_graph,
    student_analysis,
    student_dashboard,
    course_recommendations,
    training_metrics,
    data_explorer
)

# Configure the page
st.set_page_config(
    page_title="AI Curriculum Planner",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #A23B72;
        color: white;
    }
    

    
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("🎓 AI Curriculum Planner")
    st.sidebar.markdown("---")
    
    # Navigation menu - ensure this is the ONLY navigation
    pages = {
        "🏠 Home": home,
        "📊 Curriculum Graph": curriculum_graph,
        "👥 Student Analysis": student_analysis,
        "📋 Student Dashboard": student_dashboard,
        "💡 Course Recommendations": course_recommendations,
        "📈 Training Metrics": training_metrics,
        "🔍 Data Explorer": data_explorer
    }
    
    # Page selection - ONLY in sidebar
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        index=0,
        key="main_navigation"  # Unique key to prevent conflicts
    )
    
    # Display selected page
    page_module = pages[selected_page]
    page_module.show()
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application demonstrates an AI-powered curriculum planning system 
    using reinforcement learning for adaptive academic advising.
    
    **Features:**
    - Interactive curriculum visualization
    - Student cohort analysis
    - Personalized course recommendations
    - Training metrics and evaluation
    """)

if __name__ == "__main__":
    main()
