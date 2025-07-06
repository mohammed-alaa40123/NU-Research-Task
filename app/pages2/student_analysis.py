"""
Student Analysis Page - Cohort Analytics and Distributions
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.curriculum_graph import create_sample_curriculum
from src.student_simulation import create_student_cohort

@st.cache_data
def load_student_data():
    """Load and cache student cohort data"""
    curriculum = create_sample_curriculum()
    students = create_student_cohort(curriculum, 100)
    return students, curriculum

@st.cache_data
def create_student_analysis_plots():
    """Create student analysis visualizations"""
    students, curriculum = load_student_data()
    
    # Prepare data
    gpas = [s.gpa for s in students]
    courses_completed = [len(s.completed_courses) for s in students]
    terms = [s.current_term for s in students]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['GPA Distribution', 'Courses Completed', 'Term Distribution',
                       'Interest Distribution', 'Academic Standing', 'Domain Preferences'],
        specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "pie"}, {"type": "heatmap"}]]
    )
    
    # GPA Distribution
    fig.add_trace(
        go.Histogram(x=gpas, nbinsx=20, name="GPA", marker_color='lightblue'),
        row=1, col=1
    )
    
    # Courses Completed
    fig.add_trace(
        go.Histogram(x=courses_completed, nbinsx=15, name="Courses", marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Term Distribution
    fig.add_trace(
        go.Histogram(x=terms, nbinsx=10, name="Terms", marker_color='lightcoral'),
        row=1, col=3
    )
    
    # Interest Distribution
    domains = ['AI', 'Security', 'Data Science', 'Software Engineering', 'Systems', 'Theory']
    avg_interests = []
    for domain in domains:
        avg_interest = np.mean([s.interests.get(domain, 0) for s in students])
        avg_interests.append(avg_interest)
    
    fig.add_trace(
        go.Bar(x=domains, y=avg_interests, name="Avg Interest", marker_color='gold'),
        row=2, col=1
    )
    
    # Academic Standing
    standing_counts = {}
    for s in students:
        standing_counts[s.academic_standing] = standing_counts.get(s.academic_standing, 0) + 1
    
    fig.add_trace(
        go.Pie(labels=list(standing_counts.keys()), values=list(standing_counts.values()),
               name="Academic Standing"),
        row=2, col=2
    )
    
    # Domain Preferences Heatmap
    interest_matrix = []
    for domain in domains:
        domain_interests = [s.interests.get(domain, 0) for s in students[:20]]  # Limit for readability
        interest_matrix.append(domain_interests)
    
    fig.add_trace(
        go.Heatmap(z=interest_matrix, y=domains,
                   x=[f"S{i+1}" for i in range(20)],
                   colorscale='Viridis', name="Interests"),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="Student Cohort Analysis",
        height=800,
        showlegend=False
    )
    
    return fig, students

def show():
    """Display the student analysis page"""
    
    st.markdown('<h1 class="main-header">👥 Student Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Student Cohort Analytics
    
    Comprehensive analysis of the simulated student cohort including academic performance,
    course completion patterns, and interest distributions across different domains.
    """)
    
    try:
        # Load data
        fig, students = create_student_analysis_plots()
        
        # Overview metrics
        st.markdown("### 📊 Cohort Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(students))
        
        with col2:
            avg_gpa = np.mean([s.gpa for s in students])
            st.metric("Average GPA", f"{avg_gpa:.2f}")
        
        with col3:
            avg_courses = np.mean([len(s.completed_courses) for s in students])
            st.metric("Avg Courses Completed", f"{avg_courses:.1f}")
        
        with col4:
            avg_term = np.mean([s.current_term for s in students])
            st.metric("Average Term", f"{avg_term:.1f}")
        
        # Main analysis plots
        st.markdown("### 📈 Distribution Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdowns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Academic Standing Breakdown")
            
            standing_data = {}
            for student in students:
                standing = student.academic_standing
                standing_data[standing] = standing_data.get(standing, 0) + 1
            
            standing_df = pd.DataFrame(
                list(standing_data.items()),
                columns=['Academic Standing', 'Count']
            )
            
            standing_fig = px.bar(
                standing_df,
                x='Academic Standing',
                y='Count',
                color='Academic Standing',
                title="Students by Academic Standing"
            )
            
            st.plotly_chart(standing_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📚 Course Load Distribution")
            
            # Calculate course load categories
            course_loads = [len(s.completed_courses) for s in students]
            load_categories = []
            
            for load in course_loads:
                if load < 10:
                    load_categories.append('Light (< 10)')
                elif load < 20:
                    load_categories.append('Moderate (10-19)')
                elif load < 30:
                    load_categories.append('Heavy (20-29)')
                else:
                    load_categories.append('Very Heavy (30+)')
            
            load_counts = {}
            for category in load_categories:
                load_counts[category] = load_counts.get(category, 0) + 1
            
            load_fig = go.Figure(data=[
                go.Pie(
                    labels=list(load_counts.keys()),
                    values=list(load_counts.values()),
                    hole=0.3
                )
            ])
            
            load_fig.update_layout(title="Course Load Distribution")
            st.plotly_chart(load_fig, use_container_width=True)
        
        # Interest patterns analysis
        st.markdown("### 🧠 Interest Pattern Analysis")
        
        # Calculate interest correlations
        domains = ['AI', 'Security', 'Data Science', 'Software Engineering', 'Systems', 'Theory']
        interest_data = []
        
        for student in students:
            student_interests = [student.interests.get(domain, 0) for domain in domains]
            interest_data.append(student_interests)
        
        interest_df = pd.DataFrame(interest_data, columns=domains)
        correlation_matrix = interest_df.corr()
        
        # Create correlation heatmap
        corr_fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0
        ))
        
        corr_fig.update_layout(
            title="Domain Interest Correlations",
            height=500
        )
        
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Performance analysis
        st.markdown("### 📊 Performance Analysis")
        
        # GPA vs Course Load
        gpa_data = [s.gpa for s in students]
        course_count_data = [len(s.completed_courses) for s in students]
        term_data = [s.current_term for s in students]
        
        performance_fig = go.Figure()
        
        performance_fig.add_trace(go.Scatter(
            x=course_count_data,
            y=gpa_data,
            mode='markers',
            marker=dict(
                size=8,
                color=term_data,
                colorscale='Viridis',
                colorbar=dict(title="Current Term"),
                opacity=0.7
            ),
            text=[f"Student: {s.student_id}<br>Term: {s.current_term}" for s in students],
            hoverinfo='text',
            name='Students'
        ))
        
        performance_fig.update_layout(
            title="GPA vs Course Completion (Color = Current Term)",
            xaxis_title="Courses Completed",
            yaxis_title="GPA",
            height=500
        )
        
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Student data table
        st.markdown("### 📋 Student Data Sample")
        
        # Create sample data table
        sample_data = []
        for i, student in enumerate(students[:10]):  # Show first 10 students
            sample_data.append({
                'Student ID': student.student_id,
                'GPA': f"{student.gpa:.2f}",
                'Current Term': student.current_term,
                'Courses Completed': len(student.completed_courses),
                'Academic Standing': student.academic_standing,
                'Top Interest': max(student.interests, key=student.interests.get)
            })
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        # Download option
        if st.button("📥 Download Full Cohort Data"):
            full_data = []
            for student in students:
                full_data.append({
                    'Student ID': student.student_id,
                    'GPA': student.gpa,
                    'Current Term': student.current_term,
                    'Courses Completed': len(student.completed_courses),
                    'Academic Standing': student.academic_standing,
                    **{f'Interest_{domain}': student.interests.get(domain, 0) for domain in domains}
                })
            
            full_df = pd.DataFrame(full_data)
            csv = full_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="student_cohort_data.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error loading student analysis: {str(e)}")
        st.info("Please ensure the student data is properly generated.")
    
    # Additional information
    with st.expander("ℹ️ About Student Simulation"):
        st.markdown("""
        ### Student Profile Generation
        
        The student cohort is generated using realistic simulation parameters:
        
        **Academic Metrics:**
        - GPA distribution based on typical university patterns
        - Course completion rates varying by academic standing
        - Term progression with realistic timelines
        
        **Interest Modeling:**
        - Domain preferences with realistic correlations
        - Interest evolution over academic progression
        - Influence on course selection patterns
        
        **Diversity Factors:**
        - Academic standing distribution
        - Varied completion rates and timelines
        - Different interest combinations and intensities
        
        ### Analysis Insights
        
        The visualizations reveal patterns in:
        - **Academic Performance**: GPA distributions and trends
        - **Course Progression**: Completion patterns and load preferences
        - **Interest Alignment**: Domain preferences and correlations
        - **Timeline Patterns**: Term progression and completion rates
        """)
