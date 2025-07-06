"""
Student Dashboard Page - Individual Student Profiles
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.curriculum_graph import create_sample_curriculum
from src.student_simulation import create_student_cohort
from src.constraints import AcademicConstraints

@st.cache_data
def load_dashboard_data():
    """Load data for student dashboard"""
    curriculum = create_sample_curriculum()
    students = create_student_cohort(curriculum, 50)  # Smaller set for performance
    constraints = AcademicConstraints(curriculum)
    return curriculum, students, constraints

def create_student_dashboard_viz(student, curriculum, recommendations):
    """Create student dashboard visualization"""
    
    # Create figure with custom subplot layout
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Academic Progress', 'Interest Profile', 'Completed Courses by Domain',
            'GPA Trend (Simulated)', 'Recommended Courses', 'Course Difficulty Timeline',
            'Graduation Path', 'Constraint Compliance', 'Performance Metrics'
        ],
        specs=[
            [{"type": "bar"}, {"type": "polar"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]
        ]
    )
    
    # 1. Academic Progress
    progress_metrics = {
        'Credits Completed': sum(curriculum.get_course_info(c).get('credits', 3) 
                               for c, g in student.completed_courses.items() if g >= 2.0),
        'Current GPA': student.gpa,
        'Courses Passed': len([g for g in student.completed_courses.values() if g >= 2.0]),
        'Current Term': student.current_term
    }
    
    fig.add_trace(
        go.Bar(x=list(progress_metrics.keys()), y=list(progress_metrics.values()),
               marker_color='lightblue'),
        row=1, col=1
    )
    
    # 2. Interest Profile (Radar Chart)
    domains = list(student.interests.keys())
    interest_values = list(student.interests.values())
    
    fig.add_trace(
        go.Scatterpolar(
            r=interest_values,
            theta=domains,
            fill='toself',
            marker_color='green'
        ),
        row=1, col=2
    )
    
    # 3. Completed Courses by Domain
    domain_counts = {}
    for course, grade in student.completed_courses.items():
        if grade >= 2.0:
            course_info = curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Other')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    if domain_counts:
        fig.add_trace(
            go.Pie(labels=list(domain_counts.keys()), 
                   values=list(domain_counts.values())),
            row=1, col=3
        )
    
    # 4. GPA Trend (Simulated)
    terms = list(range(1, student.current_term + 1))
    gpa_trend = [2.5 + (student.gpa - 2.5) * (t / student.current_term) for t in terms]
    
    fig.add_trace(
        go.Scatter(x=terms, y=gpa_trend, mode='lines+markers',
                   line=dict(color='blue'), name='GPA Trend'),
        row=2, col=1
    )
    
    # 5. Recommended Courses
    if recommendations:
        fig.add_trace(
            go.Bar(x=recommendations, y=[1]*len(recommendations),
                   marker_color='orange', text=recommendations, textposition='auto'),
            row=2, col=2
        )
    
    # 6. Course Difficulty Timeline
    completed_courses = [c for c, g in student.completed_courses.items() if g >= 2.0]
    difficulties = []
    for course in completed_courses:
        course_info = curriculum.get_course_info(course)
        difficulty = course_info.get('difficulty', 'Intermediate')
        diff_val = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}[difficulty]
        difficulties.append(diff_val)
    
    if difficulties:
        fig.add_trace(
            go.Scatter(x=list(range(len(difficulties))), y=difficulties,
                       mode='lines+markers', line=dict(color='red')),
            row=2, col=3
        )
    
    # 7. Graduation Path
    remaining_courses = curriculum.get_graduation_path(
        {c for c, g in student.completed_courses.items() if g >= 2.0}
    )
    
    path_domains = {}
    for course in remaining_courses[:10]:  # Show first 10
        course_info = curriculum.get_course_info(course)
        domain = course_info.get('domain', 'Other')
        path_domains[domain] = path_domains.get(domain, 0) + 1
    
    if path_domains:
        fig.add_trace(
            go.Bar(x=list(path_domains.keys()), y=list(path_domains.values()),
                   marker_color='purple'),
            row=3, col=1
        )
    
    # 8. Constraint Compliance
    constraints = AcademicConstraints(curriculum)
    constraint_score = constraints.get_constraint_score(recommendations, student)
    
    compliance_data = {
        'Constraint Score': constraint_score,
        'Course Load': len(recommendations) / student.max_courses_per_term if recommendations else 0,
        'Interest Alignment': np.mean([student.interests.get(
            curriculum.get_course_info(c).get('domain', 'Theory'), 0.3
        ) for c in recommendations]) if recommendations else 0
    }
    
    fig.add_trace(
        go.Bar(x=list(compliance_data.keys()), y=list(compliance_data.values()),
               marker_color='teal'),
        row=3, col=2
    )
    
    # 9. Performance Metrics Table
    metrics_data = [
        ['Student ID', student.student_id],
        ['Current GPA', f"{student.gpa:.2f}"],
        ['Academic Standing', student.academic_standing],
        ['Credits Completed', f"{progress_metrics['Credits Completed']}"],
        ['Graduation Target', f"Term {student.target_graduation_term}"],
        ['Recommended Courses', f"{len(recommendations) if recommendations else 0}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=[[row[0] for row in metrics_data],
                             [row[1] for row in metrics_data]])
        ),
        row=3, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"Academic Dashboard: {student.student_id}",
        height=1200,
        showlegend=False
    )
    
    return fig

def show():
    """Display the student dashboard page"""
    
    st.markdown('<h1 class="main-header">📋 Student Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Individual Student Profile Analysis
    
    Comprehensive view of individual student academic progress, interests, and personalized recommendations.
    """)
    
    try:
        # Load data
        curriculum, students, constraints = load_dashboard_data()
        
        # Student selector
        st.markdown("### 👤 Select Student")
        
        student_options = {f"{s.student_id} (GPA: {s.gpa:.2f}, Term: {s.current_term})": s for s in students}
        
        selected_student_key = st.selectbox(
            "Choose a student to analyze:",
            list(student_options.keys()),
            index=0
        )
        
        selected_student = student_options[selected_student_key]
        
        # Generate recommendations for selected student
        passed_courses = {c for c, g in selected_student.completed_courses.items() if g >= 2.0}
        eligible_courses = curriculum.get_eligible_courses(passed_courses)
        recommendations = eligible_courses[:4] if len(eligible_courses) >= 4 else eligible_courses
        
        # Student overview cards
        st.markdown("### 📊 Student Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Student ID", selected_student.student_id)
        
        with col2:
            st.metric("Current GPA", f"{selected_student.gpa:.2f}")
        
        with col3:
            st.metric("Current Term", selected_student.current_term)
        
        with col4:
            st.metric("Academic Standing", selected_student.academic_standing)
        
        # Detailed student information
        st.markdown("### 📝 Detailed Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📚 Academic Information")
            st.write(f"**Target Graduation:** Term {selected_student.target_graduation_term}")
            st.write(f"**Max Courses per Term:** {selected_student.max_courses_per_term}")
            st.write(f"**Courses Completed:** {len(selected_student.completed_courses)}")
            
            # Show completed courses
            completed_passing = [c for c, g in selected_student.completed_courses.items() if g >= 2.0]
            st.write(f"**Passing Grades:** {len(completed_passing)}")
            
            if st.checkbox("Show Completed Courses"):
                for course, grade in selected_student.completed_courses.items():
                    status = "✅" if grade >= 2.0 else "❌"
                    st.write(f"{status} {course}: {grade:.1f}")
        
        with col2:
            st.markdown("#### 🎯 Interest Profile")
            
            # Interest breakdown
            for domain, interest in selected_student.interests.items():
                st.write(f"**{domain}:** {interest:.2f}")
            
            # Top interests
            top_interests = sorted(selected_student.interests.items(), key=lambda x: x[1], reverse=True)[:3]
            st.write("**Top 3 Interests:**")
            for i, (domain, score) in enumerate(top_interests, 1):
                st.write(f"{i}. {domain} ({score:.2f})")
        
        # Course recommendations
        st.markdown("### 💡 Course Recommendations")
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommended courses:")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                for i, course in enumerate(recommendations):
                    course_info = curriculum.get_course_info(course)
                    st.write(f"**{i+1}. {course}**")
                    st.write(f"   Domain: {course_info.get('domain', 'Unknown')}")
                    st.write(f"   Difficulty: {course_info.get('difficulty', 'Unknown')}")
            
            with rec_col2:
                # Recommendation analysis
                if recommendations:
                    # Interest alignment
                    alignments = []
                    for course in recommendations:
                        course_info = curriculum.get_course_info(course)
                        domain = course_info.get('domain', 'Theory')
                        alignment = selected_student.interests.get(domain, 0.3)
                        alignments.append(alignment)
                    
                    avg_alignment = np.mean(alignments)
                    st.metric("Interest Alignment", f"{avg_alignment:.2f}")
                    
                    # Validate recommendations
                    is_valid, violations = constraints.validate_course_selection(recommendations, selected_student)
                    
                    if is_valid:
                        st.success("✅ All recommendations meet constraints")
                    else:
                        st.warning(f"⚠️ {len(violations)} constraint violations")
                        for violation in violations:
                            st.write(f"- {violation}")
        else:
            st.info("No eligible courses found for recommendations.")
        
        # Main dashboard visualization
        st.markdown("### 📊 Comprehensive Dashboard")
        
        # Create and display dashboard
        dashboard_fig = create_student_dashboard_viz(selected_student, curriculum, recommendations)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Additional analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Domain Progress")
            
            # Calculate progress by domain
            domain_progress = {}
            for course, grade in selected_student.completed_courses.items():
                if grade >= 2.0:
                    course_info = curriculum.get_course_info(course)
                    domain = course_info.get('domain', 'Other')
                    domain_progress[domain] = domain_progress.get(domain, 0) + 1
            
            # Show progress
            all_domains = ['AI', 'Security', 'Data Science', 'Software Engineering', 'Systems', 'Theory']
            for domain in all_domains:
                completed = domain_progress.get(domain, 0)
                st.write(f"**{domain}:** {completed} courses")
        
        with col2:
            st.markdown("#### 📈 Academic Trajectory")
            
            # Calculate some trajectory metrics
            current_credits = sum(
                curriculum.get_course_info(course).get('credits', 3)
                for course, grade in selected_student.completed_courses.items()
                if grade >= 2.0
            )
            
            st.write(f"**Current Credits:** {current_credits}")
            st.write(f"**Credits to Graduate:** {120 - current_credits}")
            
            if current_credits > 0:
                progress_pct = (current_credits / 120) * 100
                st.write(f"**Graduation Progress:** {progress_pct:.1f}%")
                
                # Estimated completion
                avg_credits_per_term = current_credits / selected_student.current_term if selected_student.current_term > 0 else 0
                if avg_credits_per_term > 0:
                    terms_remaining = (120 - current_credits) / avg_credits_per_term
                    estimated_graduation = selected_student.current_term + terms_remaining
                    st.write(f"**Estimated Graduation:** Term {estimated_graduation:.1f}")
    
    except Exception as e:
        st.error(f"Error loading student dashboard: {str(e)}")
        st.info("Please ensure the student data is properly generated.")
    
    # Additional information
    with st.expander("ℹ️ About Student Dashboards"):
        st.markdown("""
        ### Dashboard Components
        
        **Academic Progress:** Current academic metrics and milestones
        
        **Interest Profile:** Radar chart showing domain preferences
        
        **Course Completion:** Distribution of completed courses by domain
        
        **GPA Trend:** Simulated academic performance over time
        
        **Recommendations:** AI-generated course suggestions
        
        **Difficulty Timeline:** Progression through course difficulty levels
        
        **Graduation Path:** Remaining requirements by domain
        
        **Constraint Compliance:** Validation of recommendations against academic rules
        
        **Performance Metrics:** Key academic indicators and targets
        
        ### Recommendation System
        
        Course recommendations are generated using:
        - Prerequisite validation
        - Interest alignment scoring
        - Academic constraint checking
        - Course load optimization
        """)
