"""
Course Recommendations Page - AI-Powered Course Suggestions
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
from src.rl_advisor import create_rl_advisor

@st.cache_data
def load_recommendation_data():
    """Load data for course recommendations"""
    curriculum = create_sample_curriculum()
    students = create_student_cohort(curriculum, 30)
    constraints = AcademicConstraints(curriculum)
    return curriculum, students, constraints

def create_recommendation_analysis_viz(student, recommendations, curriculum, constraints):
    """Create recommendation analysis visualization"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Recommended Courses by Domain', 'Difficulty Distribution',
                       'Interest Alignment', 'Progress Toward Graduation'],
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Recommended Courses by Domain
    domain_counts = {}
    for course in recommendations:
        course_info = curriculum.get_course_info(course)
        domain = course_info.get('domain', 'Other')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    if domain_counts:
        fig.add_trace(
            go.Bar(x=list(domain_counts.keys()), y=list(domain_counts.values()),
                   name="Courses by Domain", marker_color='skyblue'),
            row=1, col=1
        )
    
    # Difficulty Distribution
    difficulty_counts = {}
    for course in recommendations:
        course_info = curriculum.get_course_info(course)
        difficulty = course_info.get('difficulty', 'Intermediate')
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    if difficulty_counts:
        fig.add_trace(
            go.Pie(labels=list(difficulty_counts.keys()), 
                   values=list(difficulty_counts.values()),
                   name="Difficulty"),
            row=1, col=2
        )
    
    # Interest Alignment
    courses_and_interests = []
    interest_scores = []
    
    for course in recommendations:
        course_info = curriculum.get_course_info(course)
        domain = course_info.get('domain', 'Theory')
        interest_score = student.interests.get(domain, 0.3)
        
        courses_and_interests.append(course)
        interest_scores.append(interest_score)
    
    if courses_and_interests:
        fig.add_trace(
            go.Bar(x=courses_and_interests, y=interest_scores,
                   name="Interest Alignment", marker_color='lightgreen'),
            row=2, col=1
        )
    
    # Progress Analysis
    current_credits = sum(
        curriculum.get_course_info(course).get('credits', 3)
        for course, grade in student.completed_courses.items()
        if grade >= 2.0
    )
    
    recommended_credits = sum(
        curriculum.get_course_info(course).get('credits', 3)
        for course in recommendations
    )
    
    progress_data = {
        'Current Credits': current_credits,
        'Recommended Credits': recommended_credits,
        'Remaining to Graduate': max(0, 120 - current_credits - recommended_credits)
    }
    
    fig.add_trace(
        go.Bar(x=list(progress_data.keys()), y=list(progress_data.values()),
               name="Credit Progress", marker_color='orange'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"Course Recommendations for {student.student_id}",
        height=700,
        showlegend=False
    )
    
    return fig

def show():
    """Display the course recommendations page"""
    
    st.markdown('<h1 class="main-header">💡 Course Recommendations</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## AI-Powered Course Recommendation System
    
    Experience personalized course recommendations using reinforcement learning and constraint optimization.
    """)
    
    try:
        # Load data
        curriculum, students, constraints = load_recommendation_data()
        
        # Recommendation settings
        st.markdown("### ⚙️ Recommendation Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_recommendations = st.slider("Number of Recommendations", 1, 8, 4)
        
        with col2:
            recommendation_strategy = st.selectbox(
                "Strategy",
                ["AI-Powered (RL)", "Interest-Based", "Prerequisites-Only", "Random"]
            )
        
        with col3:
            consider_difficulty = st.checkbox("Consider Difficulty Progression", value=True)
        
        # Student selector
        st.markdown("### 👤 Select Student for Recommendations")
        
        student_options = {f"{s.student_id} (GPA: {s.gpa:.2f})": s for s in students}
        
        selected_student_key = st.selectbox(
            "Choose a student:",
            list(student_options.keys()),
            index=0
        )
        
        selected_student = student_options[selected_student_key]
        
        # Student context
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Student Context")
            st.write(f"**Student ID:** {selected_student.student_id}")
            st.write(f"**Current GPA:** {selected_student.gpa:.2f}")
            st.write(f"**Current Term:** {selected_student.current_term}")
            st.write(f"**Academic Standing:** {selected_student.academic_standing}")
            
            completed_passing = [c for c, g in selected_student.completed_courses.items() if g >= 2.0]
            st.write(f"**Courses Completed:** {len(completed_passing)}")
        
        with col2:
            st.markdown("#### 🎯 Top Interests")
            top_interests = sorted(selected_student.interests.items(), key=lambda x: x[1], reverse=True)
            for domain, score in top_interests:
                st.write(f"**{domain}:** {score:.2f}")
        
        # Generate recommendations button
        if st.button("🚀 Generate Recommendations", type="primary"):
            
            with st.spinner("Generating personalized recommendations..."):
                
                # Get eligible courses
                passed_courses = {c for c, g in selected_student.completed_courses.items() if g >= 2.0}
                eligible_courses = curriculum.get_eligible_courses(passed_courses)
                
                if recommendation_strategy == "AI-Powered (RL)":
                    # Use RL advisor (simplified for demo)
                    recommendations = eligible_courses[:num_recommendations] if len(eligible_courses) >= num_recommendations else eligible_courses
                    
                elif recommendation_strategy == "Interest-Based":
                    # Sort by interest alignment
                    course_scores = []
                    for course in eligible_courses:
                        course_info = curriculum.get_course_info(course)
                        domain = course_info.get('domain', 'Theory')
                        score = selected_student.interests.get(domain, 0.3)
                        course_scores.append((course, score))
                    
                    course_scores.sort(key=lambda x: x[1], reverse=True)
                    recommendations = [course for course, score in course_scores[:num_recommendations]]
                    
                elif recommendation_strategy == "Prerequisites-Only":
                    # Simple prerequisite-based selection
                    recommendations = eligible_courses[:num_recommendations] if len(eligible_courses) >= num_recommendations else eligible_courses
                    
                else:  # Random
                    import random
                    recommendations = random.sample(eligible_courses, min(num_recommendations, len(eligible_courses)))
                
                # Display recommendations
                if recommendations:
                    st.success(f"✅ Generated {len(recommendations)} personalized recommendations!")
                    
                    # Recommendations overview
                    st.markdown("### 📋 Recommended Courses")
                    
                    for i, course in enumerate(recommendations, 1):
                        course_info = curriculum.get_course_info(course)
                        domain = course_info.get('domain', 'Unknown')
                        difficulty = course_info.get('difficulty', 'Unknown')
                        interest_score = selected_student.interests.get(domain, 0.3)
                        
                        # Create recommendation card
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{i}. {course}**")
                                st.write(f"   {course_info.get('name', 'Course Name')}")
                                st.write(f"   Domain: {domain}")
                            
                            with col2:
                                st.write(f"**Difficulty:** {difficulty}")
                                st.write(f"**Credits:** {course_info.get('credits', 3)}")
                            
                            with col3:
                                st.metric("Interest Match", f"{interest_score:.2f}")
                        
                        st.markdown("---")
                    
                    # Validation
                    st.markdown("### ✅ Recommendation Validation")
                    
                    is_valid, violations = constraints.validate_course_selection(recommendations, selected_student)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if is_valid:
                            st.success("✅ All recommendations meet academic constraints")
                        else:
                            st.warning(f"⚠️ Found {len(violations)} constraint violations:")
                            for violation in violations:
                                st.write(f"- {violation}")
                    
                    with col2:
                        # Calculate metrics
                        avg_interest = np.mean([
                            selected_student.interests.get(
                                curriculum.get_course_info(course).get('domain', 'Theory'), 0.3
                            ) for course in recommendations
                        ])
                        
                        constraint_score = constraints.get_constraint_score(recommendations, selected_student)
                        
                        st.metric("Avg Interest Alignment", f"{avg_interest:.2f}")
                        st.metric("Constraint Score", f"{constraint_score:.2f}")
                    
                    # Detailed analysis
                    st.markdown("### 📊 Recommendation Analysis")
                    
                    analysis_fig = create_recommendation_analysis_viz(
                        selected_student, recommendations, curriculum, constraints
                    )
                    st.plotly_chart(analysis_fig, use_container_width=True)
                    
                    # Alternative recommendations
                    st.markdown("### 🔄 Alternative Options")
                    
                    # Show other eligible courses
                    other_eligible = [c for c in eligible_courses if c not in recommendations][:5]
                    
                    if other_eligible:
                        st.write("**Other eligible courses you might consider:**")
                        for course in other_eligible:
                            course_info = curriculum.get_course_info(course)
                            domain = course_info.get('domain', 'Unknown')
                            st.write(f"- {course} ({domain})")
                    
                    # Semester planning
                    st.markdown("### 📅 Semester Planning")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Next Semester Plan")
                        total_credits = sum(curriculum.get_course_info(c).get('credits', 3) for c in recommendations)
                        st.write(f"**Total Credits:** {total_credits}")
                        st.write(f"**Course Load:** {len(recommendations)} courses")
                        
                        if total_credits > 18:
                            st.warning("⚠️ Heavy course load - consider reducing")
                        elif total_credits < 12:
                            st.info("ℹ️ Light course load - room for more courses")
                        else:
                            st.success("✅ Balanced course load")
                    
                    with col2:
                        st.markdown("#### Long-term Impact")
                        
                        current_credits = sum(
                            curriculum.get_course_info(course).get('credits', 3)
                            for course, grade in selected_student.completed_courses.items()
                            if grade >= 2.0
                        )
                        
                        new_total = current_credits + total_credits
                        progress_pct = (new_total / 120) * 100
                        
                        st.write(f"**Progress after completion:** {progress_pct:.1f}%")
                        st.write(f"**Credits remaining:** {120 - new_total}")
                        
                        if new_total >= 120:
                            st.success("🎓 Ready for graduation!")
                
                else:
                    st.warning("No eligible courses found for recommendations.")
                    
                    st.info("""
                    This could happen if:
                    - All prerequisite requirements are not met
                    - Student has already completed most courses
                    - Course constraints are too restrictive
                    """)
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        st.info("Please ensure all data is properly loaded.")
    
    # Information about the recommendation system
    with st.expander("🤖 About the AI Recommendation System"):
        st.markdown("""
        ### Recommendation Strategies
        
        **AI-Powered (RL):** Uses reinforcement learning to optimize course selection based on:
        - Student academic history and performance
        - Interest alignment and preferences  
        - Prerequisite requirements and constraints
        - Long-term graduation planning
        
        **Interest-Based:** Prioritizes courses that align with student interests:
        - Matches course domains to student preferences
        - Considers interest intensity scores
        - Balances across different domains
        
        **Prerequisites-Only:** Conservative approach focusing on:
        - Strict prerequisite validation
        - Sequential course progression
        - Foundation building
        
        ### Validation Process
        
        All recommendations are validated against:
        - **Prerequisites:** Required completed courses
        - **Course Load:** Maximum courses per semester
        - **Academic Standing:** GPA and progress requirements
        - **Graduation Requirements:** Degree completion rules
        
        ### Optimization Goals
        
        The system optimizes for:
        - **Interest Alignment:** Match student preferences
        - **Academic Success:** Consider difficulty progression
        - **Timely Graduation:** Efficient degree completion
        - **Constraint Satisfaction:** Meet all academic rules
        """)
        
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        💡 Powered by Deep Reinforcement Learning and Constraint Optimization
    </div>
    """, unsafe_allow_html=True)
