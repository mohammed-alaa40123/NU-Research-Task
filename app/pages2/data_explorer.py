"""
Data Explorer Page - Interactive Data Analysis
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.curriculum_graph import create_sample_curriculum
from src.student_simulation import create_student_cohort

@st.cache_data
def load_explorer_data():
    """Load data for exploration"""
    try:
        curriculum = create_sample_curriculum()
        students = create_student_cohort(curriculum, 100)
        
        # Create comprehensive datasets
        course_data = []
        for course in curriculum.graph.nodes():
            course_info = curriculum.get_course_info(course)
            prereqs = curriculum.get_prerequisites(course)
            
            course_data.append({
                'Course': course,
                'Name': course_info.get('name', 'Unknown'),
                'Domain': course_info.get('domain', 'Unknown'),
                'Difficulty': course_info.get('difficulty', 'Unknown'),
                'Credits': course_info.get('credits', 3),
                'Prerequisites': len(prereqs),
                'PrereqList': ', '.join(prereqs) if prereqs else 'None'
            })
        
        course_df = pd.DataFrame(course_data)
        
        # Student data
        student_data = []
        for student in students:
            completed_courses = [c for c, g in student.completed_courses.items() if g >= 2.0]
            
            student_data.append({
                'StudentID': student.student_id,
                'GPA': student.gpa,
                'CurrentTerm': student.current_term,
                'AcademicStanding': student.academic_standing,
                'CoursesCompleted': len(completed_courses),
                'TargetGraduation': student.target_graduation_term,
                'MaxCoursesPerTerm': student.max_courses_per_term,
                **{f'Interest_{domain}': student.interests.get(domain, 0) 
                   for domain in ['AI', 'Security', 'Data Science', 'Software Engineering', 'Systems', 'Theory']}
            })
        
        student_df = pd.DataFrame(student_data)
        
        # Validate data
        if course_df.empty:
            raise ValueError("No course data generated")
        if student_df.empty:
            raise ValueError("No student data generated")
        
        return course_df, student_df, curriculum, students
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty dataframes as fallback
        return pd.DataFrame(), pd.DataFrame(), None, []

def show():
    """Display the data explorer page"""
    
    st.markdown('<h1 class="main-header">🔍 Data Explorer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Interactive Data Analysis and Exploration
    
    Deep dive into the curriculum and student data with interactive filters, visualizations, and statistical analysis.
    """)
    
    try:
        # Load data
        course_df, student_df, curriculum, students = load_explorer_data()
        
        # Check if data loaded successfully
        if course_df.empty or student_df.empty:
            st.error("Failed to load data. Please check the data generation process.")
            st.info("Try running the main application to generate data first.")
            return
        
        # Data selection tabs
        tab1, tab2, tab3 = st.tabs(["📚 Course Data", "👥 Student Data", "🔗 Relationships"])
        
        with tab1:
            st.markdown("### 📚 Course Database Explorer")
            
            # Course filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                domain_filter = st.multiselect(
                    "Filter by Domain:",
                    options=course_df['Domain'].unique(),
                    default=[]
                )
            
            with col2:
                difficulty_filter = st.multiselect(
                    "Filter by Difficulty:",
                    options=course_df['Difficulty'].unique(),
                    default=[]
                )
            
            with col3:
                min_credits = int(course_df['Credits'].min())
                max_credits = int(course_df['Credits'].max())
                
                # Handle case where all courses have same credits
                if min_credits == max_credits:
                    st.write(f"**Credits:** {min_credits}")
                    credit_filter = (min_credits, max_credits)
                else:
                    credit_filter = st.slider(
                        "Credits Range:",
                        min_value=min_credits,
                        max_value=max_credits,
                        value=(min_credits, max_credits)
                    )
            
            # Apply filters
            filtered_courses = course_df.copy()
            
            if domain_filter:
                filtered_courses = filtered_courses[filtered_courses['Domain'].isin(domain_filter)]
            
            if difficulty_filter:
                filtered_courses = filtered_courses[filtered_courses['Difficulty'].isin(difficulty_filter)]
            
            filtered_courses = filtered_courses[
                (filtered_courses['Credits'] >= credit_filter[0]) & 
                (filtered_courses['Credits'] <= credit_filter[1])
            ]
            
            # Course statistics
            st.markdown("#### 📊 Course Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Courses", len(filtered_courses))
            
            with col2:
                avg_credits = filtered_courses['Credits'].mean()
                st.metric("Avg Credits", f"{avg_credits:.1f}")
            
            with col3:
                avg_prereqs = filtered_courses['Prerequisites'].mean()
                st.metric("Avg Prerequisites", f"{avg_prereqs:.1f}")
            
            with col4:
                domains = filtered_courses['Domain'].nunique()
                st.metric("Domains", domains)
            
            # Course visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Domain distribution
                domain_counts = filtered_courses['Domain'].value_counts()
                
                domain_fig = px.pie(
                    values=domain_counts.values,
                    names=domain_counts.index,
                    title="Course Distribution by Domain"
                )
                
                st.plotly_chart(domain_fig, use_container_width=True)
            
            with col2:
                # Difficulty vs Prerequisites
                difficulty_fig = px.scatter(
                    filtered_courses,
                    x='Prerequisites',
                    y='Credits',
                    color='Difficulty',
                    title="Prerequisites vs Credits by Difficulty",
                    hover_data=['Course', 'Domain']
                )
                
                st.plotly_chart(difficulty_fig, use_container_width=True)
            
            # Course data table
            st.markdown("#### 📋 Course Details")
            
            # Search functionality
            search_term = st.text_input("🔍 Search courses:", placeholder="Enter course code or name...")
            
            if search_term:
                filtered_courses = filtered_courses[
                    filtered_courses['Course'].str.contains(search_term, case=False) |
                    filtered_courses['Name'].str.contains(search_term, case=False)
                ]
            
            # Display table
            st.dataframe(
                filtered_courses[['Course', 'Name', 'Domain', 'Difficulty', 'Credits', 'Prerequisites']],
                use_container_width=True
            )
            
            # Download option
            if st.button("📥 Download Course Data"):
                csv = filtered_courses.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="course_data.csv",
                    mime="text/csv"
                )
        
        with tab2:
            st.markdown("### 👥 Student Cohort Explorer")
            
            # Student filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_gpa = float(student_df['GPA'].min())
                max_gpa = float(student_df['GPA'].max())
                
                # Handle case where all students have same GPA (unlikely but possible)
                if abs(min_gpa - max_gpa) < 0.01:  # Very small difference
                    st.write(f"**GPA:** {min_gpa:.2f}")
                    gpa_range = (min_gpa, max_gpa)
                else:
                    gpa_range = st.slider(
                        "GPA Range:",
                        min_value=min_gpa,
                        max_value=max_gpa,
                        value=(min_gpa, max_gpa),
                        step=0.1
                    )
            
            with col2:
                standing_filter = st.multiselect(
                    "Academic Standing:",
                    options=student_df['AcademicStanding'].unique(),
                    default=[]
                )
            
            with col3:
                min_term = int(student_df['CurrentTerm'].min())
                max_term = int(student_df['CurrentTerm'].max())
                
                # Handle case where all students are in same term
                if min_term == max_term:
                    st.write(f"**Current Term:** {min_term}")
                    term_range = (min_term, max_term)
                else:
                    term_range = st.slider(
                        "Current Term:",
                        min_value=min_term,
                        max_value=max_term,
                        value=(min_term, max_term)
                    )
            
            # Apply filters
            filtered_students = student_df.copy()
            
            filtered_students = filtered_students[
                (filtered_students['GPA'] >= gpa_range[0]) & 
                (filtered_students['GPA'] <= gpa_range[1])
            ]
            
            if standing_filter:
                filtered_students = filtered_students[filtered_students['AcademicStanding'].isin(standing_filter)]
            
            filtered_students = filtered_students[
                (filtered_students['CurrentTerm'] >= term_range[0]) & 
                (filtered_students['CurrentTerm'] <= term_range[1])
            ]
            
            # Student statistics
            st.markdown("#### 📊 Cohort Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Students", len(filtered_students))
            
            with col2:
                avg_gpa = filtered_students['GPA'].mean()
                st.metric("Avg GPA", f"{avg_gpa:.2f}")
            
            with col3:
                avg_courses = filtered_students['CoursesCompleted'].mean()
                st.metric("Avg Courses", f"{avg_courses:.1f}")
            
            with col4:
                avg_term = filtered_students['CurrentTerm'].mean()
                st.metric("Avg Term", f"{avg_term:.1f}")
            
            # Student visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # GPA distribution
                gpa_fig = px.histogram(
                    filtered_students,
                    x='GPA',
                    title="GPA Distribution",
                    nbins=20
                )
                
                st.plotly_chart(gpa_fig, use_container_width=True)
            
            with col2:
                # Academic standing
                standing_counts = filtered_students['AcademicStanding'].value_counts()
                
                standing_fig = px.bar(
                    x=standing_counts.index,
                    y=standing_counts.values,
                    title="Academic Standing Distribution"
                )
                
                st.plotly_chart(standing_fig, use_container_width=True)
            
            # Interest analysis
            st.markdown("#### 🎯 Interest Analysis")
            
            interest_cols = [col for col in filtered_students.columns if col.startswith('Interest_')]
            interest_data = filtered_students[interest_cols].mean()
            interest_data.index = [col.replace('Interest_', '') for col in interest_data.index]
            
            interest_fig = px.bar(
                x=interest_data.index,
                y=interest_data.values,
                title="Average Interest by Domain"
            )
            
            st.plotly_chart(interest_fig, use_container_width=True)
            
            # Correlation analysis
            st.markdown("#### 🔗 Correlation Analysis")
            
            numeric_cols = ['GPA', 'CurrentTerm', 'CoursesCompleted'] + interest_cols
            correlation_matrix = filtered_students[numeric_cols].corr()
            
            # Clean column names for display
            display_names = {col: col.replace('Interest_', '') for col in correlation_matrix.columns}
            correlation_matrix = correlation_matrix.rename(columns=display_names, index=display_names)
            
            corr_fig = px.imshow(
                correlation_matrix,
                title="Student Attribute Correlations",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            
            st.plotly_chart(corr_fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔗 Data Relationships")
            
            # Advanced analytics
            st.markdown("#### 📊 Advanced Analytics")
            
            analysis_type = st.selectbox(
                "Choose Analysis:",
                ["Student Performance vs Interest", "Course Difficulty Progression", 
                 "Academic Standing Patterns", "Prerequisites Network"]
            )
            
            if analysis_type == "Student Performance vs Interest":
                # Scatter plot of GPA vs primary interest
                student_interest_data = []
                
                for _, student in student_df.iterrows():
                    interest_cols = [col for col in student_df.columns if col.startswith('Interest_')]
                    interests = {col.replace('Interest_', ''): student[col] for col in interest_cols}
                    primary_interest = max(interests, key=interests.get)
                    primary_score = interests[primary_interest]
                    
                    student_interest_data.append({
                        'GPA': student['GPA'],
                        'PrimaryInterest': primary_interest,
                        'InterestScore': primary_score,
                        'CoursesCompleted': student['CoursesCompleted']
                    })
                
                interest_df = pd.DataFrame(student_interest_data)
                
                interest_perf_fig = px.scatter(
                    interest_df,
                    x='InterestScore',
                    y='GPA',
                    color='PrimaryInterest',
                    size='CoursesCompleted',
                    title="Student Performance vs Primary Interest",
                    hover_data=['CoursesCompleted']
                )
                
                st.plotly_chart(interest_perf_fig, use_container_width=True)
            
            elif analysis_type == "Course Difficulty Progression":
                # Show difficulty progression patterns
                difficulty_order = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
                course_df['DifficultyNum'] = course_df['Difficulty'].map(difficulty_order)
                
                diff_prog_fig = px.scatter(
                    course_df,
                    x='Prerequisites',
                    y='DifficultyNum',
                    color='Domain',
                    title="Course Difficulty vs Prerequisites",
                    hover_data=['Course', 'Credits']
                )
                
                diff_prog_fig.update_yaxis(
                    tickvals=[1, 2, 3],
                    ticktext=['Beginner', 'Intermediate', 'Advanced']
                )
                
                st.plotly_chart(diff_prog_fig, use_container_width=True)
            
            elif analysis_type == "Academic Standing Patterns":
                # Group analysis by academic standing
                standing_analysis = student_df.groupby('AcademicStanding').agg({
                    'GPA': 'mean',
                    'CoursesCompleted': 'mean',
                    'CurrentTerm': 'mean'
                }).reset_index()
                
                standing_fig = px.bar(
                    standing_analysis,
                    x='AcademicStanding',
                    y=['GPA', 'CoursesCompleted', 'CurrentTerm'],
                    title="Academic Metrics by Standing",
                    barmode='group'
                )
                
                st.plotly_chart(standing_fig, use_container_width=True)
            
            elif analysis_type == "Prerequisites Network":
                # Show prerequisite relationships
                st.markdown("#### 🌐 Prerequisites Network Analysis")
                
                # Calculate network metrics
                total_courses = len(course_df)
                courses_with_prereqs = len(course_df[course_df['Prerequisites'] > 0])
                avg_prereqs = course_df['Prerequisites'].mean()
                max_prereqs = course_df['Prerequisites'].max()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Courses", total_courses)
                
                with col2:
                    st.metric("With Prerequisites", courses_with_prereqs)
                
                with col3:
                    st.metric("Avg Prerequisites", f"{avg_prereqs:.1f}")
                
                with col4:
                    st.metric("Max Prerequisites", max_prereqs)
                
                # Prerequisites distribution
                prereq_dist_fig = px.histogram(
                    course_df,
                    x='Prerequisites',
                    title="Distribution of Prerequisite Counts"
                )
                
                st.plotly_chart(prereq_dist_fig, use_container_width=True)
            
            # Data summary
            st.markdown("#### 📈 Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Course Data Summary:**")
                st.dataframe(course_df.describe())
            
            with col2:
                st.markdown("**Student Data Summary:**")
                numeric_student_df = student_df.select_dtypes(include=[np.number])
                st.dataframe(numeric_student_df.describe())
    
    except Exception as e:
        st.error(f"Error loading data explorer: {str(e)}")
        st.info("Please ensure all data is properly generated.")
    
    # Export all data
    st.markdown("### 💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Complete Dataset"):
            try:
                course_df, student_df, _, _ = load_explorer_data()
                
                if not course_df.empty:
                    # Create a combined export
                    with st.spinner("Preparing data export..."):
                        # Course data
                        course_csv = course_df.to_csv(index=False)
                        st.download_button(
                            label="Download Course Data",
                            data=course_csv,
                            file_name="complete_course_data.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("No course data available for export")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        if st.button("📊 Download Student Dataset"):
            try:
                course_df, student_df, _, _ = load_explorer_data()
                
                if not student_df.empty:
                    with st.spinner("Preparing student export..."):
                        student_csv = student_df.to_csv(index=False)
                        st.download_button(
                            label="Download Student Data",
                            data=student_csv,
                            file_name="complete_student_data.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("No student data available for export")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    # Additional information
    with st.expander("ℹ️ Data Explorer Guide"):
        st.markdown("""
        ### How to Use the Data Explorer
        
        **Course Data Tab:**
        - Filter courses by domain, difficulty, and credits
        - Search for specific courses by code or name
        - Analyze prerequisite patterns and distributions
        - Export filtered course data
        
        **Student Data Tab:**
        - Filter students by GPA, academic standing, and term
        - Explore interest patterns and correlations
        - Analyze academic performance distributions
        - Export cohort data for external analysis
        
        **Relationships Tab:**
        - Advanced analytics and cross-domain analysis
        - Performance vs interest correlations
        - Academic progression patterns
        - Network analysis of prerequisites
        
        ### Data Insights
        
        Use this explorer to discover patterns such as:
        - Which domains have the most prerequisite dependencies
        - How student interests correlate with academic performance
        - What academic standing patterns emerge in the cohort
        - How course difficulty relates to prerequisites
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        🔍 Interactive Data Exploration | Filter, Analyze, Export
    </div>
    """, unsafe_allow_html=True)
