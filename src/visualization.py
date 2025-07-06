"""
Visualization Module

This module provides visualization tools for:
- Curriculum graph structure
- Student progress and recommendations
- Training metrics and performance analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

try:
    from .curriculum_graph import CurriculumGraph
    from .student_simulation import StudentProfile
    from .constraints import AcademicConstraints
except ImportError:
    from curriculum_graph import CurriculumGraph
    from student_simulation import StudentProfile
    from constraints import AcademicConstraints


class CurriculumVisualizer:
    """Visualization tools for curriculum and student data"""
    
    def __init__(self, curriculum: CurriculumGraph):
        self.curriculum = curriculum
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_curriculum_graph(self, save_path: Optional[str] = None, 
                            interactive: bool = True) -> None:
        """Create an interactive visualization of the curriculum graph"""
        
        if interactive:
            self._plot_interactive_curriculum(save_path)
        else:
            self._plot_static_curriculum(save_path)
    
    def _plot_interactive_curriculum(self, save_path: Optional[str] = None) -> None:
        """Create interactive curriculum graph with Plotly"""
        G = self.curriculum.graph
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        domain_colors = {
            'AI': '#FF6B6B',
            'Security': '#4ECDC4', 
            'Data Science': '#45B7D1',
            'Software Engineering': '#96CEB4',
            'Systems': '#FFEAA7',
            'Theory': '#DDA0DD'
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            course_info = self.curriculum.get_course_info(node)
            domain = course_info.get('domain', 'Other')
            difficulty = course_info.get('difficulty', 'Intermediate')
            credits = course_info.get('credits', 3)
            
            # Color by domain
            node_color.append(domain_colors.get(domain, '#95A5A6'))
            
            # Size by difficulty
            size_map = {'Beginner': 20, 'Intermediate': 30, 'Advanced': 40}
            node_size.append(size_map.get(difficulty, 25))
            
            # Hover text
            prereqs = self.curriculum.get_prerequisites(node)
            text = f"<b>{node}</b><br>" + \
                   f"Name: {course_info.get('name', 'Unknown')}<br>" + \
                   f"Domain: {domain}<br>" + \
                   f"Difficulty: {difficulty}<br>" + \
                   f"Credits: {credits}<br>" + \
                   f"Prerequisites: {', '.join(prereqs) if prereqs else 'None'}"
            node_text.append(text)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Prerequisites'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            text=[node for node in G.nodes()],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertext=node_text,
            hoverinfo='text',
            name='Courses'
        ))
        
        # Update layout
        fig.update_layout(
            title="Computer Science Curriculum Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Node color = Domain, Node size = Difficulty level",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def _plot_static_curriculum(self, save_path: Optional[str] = None) -> None:
        """Create static curriculum graph with matplotlib"""
        G = self.curriculum.graph
        
        plt.figure(figsize=(16, 12))
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color nodes by domain
        domain_colors = {
            'AI': 'red', 'Security': 'cyan', 'Data Science': 'blue',
            'Software Engineering': 'green', 'Systems': 'yellow', 'Theory': 'purple'
        }
        
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            course_info = self.curriculum.get_course_info(node)
            domain = course_info.get('domain', 'Other')
            difficulty = course_info.get('difficulty', 'Intermediate')
            
            node_colors.append(domain_colors.get(domain, 'gray'))
            
            size_map = {'Beginner': 300, 'Intermediate': 500, 'Advanced': 700}
            node_sizes.append(size_map.get(difficulty, 400))
        
        # Draw graph
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.8)
        
        plt.title("Computer Science Curriculum Graph", size=16, fontweight='bold')
        
        # Add legend
        legend_elements = []
        for domain, color in domain_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=domain))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_student_progress(self, students: List[StudentProfile], 
                            save_path: Optional[str] = None) -> None:
        """Visualize student progress and statistics"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['GPA Distribution', 'Courses Completed', 'Term Distribution',
                          'Interest Distribution', 'Academic Standing', 'Domain Preferences'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}, {"type": "heatmap"}]]
        )
        
        # Prepare data
        gpas = [s.gpa for s in students]
        courses_completed = [len(s.completed_courses) for s in students]
        terms = [s.current_term for s in students]
        
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
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_recommendation_analysis(self, student: StudentProfile, 
                                   recommendations: List[str],
                                   constraints: AcademicConstraints,
                                   save_path: Optional[str] = None) -> None:
        """Visualize course recommendations for a specific student"""
        
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
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Other')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        fig.add_trace(
            go.Bar(x=list(domain_counts.keys()), y=list(domain_counts.values()),
                   name="Courses by Domain", marker_color='skyblue'),
            row=1, col=1
        )
        
        # Difficulty Distribution
        difficulty_counts = {}
        for course in recommendations:
            course_info = self.curriculum.get_course_info(course)
            difficulty = course_info.get('difficulty', 'Intermediate')
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
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
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Theory')
            interest_score = student.interests.get(domain, 0.3)
            
            courses_and_interests.append(course)
            interest_scores.append(interest_score)
        
        fig.add_trace(
            go.Bar(x=courses_and_interests, y=interest_scores,
                   name="Interest Alignment", marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Progress Analysis
        current_credits = sum(
            self.curriculum.get_course_info(course).get('credits', 3)
            for course, grade in student.completed_courses.items()
            if grade >= 2.0
        )
        
        recommended_credits = sum(
            self.curriculum.get_course_info(course).get('credits', 3)
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
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def create_student_dashboard(self, student: StudentProfile,
                               recommendations: List[str],
                               save_path: Optional[str] = None) -> None:
        """Create a comprehensive dashboard for a student"""
        
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
            'Credits Completed': sum(self.curriculum.get_course_info(c).get('credits', 3) 
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
                course_info = self.curriculum.get_course_info(course)
                domain = course_info.get('domain', 'Other')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if domain_counts:
            fig.add_trace(
                go.Pie(labels=list(domain_counts.keys()), 
                       values=list(domain_counts.values())),
                row=1, col=3
            )
        
        # 4. GPA Trend (Simulated)
        # Simulate GPA progression
        terms = list(range(1, student.current_term + 1))
        gpa_trend = [2.5 + (student.gpa - 2.5) * (t / student.current_term) for t in terms]
        
        fig.add_trace(
            go.Scatter(x=terms, y=gpa_trend, mode='lines+markers',
                       line=dict(color='blue'), name='GPA Trend'),
            row=2, col=1
        )
        
        # 5. Recommended Courses
        fig.add_trace(
            go.Bar(x=recommendations, y=[1]*len(recommendations),
                   marker_color='orange', text=recommendations, textposition='auto'),
            row=2, col=2
        )
        
        # 6. Course Difficulty Timeline
        completed_courses = [c for c, g in student.completed_courses.items() if g >= 2.0]
        difficulties = []
        for course in completed_courses:
            course_info = self.curriculum.get_course_info(course)
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
        remaining_courses = self.curriculum.get_graduation_path(
            {c for c, g in student.completed_courses.items() if g >= 2.0}
        )
        
        path_domains = {}
        for course in remaining_courses[:10]:  # Show first 10
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Other')
            path_domains[domain] = path_domains.get(domain, 0) + 1
        
        if path_domains:
            fig.add_trace(
                go.Bar(x=list(path_domains.keys()), y=list(path_domains.values()),
                       marker_color='purple'),
                row=3, col=1
            )
        
        # 8. Constraint Compliance
        constraints = AcademicConstraints(self.curriculum)
        constraint_score = constraints.get_constraint_score(recommendations, student)
        
        compliance_data = {
            'Constraint Score': constraint_score,
            'Course Load': len(recommendations) / student.max_courses_per_term,
            'Interest Alignment': np.mean([student.interests.get(
                self.curriculum.get_course_info(c).get('domain', 'Theory'), 0.3
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
            ['Recommended Courses', f"{len(recommendations)}"]
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
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_training_metrics(self, training_stats: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> None:
        """Plot RL training metrics"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Training Reward', 'Epsilon Decay', 'Episode Length', 'Loss']
        )
        
        episodes = list(range(len(training_stats.get('rewards', []))))
        
        # Training Reward
        if 'rewards' in training_stats:
            fig.add_trace(
                go.Scatter(x=episodes, y=training_stats['rewards'],
                           mode='lines', name='Reward'),
                row=1, col=1
            )
        
        # Epsilon Decay
        if 'epsilon' in training_stats:
            fig.add_trace(
                go.Scatter(x=episodes, y=training_stats['epsilon'],
                           mode='lines', name='Epsilon'),
                row=1, col=2
            )
        
        # Episode Length
        if 'episode_length' in training_stats:
            fig.add_trace(
                go.Scatter(x=episodes, y=training_stats['episode_length'],
                           mode='lines', name='Episode Length'),
                row=2, col=1
            )
        
        # Loss
        if 'loss' in training_stats:
            fig.add_trace(
                go.Scatter(x=episodes, y=training_stats['loss'],
                           mode='lines', name='Loss'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="RL Training Metrics",
            height=600,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()


def create_visualizer(curriculum: CurriculumGraph) -> CurriculumVisualizer:
    """Create a curriculum visualizer"""
    return CurriculumVisualizer(curriculum)


if __name__ == "__main__":
    from curriculum_graph import create_sample_curriculum
    from student_simulation import create_student_cohort
    from constraints import AcademicConstraints
    
    # Create test data
    curriculum = create_sample_curriculum()
    students = create_student_cohort(curriculum, 100)
    visualizer = create_visualizer(curriculum)
    
    print("Creating visualizations...")
    
    # Plot curriculum graph
    visualizer.plot_curriculum_graph(
        save_path="/Users/mohamedahmed/NU Research Task/reports/curriculum_graph.html"
    )
    
    # Plot student progress
    visualizer.plot_student_progress(
        students,
        save_path="/Users/mohamedahmed/NU Research Task/reports/student_analysis.html"
    )
    
    # Create student dashboard for first student
    test_student = students[0]
    
    # Generate sample recommendations
    constraints = AcademicConstraints(curriculum)
    passed_courses = {c for c, g in test_student.completed_courses.items() if g >= 2.0}
    eligible = curriculum.get_eligible_courses(passed_courses)
    recommendations = eligible[:4] if len(eligible) >= 4 else eligible
    
    visualizer.create_student_dashboard(
        test_student,
        recommendations,
        save_path="/Users/mohamedahmed/NU Research Task/reports/student_dashboard.html"
    )
    
    print("Visualizations saved to reports/ directory")
