"""
Main Execution Script for AI Curriculum Planner

This script coordinates all components of the AI academic advising system:
- Generates curriculum graph and student data
- Trains the RL advisor
- Generates recommendations
- Creates visualizations
"""

import argparse
import json
import random
import time
from typing import List, Dict, Any
import os

# Import our modules
from src.curriculum_graph import CurriculumGraph, create_sample_curriculum
from src.student_simulation import StudentSimulator, StudentProfile, create_student_cohort
from src.constraints import AcademicConstraints, create_constraint_validator
from src.rl_advisor import DQNAdvisor, create_rl_advisor
from src.visualization import CurriculumVisualizer, create_visualizer


class CurriculumPlannerSystem:
    """Main system for AI curriculum planning"""
    
    def __init__(self):
        self.curriculum = None
        self.students = []
        self.constraints = None
        self.advisor = None
        self.visualizer = None
        
        # File paths
        self.data_dir = "data"
        self.reports_dir = "reports"
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_data(self, num_students: int = 100) -> None:
        """Generate curriculum graph and student data"""
        print("=" * 60)
        print("GENERATING CURRICULUM AND STUDENT DATA")
        print("=" * 60)
        
        # Create curriculum
        print("Creating curriculum graph...")
        self.curriculum = create_sample_curriculum()
        
        # Save curriculum
        curriculum_path = os.path.join(self.data_dir, "curriculum_data.json")
        self.curriculum.save_to_file(curriculum_path)
        
        # Print curriculum statistics
        stats = self.curriculum.get_stats()
        print(f"Curriculum created with:")
        print(f"  - {stats['total_courses']} courses")
        print(f"  - {stats['total_prerequisites']} prerequisite relationships")
        print(f"  - Average prerequisites per course: {stats['avg_prerequisites']:.2f}")
        print(f"  - Domain distribution: {stats['domains']}")
        
        # Generate students
        print(f"\nGenerating {num_students} student profiles...")
        simulator = StudentSimulator(self.curriculum)
        self.students = simulator.generate_cohort(num_students)
        
        # Save students
        students_path = os.path.join(self.data_dir, "students_data.json")
        simulator.save_students_to_file(self.students, students_path)
        
        # Print student statistics
        student_stats = simulator.get_cohort_statistics(self.students)
        print(f"Student cohort created:")
        print(f"  - Total students: {student_stats['total_students']}")
        print(f"  - Average GPA: {student_stats['gpa_stats']['mean']:.2f}")
        print(f"  - GPA range: {student_stats['gpa_stats']['min']:.2f} - {student_stats['gpa_stats']['max']:.2f}")
        print(f"  - Average courses completed: {student_stats['courses_completed']['mean']:.1f}")
        print(f"  - Academic standing: {student_stats['academic_standing']}")
        
        print(f"\nData saved to:")
        print(f"  - {curriculum_path}")
        print(f"  - {students_path}")
    
    def load_data(self) -> None:
        """Load existing curriculum and student data"""
        print("Loading existing data...")
        
        # Load curriculum
        curriculum_path = os.path.join(self.data_dir, "curriculum_data.json")
        if os.path.exists(curriculum_path):
            self.curriculum = CurriculumGraph()
            self.curriculum.load_from_file(curriculum_path)
            print(f"Loaded curriculum with {len(self.curriculum.graph.nodes())} courses")
        else:
            print("No curriculum data found. Please run --generate-data first.")
            return
        
        # Load students
        students_path = os.path.join(self.data_dir, "students_data.json")
        if os.path.exists(students_path):
            simulator = StudentSimulator(self.curriculum)
            self.students = simulator.load_students_from_file(students_path)
            print(f"Loaded {len(self.students)} student profiles")
        else:
            print("No student data found. Please run --generate-data first.")
            return
        
        # Initialize other components
        self.constraints = create_constraint_validator(self.curriculum)
        self.visualizer = create_visualizer(self.curriculum)
    
    def train_rl_model(self, episodes: int = 100) -> None:
        """Train the reinforcement learning advisor"""
        print("=" * 60)
        print("TRAINING REINFORCEMENT LEARNING MODEL")
        print("=" * 60)
        
        if not self.curriculum or not self.students:
            print("Please load or generate data first.")
            return
        
        # Initialize RL advisor
        self.constraints = create_constraint_validator(self.curriculum)
        self.advisor = create_rl_advisor(self.curriculum, self.constraints)
        
        print(f"Training DQN advisor for {episodes} episodes...")
        print("This may take a few minutes...\n")
        
        training_rewards = []
        training_times = []
        
        start_time = time.time()
        
        for episode in range(episodes):
            episode_start = time.time()
            
            # Select random student for training
            student = random.choice(self.students)
            
            # Train episode
            episode_stats = self.advisor.train_episode(student)
            
            # Record metrics
            training_rewards.append(episode_stats['total_reward'])
            training_times.append(time.time() - episode_start)
            
            # Update target network periodically
            if episode % 10 == 0:
                self.advisor.update_target_network()
            
            # Print progress
            if (episode + 1) % 20 == 0:
                avg_reward = sum(training_rewards[-20:]) / 20
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Avg Reward (last 20): {avg_reward:.2f}, "
                      f"Epsilon: {self.advisor.epsilon:.3f}")
        
        total_time = time.time() - start_time
        
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Average reward: {sum(training_rewards) / len(training_rewards):.2f}")
        print(f"Final epsilon: {self.advisor.epsilon:.3f}")
        
        # Save trained model
        model_path = os.path.join(self.data_dir, "trained_advisor.pth")
        self.advisor.save_model(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save training statistics
        training_stats = {
            'rewards': training_rewards,
            'episode_times': training_times,
            'total_episodes': episodes,
            'total_time': total_time
        }
        
        stats_path = os.path.join(self.data_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
    
    def generate_recommendations(self, num_students: int = 10) -> None:
        """Generate course recommendations for students"""
        print("=" * 60)
        print("GENERATING COURSE RECOMMENDATIONS")
        print("=" * 60)
        
        if not self.curriculum or not self.students:
            print("Please load or generate data first.")
            return
        
        # Initialize components if needed
        if not self.constraints:
            self.constraints = create_constraint_validator(self.curriculum)
        
        if not self.advisor:
            self.advisor = create_rl_advisor(self.curriculum, self.constraints)
            # Try to load trained model
            model_path = os.path.join(self.data_dir, "trained_advisor.pth")
            if os.path.exists(model_path):
                self.advisor.load_model(model_path)
                print("Loaded trained RL model")
            else:
                print("Using untrained model (consider running --train-rl first)")
        
        # Select students for recommendations
        selected_students = self.students[:num_students]
        
        recommendations_data = []
        
        print(f"Generating recommendations for {len(selected_students)} students...\n")
        
        for i, student in enumerate(selected_students):
            print(f"Student {i+1}: {student.student_id}")
            print(f"  Current GPA: {student.gpa:.2f}")
            print(f"  Completed courses: {len(student.completed_courses)}")
            print(f"  Academic standing: {student.academic_standing}")
            
            # Generate recommendations
            recommendations = self.advisor.select_courses(student, num_courses=3)
            
            print(f"  Recommended courses: {recommendations}")
            
            # Validate recommendations
            is_valid, violations = self.constraints.validate_course_selection(
                recommendations, student
            )
            
            print(f"  Valid: {is_valid}")
            if violations:
                print(f"  Warnings/Errors: {len(violations)}")
                for v in violations[:2]:  # Show first 2 violations
                    print(f"    - {v.description}")
            
            # Calculate interest alignment
            if recommendations:
                alignment_scores = []
                for course in recommendations:
                    course_info = self.curriculum.get_course_info(course)
                    domain = course_info.get('domain', 'Theory')
                    alignment = student.interests.get(domain, 0.3)
                    alignment_scores.append(alignment)
                
                avg_alignment = sum(alignment_scores) / len(alignment_scores)
                print(f"  Interest alignment: {avg_alignment:.2f}")
            
            # Store recommendation data
            recommendation_record = {
                'student_id': student.student_id,
                'student_gpa': student.gpa,
                'student_term': student.current_term,
                'completed_courses': len(student.completed_courses),
                'academic_standing': student.academic_standing,
                'recommendations': recommendations,
                'is_valid': is_valid,
                'num_violations': len(violations),
                'interest_alignment': avg_alignment if recommendations else 0.0,
                'constraint_score': self.constraints.get_constraint_score(
                    recommendations, student
                )
            }
            
            recommendations_data.append(recommendation_record)
            print()
        
        # Save recommendations
        recommendations_path = os.path.join(self.data_dir, "recommendations.json")
        with open(recommendations_path, 'w') as f:
            json.dump(recommendations_data, f, indent=2)
        
        print(f"Recommendations saved to: {recommendations_path}")
        
        # Print summary statistics
        valid_recommendations = [r for r in recommendations_data if r['is_valid']]
        avg_alignment = sum(r['interest_alignment'] for r in recommendations_data) / len(recommendations_data)
        avg_constraint_score = sum(r['constraint_score'] for r in recommendations_data) / len(recommendations_data)
        
        print(f"\nRecommendation Summary:")
        print(f"  - Valid recommendations: {len(valid_recommendations)}/{len(recommendations_data)}")
        print(f"  - Average interest alignment: {avg_alignment:.2f}")
        print(f"  - Average constraint score: {avg_constraint_score:.2f}")
    
    def create_visualizations(self) -> None:
        """Create visualizations and reports"""
        print("=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        if not self.curriculum or not self.students:
            print("Please load or generate data first.")
            return
        
        if not self.visualizer:
            self.visualizer = create_visualizer(self.curriculum)
        
        print("Creating curriculum graph visualization...")
        try:
            self.visualizer.plot_curriculum_graph(
                save_path=os.path.join(self.reports_dir, "curriculum_graph.html"),
                interactive=True
            )
            print(f"✓ Curriculum graph saved to {self.reports_dir}/curriculum_graph.html")
        except Exception as e:
            print(f"✗ Failed to create curriculum graph: {e}")
        
        print("Creating student cohort analysis...")
        try:
            self.visualizer.plot_student_progress(
                self.students,
                save_path=os.path.join(self.reports_dir, "student_analysis.html")
            )
            print(f"✓ Student analysis saved to {self.reports_dir}/student_analysis.html")
        except Exception as e:
            print(f"✗ Failed to create student analysis: {e}")
        
        # Create individual student dashboards for top 3 students
        print("Creating individual student dashboards...")
        
        if not self.constraints:
            self.constraints = create_constraint_validator(self.curriculum)
        
        for i, student in enumerate(self.students[:3]):
            try:
                # Generate sample recommendations
                passed_courses = {
                    c for c, g in student.completed_courses.items() if g >= 2.0
                }
                eligible = self.curriculum.get_eligible_courses(passed_courses)
                recommendations = eligible[:4] if len(eligible) >= 4 else eligible
                
                dashboard_path = os.path.join(
                    self.reports_dir, f"student_dashboard_{student.student_id}.html"
                )
                
                self.visualizer.create_student_dashboard(
                    student, recommendations, save_path=dashboard_path
                )
                
                print(f"✓ Dashboard for {student.student_id} saved to {dashboard_path}")
            except Exception as e:
                print(f"✗ Failed to create dashboard for {student.student_id}: {e}")
        
        print(f"\nAll visualizations saved to {self.reports_dir}/ directory")
    
    def run_complete_pipeline(self, num_students: int = 100, 
                            training_episodes: int = 50) -> None:
        """Run the complete AI curriculum planning pipeline"""
        print("=" * 60)
        print("AI CURRICULUM PLANNER - COMPLETE PIPELINE")
        print("=" * 60)
        
        # Generate data
        self.generate_data(num_students)
        
        # Train RL model
        self.train_rl_model(training_episodes)
        
        # Generate recommendations
        self.generate_recommendations(min(10, num_students))
        
        # Create visualizations
        self.create_visualizations()
        
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print(f"  Data: {self.data_dir}/")
        print(f"  Reports: {self.reports_dir}/")
        print(f"  Visualizations: {self.reports_dir}/*.html")
    
    def print_sample_results(self) -> None:
        """Print sample results for demonstration"""
        if not self.students:
            print("No student data available.")
            return
        
        print("=" * 60)
        print("SAMPLE RESULTS")
        print("=" * 60)
        
        # Show 3 sample students
        for i, student in enumerate(self.students[:3]):
            print(f"\nStudent {i+1}: {student.student_id}")
            print(f"  Name: {student.name}")
            print(f"  GPA: {student.gpa:.2f}")
            print(f"  Current Term: {student.current_term}")
            print(f"  Completed Courses: {len(student.completed_courses)}")
            print(f"  Academic Standing: {student.academic_standing}")
            
            # Show top interests
            top_interests = sorted(student.interests.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top Interests: {', '.join([f'{k}({v:.2f})' for k, v in top_interests])}")
            
            # Show sample completed courses
            passed_courses = [c for c, g in student.completed_courses.items() if g >= 2.0]
            print(f"  Sample Completed: {', '.join(passed_courses[:5])}")
            
            if len(passed_courses) > 5:
                print(f"    ... and {len(passed_courses) - 5} more")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="AI Curriculum Planner: Adaptive Academic Advising System"
    )
    
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate curriculum graph and student data')
    parser.add_argument('--train-rl', action='store_true',
                       help='Train the reinforcement learning advisor')
    parser.add_argument('--recommend', action='store_true',
                       help='Generate course recommendations')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations and reports')
    parser.add_argument('--complete', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--sample-results', action='store_true',
                       help='Show sample results')
    
    parser.add_argument('--num-students', type=int, default=100,
                       help='Number of students to generate (default: 100)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes (default: 50)')
    
    args = parser.parse_args()
    
    # Create system
    system = CurriculumPlannerSystem()
    
    # Execute based on arguments
    if args.complete:
        system.run_complete_pipeline(args.num_students, args.episodes)
    else:
        if args.generate_data:
            system.generate_data(args.num_students)
        
        if args.train_rl or args.recommend or args.visualize or args.sample_results:
            system.load_data()
        
        if args.train_rl:
            system.train_rl_model(args.episodes)
        
        if args.recommend:
            system.generate_recommendations()
        
        if args.visualize:
            system.create_visualizations()
        
        if args.sample_results:
            system.print_sample_results()
    
    # If no arguments provided, run complete pipeline with default settings
    if not any(vars(args).values()):
        print("No arguments provided. Running complete pipeline with default settings.")
        print("Use --help to see available options.\n")
        system.run_complete_pipeline(num_students=100, training_episodes=50)


if __name__ == "__main__":
    main()
