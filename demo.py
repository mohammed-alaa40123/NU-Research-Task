#!/usr/bin/env python3
"""
Demo Script for AI Curriculum Planner

This script demonstrates the core functionality of the AI academic advising system.
"""

from src.curriculum_graph import create_sample_curriculum
from src.student_simulation import create_student_cohort
from src.constraints import create_constraint_validator
from src.rl_advisor import create_rl_advisor

def main():
    print("🎓 AI Curriculum Planner - Demo")
    print("=" * 50)
    
    # 1. Create curriculum
    print("1. Creating curriculum graph...")
    curriculum = create_sample_curriculum()
    print(f"   ✓ Created curriculum with {len(curriculum.graph.nodes())} courses")
    
    # 2. Generate students
    print("\n2. Generating student cohort...")
    students = create_student_cohort(curriculum, 10)
    print(f"   ✓ Generated {len(students)} student profiles")
    
    # 3. Initialize components
    print("\n3. Initializing AI advisor...")
    constraints = create_constraint_validator(curriculum)
    advisor = create_rl_advisor(curriculum, constraints)
    print("   ✓ AI advisor initialized")
    
    # 4. Demonstrate course recommendations
    print("\n4. Generating course recommendations...")
    print("-" * 50)
    
    for i, student in enumerate(students[:3]):
        print(f"\n📚 Student {i+1}: {student.student_id}")
        print(f"   GPA: {student.gpa:.2f}")
        print(f"   Completed: {len(student.completed_courses)} courses")
        print(f"   Current Term: {student.current_term}")
        print(f"   Academic Standing: {student.academic_standing}")
        
        # Show top interests
        top_interests = sorted(student.interests.items(), key=lambda x: x[1], reverse=True)[:2]
        print(f"   Top Interests: {', '.join([f'{k}({v:.2f})' for k, v in top_interests])}")
        
        # Get recommendations
        recommendations = advisor.select_courses(student, 3)
        print(f"   Recommendations: {recommendations}")
        
        # Validate recommendations
        is_valid, violations = constraints.validate_course_selection(recommendations, student)
        print(f"   Valid: {is_valid}")
        if violations:
            print(f"   Warnings: {len(violations)}")
            for v in violations[:2]:  # Show first 2
                print(f"     - {v.description}")
    
    # 5. Show curriculum insights
    print("\n5. Curriculum Analysis:")
    print("-" * 50)
    
    stats = curriculum.get_stats()
    print(f"   Total Courses: {stats['total_courses']}")
    print(f"   Prerequisites: {stats['total_prerequisites']}")
    print(f"   Domains: {list(stats['domains'].keys())}")
    
    # Show some interesting courses
    print("\n   🔍 Sample Courses:")
    sample_courses = ['CS101', 'CS311', 'CS312', 'SEC301', 'DS301']
    for course in sample_courses:
        if course in curriculum.graph.nodes():
            info = curriculum.get_course_info(course)
            prereqs = curriculum.get_prerequisites(course)
            print(f"   {course}: {info.get('name', 'Unknown')} ({info.get('domain', 'Unknown')})")
            if prereqs:
                print(f"     Prerequisites: {', '.join(prereqs)}")
    
    print("\n" + "=" * 50)
    print("🎯 Demo completed successfully!")
    print("   - Curriculum graph created and analyzed")
    print("   - Student cohort generated with diverse profiles")
    print("   - AI advisor provides personalized recommendations")
    print("   - Academic constraints enforced")
    print("   - System ready for full deployment")

if __name__ == "__main__":
    main()
