"""
Student Simulation Module

This module generates realistic student profiles with:
- Completed courses and grades
- Academic interests and preferences
- GPA and academic standing
- Graduation timeline and constraints
"""

import random
import json
import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
try:
    from .curriculum_graph import CurriculumGraph
except ImportError:
    from curriculum_graph import CurriculumGraph


@dataclass
class StudentProfile:
    """Represents a student's academic profile"""
    student_id: str
    name: str
    completed_courses: Dict[str, float]  # course_code: grade
    failed_courses: Set[str]
    current_term: int
    gpa: float
    interests: Dict[str, float]  # domain: interest_level (0-1)
    max_courses_per_term: int
    target_graduation_term: int
    academic_standing: str  # "Good", "Probation", "Warning"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['failed_courses'] = list(self.failed_courses)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudentProfile':
        """Create StudentProfile from dictionary"""
        data['failed_courses'] = set(data['failed_courses'])
        return cls(**data)


class StudentSimulator:
    """Generates realistic student profiles for curriculum planning"""
    
    def __init__(self, curriculum: CurriculumGraph):
        self.curriculum = curriculum
        self.domains = ['AI', 'Security', 'Data Science', 'Software Engineering', 'Systems', 'Theory']
        
        # Grade distribution parameters
        self.grade_means = {
            'Beginner': 3.2,
            'Intermediate': 2.9,
            'Advanced': 2.7
        }
        self.grade_stds = {
            'Beginner': 0.6,
            'Intermediate': 0.8,
            'Advanced': 0.9
        }
    
    def generate_student_interests(self) -> Dict[str, float]:
        """Generate realistic interest distribution across domains"""
        # Each student has 1-3 primary interests
        num_primary = random.randint(1, 3)
        primary_domains = random.sample(self.domains, num_primary)
        
        interests = {}
        for domain in self.domains:
            if domain in primary_domains:
                interests[domain] = random.uniform(0.7, 1.0)
            else:
                interests[domain] = random.uniform(0.1, 0.5)
        
        return interests
    
    def simulate_grade(self, course_code: str, student_interests: Dict[str, float], 
                      student_ability: float) -> float:
        """Simulate a grade for a course based on difficulty and student factors"""
        course_info = self.curriculum.get_course_info(course_code)
        difficulty = course_info.get('difficulty', 'Intermediate')
        domain = course_info.get('domain', 'Theory')
        
        # Base grade from difficulty
        base_mean = self.grade_means[difficulty]
        base_std = self.grade_stds[difficulty]
        
        # Adjust based on student interest in domain
        interest_boost = (student_interests.get(domain, 0.3) - 0.5) * 0.8
        
        # Adjust based on student ability
        ability_boost = (student_ability - 0.5) * 1.0
        
        # Generate grade
        adjusted_mean = base_mean + interest_boost + ability_boost
        grade = np.random.normal(adjusted_mean, base_std)
        
        # Clamp to valid GPA range
        return max(0.0, min(4.0, grade))
    
    def determine_failed_courses(self, completed_courses: Dict[str, float]) -> Set[str]:
        """Determine which courses the student failed (grade < 2.0)"""
        return {course for course, grade in completed_courses.items() if grade < 2.0}
    
    def calculate_gpa(self, completed_courses: Dict[str, float]) -> float:
        """Calculate GPA from completed courses"""
        if not completed_courses:
            return 0.0
        
        total_points = 0.0
        total_credits = 0
        
        for course_code, grade in completed_courses.items():
            if grade >= 2.0:  # Only count passed courses
                credits = self.curriculum.get_course_info(course_code).get('credits', 3)
                total_points += grade * credits
                total_credits += credits
        
        return total_points / total_credits if total_credits > 0 else 0.0
    
    def generate_academic_progression(self, student_interests: Dict[str, float], 
                                    student_ability: float, max_terms: int = 8) -> Tuple[Dict[str, float], int]:
        """Generate realistic academic progression for a student"""
        completed_courses = {}
        current_term = 1
        
        # Start with foundational courses
        foundational = ['CS101', 'MATH101']
        
        for term in range(1, max_terms + 1):
            current_term = term
            
            # Determine eligible courses
            completed_set = set(course for course, grade in completed_courses.items() if grade >= 2.0)
            eligible = self.curriculum.get_eligible_courses(completed_set)
            
            # For early terms, prioritize foundational courses
            if term <= 2:
                available = [c for c in foundational if c in eligible]
                if not available:
                    available = eligible
            else:
                available = eligible
            
            # Remove already completed courses (including failed ones)
            available = [c for c in available if c not in completed_courses]
            
            if not available:
                break
            
            # Select courses based on interests and progression
            courses_this_term = self.select_courses_for_term(
                available, student_interests, term, 
                max_courses=random.randint(3, 5)
            )
            
            # Simulate grades for selected courses
            for course in courses_this_term:
                grade = self.simulate_grade(course, student_interests, student_ability)
                completed_courses[course] = grade
            
            # Random chance to stop early (transfer, gap year, etc.)
            if term >= 4 and random.random() < 0.1:
                break
        
        return completed_courses, current_term
    
    def select_courses_for_term(self, available_courses: List[str], 
                               interests: Dict[str, float], term: int, 
                               max_courses: int = 4) -> List[str]:
        """Select courses for a term based on interests and constraints"""
        if not available_courses:
            return []
        
        # Score courses based on interests and term appropriateness
        scored_courses = []
        for course in available_courses:
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Theory')
            difficulty = course_info.get('difficulty', 'Intermediate')
            
            # Base score from interest
            score = interests.get(domain, 0.3)
            
            # Adjust for term appropriateness
            if term <= 2 and difficulty == 'Beginner':
                score += 0.3
            elif term <= 4 and difficulty == 'Intermediate':
                score += 0.2
            elif term > 4 and difficulty == 'Advanced':
                score += 0.2
            
            # Slight randomness
            score += random.uniform(-0.1, 0.1)
            
            scored_courses.append((course, score))
        
        # Sort by score and select top courses
        scored_courses.sort(key=lambda x: x[1], reverse=True)
        selected = [course for course, _ in scored_courses[:max_courses]]
        
        return selected
    
    def determine_academic_standing(self, gpa: float, failed_courses: Set[str]) -> str:
        """Determine academic standing based on GPA and failed courses"""
        if gpa >= 3.0 and len(failed_courses) == 0:
            return "Good"
        elif gpa >= 2.5 and len(failed_courses) <= 1:
            return "Good"
        elif gpa >= 2.0:
            return "Warning"
        else:
            return "Probation"
    
    def generate_student(self, student_id: int) -> StudentProfile:
        """Generate a complete student profile"""
        # Generate basic attributes
        name = f"Student_{student_id:03d}"
        interests = self.generate_student_interests()
        student_ability = np.random.beta(2, 2)  # Ability distribution
        
        # Generate academic progression
        completed_courses, current_term = self.generate_academic_progression(
            interests, student_ability
        )
        
        # Calculate derived attributes
        failed_courses = self.determine_failed_courses(completed_courses)
        gpa = self.calculate_gpa(completed_courses)
        academic_standing = self.determine_academic_standing(gpa, failed_courses)
        
        # Determine graduation timeline
        completed_credits = sum(
            self.curriculum.get_course_info(course).get('credits', 3)
            for course, grade in completed_courses.items()
            if grade >= 2.0
        )
        
        remaining_credits = max(0, 120 - completed_credits)
        avg_credits_per_term = 12
        terms_to_graduation = max(1, (remaining_credits + avg_credits_per_term - 1) // avg_credits_per_term)
        target_graduation_term = current_term + terms_to_graduation
        
        return StudentProfile(
            student_id=f"STU_{student_id:03d}",
            name=name,
            completed_courses=completed_courses,
            failed_courses=failed_courses,
            current_term=current_term,
            gpa=round(gpa, 2),
            interests=interests,
            max_courses_per_term=random.randint(3, 5),
            target_graduation_term=target_graduation_term,
            academic_standing=academic_standing
        )
    
    def generate_cohort(self, num_students: int = 100) -> List[StudentProfile]:
        """Generate a cohort of students with diverse profiles"""
        students = []
        
        for i in range(num_students):
            student = self.generate_student(i + 1)
            students.append(student)
        
        return students
    
    def save_students_to_file(self, students: List[StudentProfile], filepath: str) -> None:
        """Save student profiles to JSON file"""
        data = [student.to_dict() for student in students]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_students_from_file(self, filepath: str) -> List[StudentProfile]:
        """Load student profiles from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [StudentProfile.from_dict(student_data) for student_data in data]
    
    def get_cohort_statistics(self, students: List[StudentProfile]) -> Dict:
        """Calculate statistics for the student cohort"""
        if not students:
            return {}
        
        gpas = [s.gpa for s in students]
        terms = [s.current_term for s in students]
        completed_counts = [len(s.completed_courses) for s in students]
        
        # Interest distribution
        interest_stats = {}
        for domain in self.domains:
            domain_interests = [s.interests.get(domain, 0) for s in students]
            interest_stats[domain] = {
                'mean': np.mean(domain_interests),
                'high_interest_students': sum(1 for x in domain_interests if x > 0.7)
            }
        
        return {
            'total_students': len(students),
            'gpa_stats': {
                'mean': np.mean(gpas),
                'std': np.std(gpas),
                'min': np.min(gpas),
                'max': np.max(gpas)
            },
            'term_stats': {
                'mean': np.mean(terms),
                'min': np.min(terms),
                'max': np.max(terms)
            },
            'courses_completed': {
                'mean': np.mean(completed_counts),
                'min': np.min(completed_counts),
                'max': np.max(completed_counts)
            },
            'academic_standing': {
                standing: sum(1 for s in students if s.academic_standing == standing)
                for standing in ['Good', 'Warning', 'Probation']
            },
            'interest_distribution': interest_stats
        }


def create_student_cohort(curriculum: CurriculumGraph, num_students: int = 100) -> List[StudentProfile]:
    """Create a diverse cohort of students"""
    simulator = StudentSimulator(curriculum)
    return simulator.generate_cohort(num_students)


if __name__ == "__main__":
    # Create curriculum and generate students
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from curriculum_graph import create_sample_curriculum
    
    curriculum = create_sample_curriculum()
    simulator = StudentSimulator(curriculum)
    
    print("Generating student cohort...")
    students = simulator.generate_cohort(100)
    
    print("\nCohort Statistics:")
    stats = simulator.get_cohort_statistics(students)
    
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save students
    simulator.save_students_to_file(students, '/Users/mohamedahmed/NU Research Task/data/students_data.json')
    print(f"\nGenerated {len(students)} students and saved to data/students_data.json")
    
    # Show sample student
    print(f"\nSample Student Profile:")
    sample = students[0]
    print(f"ID: {sample.student_id}")
    print(f"GPA: {sample.gpa}")
    print(f"Term: {sample.current_term}")
    print(f"Completed Courses: {len(sample.completed_courses)}")
    print(f"Top Interest: {max(sample.interests.items(), key=lambda x: x[1])}")
    print(f"Academic Standing: {sample.academic_standing}")
