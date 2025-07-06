"""
Academic Constraints Module

This module defines and enforces academic constraints for course scheduling:
- Course load limits
- Prerequisite requirements
- Retake policies
- GPA requirements
- Graduation requirements
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
try:
    from .student_simulation import StudentProfile
    from .curriculum_graph import CurriculumGraph
except ImportError:
    from student_simulation import StudentProfile
    from curriculum_graph import CurriculumGraph


@dataclass
class ConstraintViolation:
    """Represents a constraint violation"""
    constraint_type: str
    description: str
    severity: str  # "Error", "Warning"
    affected_courses: List[str]


class AcademicConstraints:
    """Manages and enforces academic constraints"""
    
    def __init__(self, curriculum: CurriculumGraph):
        self.curriculum = curriculum
        self.min_gpa_for_graduation = 2.0
        self.min_gpa_for_advanced_courses = 2.5
        self.max_retakes_per_course = 2
        self.max_total_retakes = 5
        
    def check_course_load_constraint(self, selected_courses: List[str], 
                                   student: StudentProfile) -> List[ConstraintViolation]:
        """Check if course load is within limits"""
        violations = []
        
        if len(selected_courses) > student.max_courses_per_term:
            violations.append(ConstraintViolation(
                constraint_type="CourseLoad",
                description=f"Selected {len(selected_courses)} courses exceeds maximum of {student.max_courses_per_term}",
                severity="Error",
                affected_courses=selected_courses
            ))
        
        # Check credit load
        total_credits = sum(
            self.curriculum.get_course_info(course).get('credits', 3)
            for course in selected_courses
        )
        
        max_credits = student.max_courses_per_term * 3 + 3  # Allow some flexibility
        if total_credits > max_credits:
            violations.append(ConstraintViolation(
                constraint_type="CreditLoad",
                description=f"Total credits ({total_credits}) exceeds recommended maximum ({max_credits})",
                severity="Warning",
                affected_courses=selected_courses
            ))
        
        return violations
    
    def check_prerequisite_constraints(self, selected_courses: List[str], 
                                     student: StudentProfile) -> List[ConstraintViolation]:
        """Check if all prerequisites are satisfied"""
        violations = []
        passed_courses = {
            course for course, grade in student.completed_courses.items() 
            if grade >= 2.0
        }
        
        for course in selected_courses:
            if not self.curriculum.can_take_course(course, passed_courses):
                missing_prereqs = set(self.curriculum.get_prerequisites(course)) - passed_courses
                violations.append(ConstraintViolation(
                    constraint_type="Prerequisites",
                    description=f"Course {course} requires prerequisites: {missing_prereqs}",
                    severity="Error",
                    affected_courses=[course]
                ))
        
        return violations
    
    def check_retake_constraints(self, selected_courses: List[str], 
                                student: StudentProfile) -> List[ConstraintViolation]:
        """Check retake policy compliance"""
        violations = []
        
        # Count current retakes
        retake_counts = {}
        total_retakes = 0
        
        for course in student.completed_courses:
            if student.completed_courses[course] < 2.0:
                retake_counts[course] = retake_counts.get(course, 0) + 1
                total_retakes += 1
        
        # Check individual course retake limits
        for course in selected_courses:
            if course in student.failed_courses:
                current_retakes = retake_counts.get(course, 0)
                if current_retakes >= self.max_retakes_per_course:
                    violations.append(ConstraintViolation(
                        constraint_type="CourseRetake",
                        description=f"Course {course} has been retaken {current_retakes} times (max: {self.max_retakes_per_course})",
                        severity="Error",
                        affected_courses=[course]
                    ))
        
        # Check total retake limit
        new_retakes = len([c for c in selected_courses if c in student.failed_courses])
        if total_retakes + new_retakes > self.max_total_retakes:
            violations.append(ConstraintViolation(
                constraint_type="TotalRetakes",
                description=f"Total retakes would exceed limit ({total_retakes + new_retakes} > {self.max_total_retakes})",
                severity="Warning",
                affected_courses=[c for c in selected_courses if c in student.failed_courses]
            ))
        
        return violations
    
    def check_gpa_constraints(self, selected_courses: List[str], 
                            student: StudentProfile) -> List[ConstraintViolation]:
        """Check GPA-related constraints"""
        violations = []
        
        # Check if student can take advanced courses
        advanced_courses = [
            course for course in selected_courses
            if self.curriculum.get_course_info(course).get('difficulty') == 'Advanced'
        ]
        
        if advanced_courses and student.gpa < self.min_gpa_for_advanced_courses:
            violations.append(ConstraintViolation(
                constraint_type="GPA_Advanced",
                description=f"GPA {student.gpa} below minimum {self.min_gpa_for_advanced_courses} for advanced courses",
                severity="Warning",
                affected_courses=advanced_courses
            ))
        
        # Check academic standing constraints
        if student.academic_standing == "Probation":
            if len(selected_courses) > 3:
                violations.append(ConstraintViolation(
                    constraint_type="ProbationLoad",
                    description="Students on probation limited to 3 courses per term",
                    severity="Error",
                    affected_courses=selected_courses
                ))
        
        return violations
    
    def check_sequence_constraints(self, selected_courses: List[str], 
                                 student: StudentProfile) -> List[ConstraintViolation]:
        """Check course sequence and timing constraints"""
        violations = []
        
        # Check for courses that should be taken in sequence
        sequences = [
            ['CS310', 'CS311', 'CS312'],  # AI sequence
            ['CS301', 'CS402'],           # Theory sequence
            ['SEC301', 'SEC302', 'SEC303'] # Security sequence
        ]
        
        for sequence in sequences:
            selected_in_sequence = [c for c in selected_courses if c in sequence]
            if len(selected_in_sequence) > 1:
                # Check if taking courses out of order
                completed_in_sequence = [
                    c for c in sequence 
                    if c in student.completed_courses and student.completed_courses[c] >= 2.0
                ]
                
                for course in selected_in_sequence:
                    course_index = sequence.index(course)
                    if course_index > 0:
                        prev_course = sequence[course_index - 1]
                        if prev_course not in completed_in_sequence:
                            violations.append(ConstraintViolation(
                                constraint_type="Sequence",
                                description=f"Recommended to complete {prev_course} before {course}",
                                severity="Warning",
                                affected_courses=[course]
                            ))
        
        return violations
    
    def check_graduation_progress(self, selected_courses: List[str], 
                                student: StudentProfile) -> List[ConstraintViolation]:
        """Check if student is making adequate progress toward graduation"""
        violations = []
        
        # Calculate current and projected credits
        current_credits = sum(
            self.curriculum.get_course_info(course).get('credits', 3)
            for course, grade in student.completed_courses.items()
            if grade >= 2.0
        )
        
        selected_credits = sum(
            self.curriculum.get_course_info(course).get('credits', 3)
            for course in selected_courses
        )
        
        projected_credits = current_credits + selected_credits
        
        # Check if on track for graduation
        expected_credits = student.current_term * 12  # Assume 12 credits per term
        if projected_credits < expected_credits * 0.8:
            violations.append(ConstraintViolation(
                constraint_type="GraduationProgress",
                description=f"Behind in credit accumulation: {projected_credits} vs expected {expected_credits}",
                severity="Warning",
                affected_courses=[]
            ))
        
        # Check domain coverage for graduation
        domain_coverage = self._check_domain_coverage(student, selected_courses)
        if not domain_coverage['sufficient']:
            violations.append(ConstraintViolation(
                constraint_type="DomainCoverage",
                description=f"Insufficient coverage in domains: {domain_coverage['missing']}",
                severity="Warning",
                affected_courses=[]
            ))
        
        return violations
    
    def _check_domain_coverage(self, student: StudentProfile, 
                              selected_courses: List[str]) -> Dict:
        """Check if student has sufficient domain coverage"""
        # Requirements: at least 2 courses in primary area, 1 in others
        domain_requirements = {
            'AI': 2, 'Data Science': 2, 'Security': 1, 
            'Software Engineering': 3, 'Systems': 2, 'Theory': 1
        }
        
        # Count completed and selected courses by domain
        all_courses = list(student.completed_courses.keys()) + selected_courses
        passed_courses = [
            course for course, grade in student.completed_courses.items()
            if grade >= 2.0
        ] + selected_courses
        
        domain_counts = {}
        for course in passed_courses:
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Other')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Check requirements
        missing_domains = []
        for domain, required in domain_requirements.items():
            if domain_counts.get(domain, 0) < required:
                missing_domains.append(domain)
        
        return {
            'sufficient': len(missing_domains) == 0,
            'missing': missing_domains,
            'current_coverage': domain_counts
        }
    
    def validate_course_selection(self, selected_courses: List[str], 
                                student: StudentProfile) -> Tuple[bool, List[ConstraintViolation]]:
        """Validate a course selection against all constraints"""
        all_violations = []
        
        # Run all constraint checks
        all_violations.extend(self.check_course_load_constraint(selected_courses, student))
        all_violations.extend(self.check_prerequisite_constraints(selected_courses, student))
        all_violations.extend(self.check_retake_constraints(selected_courses, student))
        all_violations.extend(self.check_gpa_constraints(selected_courses, student))
        all_violations.extend(self.check_sequence_constraints(selected_courses, student))
        all_violations.extend(self.check_graduation_progress(selected_courses, student))
        
        # Check if any critical errors
        has_errors = any(v.severity == "Error" for v in all_violations)
        
        return not has_errors, all_violations
    
    def get_constraint_score(self, selected_courses: List[str], 
                           student: StudentProfile) -> float:
        """Get a numerical score for constraint compliance (0-1, higher is better)"""
        is_valid, violations = self.validate_course_selection(selected_courses, student)
        
        if not is_valid:
            return 0.0
        
        # Penalize warnings
        warning_penalty = len([v for v in violations if v.severity == "Warning"]) * 0.1
        
        return max(0.0, 1.0 - warning_penalty)
    
    def suggest_constraint_fixes(self, selected_courses: List[str], 
                               student: StudentProfile) -> List[str]:
        """Suggest course modifications to fix constraint violations"""
        is_valid, violations = self.validate_course_selection(selected_courses, student)
        
        if is_valid:
            return []
        
        suggestions = []
        
        for violation in violations:
            if violation.constraint_type == "Prerequisites":
                suggestions.append(f"Add prerequisite courses or remove {violation.affected_courses}")
            elif violation.constraint_type == "CourseLoad":
                suggestions.append(f"Reduce course load to {student.max_courses_per_term} courses")
            elif violation.constraint_type == "GPA_Advanced":
                suggestions.append("Focus on improving GPA before taking advanced courses")
            elif violation.constraint_type == "ProbationLoad":
                suggestions.append("Limit to 3 courses while on probation")
        
        return suggestions


def create_constraint_validator(curriculum: CurriculumGraph) -> AcademicConstraints:
    """Create a constraint validator for the curriculum"""
    return AcademicConstraints(curriculum)


if __name__ == "__main__":
    from curriculum_graph import create_sample_curriculum
    from student_simulation import create_student_cohort
    
    # Test constraints
    curriculum = create_sample_curriculum()
    students = create_student_cohort(curriculum, 10)
    constraints = create_constraint_validator(curriculum)
    
    # Test with a sample student
    student = students[0]
    print(f"Testing constraints for {student.student_id}")
    print(f"Current GPA: {student.gpa}")
    print(f"Completed courses: {len(student.completed_courses)}")
    
    # Get eligible courses
    passed_courses = {
        course for course, grade in student.completed_courses.items() 
        if grade >= 2.0
    }
    eligible = curriculum.get_eligible_courses(passed_courses)
    
    # Test a course selection
    test_selection = eligible[:4] if len(eligible) >= 4 else eligible
    print(f"\nTesting course selection: {test_selection}")
    
    is_valid, violations = constraints.validate_course_selection(test_selection, student)
    print(f"Valid: {is_valid}")
    
    if violations:
        print("Violations:")
        for v in violations:
            print(f"  - {v.constraint_type}: {v.description} ({v.severity})")
    
    score = constraints.get_constraint_score(test_selection, student)
    print(f"Constraint score: {score}")
