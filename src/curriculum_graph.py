"""
Curriculum Graph Module

This module implements a graph-based curriculum model where:
- Nodes represent courses with attributes (code, name, credits, difficulty, domain)
- Edges represent prerequisite relationships
- Graph supports constraint checking and path analysis
"""

import networkx as nx
import json
import random
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd


class CurriculumGraph:
    """Graph-based curriculum model for academic advising"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.domains = ['AI', 'Security', 'Data Science', 'Software Engineering', 'Systems', 'Theory']
        self.difficulties = ['Beginner', 'Intermediate', 'Advanced']
        
    def build_curriculum(self) -> None:
        """Build a comprehensive computer science curriculum graph"""
        
        # Define core courses with their attributes
        courses = [
            # Foundational courses
            {'code': 'CS101', 'name': 'Intro to Programming', 'credits': 3, 'difficulty': 'Beginner', 'domain': 'Software Engineering'},
            {'code': 'CS102', 'name': 'Data Structures', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Software Engineering'},
            {'code': 'CS201', 'name': 'Algorithms', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Theory'},
            {'code': 'CS210', 'name': 'Computer Architecture', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Systems'},
            {'code': 'CS220', 'name': 'Operating Systems', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Systems'},
            {'code': 'CS230', 'name': 'Database Systems', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Data Science'},
            {'code': 'CS240', 'name': 'Computer Networks', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Systems'},
            {'code': 'CS250', 'name': 'Software Engineering', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Software Engineering'},
            
            # Mathematics and Theory
            {'code': 'MATH101', 'name': 'Discrete Mathematics', 'credits': 3, 'difficulty': 'Beginner', 'domain': 'Theory'},
            {'code': 'MATH201', 'name': 'Linear Algebra', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Theory'},
            {'code': 'MATH202', 'name': 'Statistics', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Data Science'},
            {'code': 'CS301', 'name': 'Theory of Computation', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Theory'},
            
            # AI and Machine Learning
            {'code': 'CS310', 'name': 'Artificial Intelligence', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'AI'},
            {'code': 'CS311', 'name': 'Machine Learning', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'AI'},
            {'code': 'CS312', 'name': 'Deep Learning', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'AI'},
            {'code': 'CS313', 'name': 'Natural Language Processing', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'AI'},
            {'code': 'CS314', 'name': 'Computer Vision', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'AI'},
            {'code': 'CS315', 'name': 'Robotics', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'AI'},
            
            # Data Science
            {'code': 'DS301', 'name': 'Data Mining', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Data Science'},
            {'code': 'DS302', 'name': 'Big Data Analytics', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Data Science'},
            {'code': 'DS303', 'name': 'Data Visualization', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Data Science'},
            {'code': 'DS304', 'name': 'Statistical Learning', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Data Science'},
            
            # Security
            {'code': 'SEC301', 'name': 'Cybersecurity Fundamentals', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Security'},
            {'code': 'SEC302', 'name': 'Cryptography', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Security'},
            {'code': 'SEC303', 'name': 'Network Security', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Security'},
            {'code': 'SEC304', 'name': 'Ethical Hacking', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Security'},
            
            # Advanced Software Engineering
            {'code': 'SE301', 'name': 'Advanced Programming', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Software Engineering'},
            {'code': 'SE302', 'name': 'Web Development', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Software Engineering'},
            {'code': 'SE303', 'name': 'Mobile Development', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Software Engineering'},
            {'code': 'SE304', 'name': 'DevOps', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Software Engineering'},
            
            # Electives
            {'code': 'CS401', 'name': 'Distributed Systems', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Systems'},
            {'code': 'CS402', 'name': 'Compiler Design', 'credits': 3, 'difficulty': 'Advanced', 'domain': 'Theory'},
            {'code': 'CS403', 'name': 'Game Development', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Software Engineering'},
            {'code': 'CS404', 'name': 'Human-Computer Interaction', 'credits': 3, 'difficulty': 'Intermediate', 'domain': 'Software Engineering'},
        ]
        
        # Add courses to graph
        for course in courses:
            self.graph.add_node(course['code'], **course)
        
        # Define prerequisite relationships
        prerequisites = [
            # Basic dependencies
            ('CS101', 'CS102'),
            ('CS102', 'CS201'),
            ('CS102', 'CS250'),
            ('CS101', 'CS210'),
            ('CS210', 'CS220'),
            ('CS210', 'CS240'),
            ('CS102', 'CS230'),
            ('MATH101', 'CS201'),
            ('MATH101', 'CS301'),
            ('MATH201', 'CS311'),
            ('MATH202', 'CS311'),
            ('MATH202', 'DS304'),
            
            # AI prerequisites
            ('CS201', 'CS310'),
            ('CS310', 'CS311'),
            ('CS311', 'CS312'),
            ('CS311', 'CS313'),
            ('CS311', 'CS314'),
            ('CS310', 'CS315'),
            ('CS220', 'CS315'),
            
            # Data Science prerequisites
            ('CS230', 'DS301'),
            ('CS230', 'DS302'),
            ('CS102', 'DS303'),
            ('MATH202', 'DS301'),
            ('MATH201', 'DS302'),
            
            # Security prerequisites
            ('CS240', 'SEC301'),
            ('SEC301', 'SEC302'),
            ('SEC301', 'SEC303'),
            ('SEC301', 'SEC304'),
            ('MATH101', 'SEC302'),
            
            # Advanced Software Engineering
            ('CS250', 'SE301'),
            ('CS102', 'SE302'),
            ('SE302', 'SE303'),
            ('CS220', 'SE304'),
            
            # Advanced courses
            ('CS220', 'CS401'),
            ('CS201', 'CS402'),
            ('SE302', 'CS403'),
            ('CS102', 'CS404'),
        ]
        
        # Add prerequisite edges
        for prereq, course in prerequisites:
            self.graph.add_edge(prereq, course, relationship='prerequisite')
    
    def get_prerequisites(self, course_code: str) -> List[str]:
        """Get direct prerequisites for a course"""
        return list(self.graph.predecessors(course_code))
    
    def get_all_prerequisites(self, course_code: str) -> Set[str]:
        """Get all prerequisites (transitive closure) for a course"""
        if course_code not in self.graph:
            return set()
        
        all_prereqs = set()
        to_visit = list(self.graph.predecessors(course_code))
        
        while to_visit:
            prereq = to_visit.pop()
            if prereq not in all_prereqs:
                all_prereqs.add(prereq)
                to_visit.extend(self.graph.predecessors(prereq))
        
        return all_prereqs
    
    def can_take_course(self, course_code: str, completed_courses: Set[str]) -> bool:
        """Check if student can take a course based on completed prerequisites"""
        prerequisites = set(self.get_prerequisites(course_code))
        return prerequisites.issubset(completed_courses)
    
    def get_eligible_courses(self, completed_courses: Set[str], 
                           failed_courses: Set[str] = None) -> List[str]:
        """Get all courses a student is eligible to take"""
        if failed_courses is None:
            failed_courses = set()
        
        eligible = []
        for course in self.graph.nodes():
            if (course not in completed_courses and 
                self.can_take_course(course, completed_courses)):
                eligible.append(course)
        
        # Add failed courses that can be retaken
        for course in failed_courses:
            if self.can_take_course(course, completed_courses):
                eligible.append(course)
        
        return eligible
    
    def get_course_info(self, course_code: str) -> Dict:
        """Get course information"""
        if course_code in self.graph:
            return dict(self.graph.nodes[course_code])
        return {}
    
    def get_courses_by_domain(self, domain: str) -> List[str]:
        """Get all courses in a specific domain"""
        return [node for node, data in self.graph.nodes(data=True) 
                if data.get('domain') == domain]
    
    def get_graduation_path(self, completed_courses: Set[str], 
                          target_credits: int = 120) -> List[str]:
        """Get a possible path to graduation"""
        current_credits = sum(self.graph.nodes[course].get('credits', 3) 
                            for course in completed_courses)
        
        if current_credits >= target_credits:
            return []
        
        remaining_courses = []
        eligible = self.get_eligible_courses(completed_courses)
        
        # Simple greedy approach - take eligible courses until credits met
        temp_completed = completed_courses.copy()
        credits_needed = target_credits - current_credits
        
        while credits_needed > 0 and eligible:
            # Prioritize advanced courses in AI/Data Science
            eligible.sort(key=lambda x: (
                self.graph.nodes[x].get('domain') in ['AI', 'Data Science'],
                self.graph.nodes[x].get('difficulty') == 'Advanced'
            ), reverse=True)
            
            course = eligible.pop(0)
            course_credits = self.graph.nodes[course].get('credits', 3)
            
            remaining_courses.append(course)
            temp_completed.add(course)
            credits_needed -= course_credits
            
            # Update eligible courses
            eligible = self.get_eligible_courses(temp_completed)
            eligible = [c for c in eligible if c not in remaining_courses]
        
        return remaining_courses
    
    def save_to_file(self, filepath: str) -> None:
        """Save curriculum graph to JSON file"""
        data = {
            'nodes': [
                {
                    'code': node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    **self.graph.edges[edge]
                }
                for edge in self.graph.edges()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load curriculum graph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.graph.clear()
        
        # Add nodes
        for node_data in data['nodes']:
            code = node_data.pop('code')
            self.graph.add_node(code, **node_data)
        
        # Add edges
        for edge_data in data['edges']:
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            self.graph.add_edge(source, target, **edge_data)
    
    def get_stats(self) -> Dict:
        """Get curriculum statistics"""
        return {
            'total_courses': len(self.graph.nodes()),
            'total_prerequisites': len(self.graph.edges()),
            'domains': {domain: len(self.get_courses_by_domain(domain)) 
                       for domain in self.domains},
            'avg_prerequisites': sum(len(list(self.graph.predecessors(node))) 
                                   for node in self.graph.nodes()) / len(self.graph.nodes()),
            'complexity': nx.number_of_nodes(self.graph) + nx.number_of_edges(self.graph)
        }


def create_sample_curriculum() -> CurriculumGraph:
    """Create and return a sample curriculum graph"""
    curriculum = CurriculumGraph()
    curriculum.build_curriculum()
    return curriculum


if __name__ == "__main__":
    # Create and test curriculum graph
    curriculum = create_sample_curriculum()
    
    print("Curriculum Statistics:")
    stats = curriculum.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample course info (CS311 - Machine Learning):")
    print(curriculum.get_course_info('CS311'))
    
    print("\nPrerequisites for CS311:")
    print(curriculum.get_all_prerequisites('CS311'))
    
    # Save to file
    curriculum.save_to_file('/Users/mohamedahmed/NU Research Task/data/curriculum_data.json')
    print("\nCurriculum saved to data/curriculum_data.json")
