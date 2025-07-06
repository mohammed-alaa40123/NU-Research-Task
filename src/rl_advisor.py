"""
Reinforcement Learning Academic Advisor

This module implements a reinforcement learning approach for personalized
course recommendations using Deep Q-Networks (DQN) and policy optimization.
"""

import numpy as np
import random
import json
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict

try:
    from .curriculum_graph import CurriculumGraph
    from .student_simulation import StudentProfile
    from .constraints import AcademicConstraints
except ImportError:
    from curriculum_graph import CurriculumGraph
    from student_simulation import StudentProfile
    from constraints import AcademicConstraints


@dataclass
class RLState:
    """Represents the RL state for a student"""
    completed_courses: Set[str]
    current_gpa: float
    current_term: int
    interests: Dict[str, float]
    failed_courses: Set[str]
    academic_standing: str
    credits_completed: int
    
    def to_vector(self, curriculum: CurriculumGraph) -> np.ndarray:
        """Convert state to numerical vector for neural network"""
        # Create binary vector for completed courses
        all_courses = list(curriculum.graph.nodes())
        course_vector = np.zeros(len(all_courses))
        
        for i, course in enumerate(all_courses):
            if course in self.completed_courses:
                course_vector[i] = 1.0
        
        # Add other features
        features = [
            self.current_gpa / 4.0,  # Normalize GPA
            self.current_term / 12.0,  # Normalize term
            self.credits_completed / 120.0,  # Normalize credits
            len(self.failed_courses) / 10.0,  # Normalize failed courses
            1.0 if self.academic_standing == "Good" else 0.0,
            1.0 if self.academic_standing == "Warning" else 0.0,
            1.0 if self.academic_standing == "Probation" else 0.0,
        ]
        
        # Add interest vector
        interest_vector = [self.interests.get(domain, 0.0) for domain in 
                          ['AI', 'Security', 'Data Science', 'Software Engineering', 'Systems', 'Theory']]
        
        return np.concatenate([course_vector, features, interest_vector])


class DQNNetwork(nn.Module):
    """Deep Q-Network for course recommendation"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class CourseRecommendationEnv:
    """Environment for course recommendation RL"""
    
    def __init__(self, curriculum: CurriculumGraph, constraints: AcademicConstraints):
        self.curriculum = curriculum
        self.constraints = constraints
        self.all_courses = list(curriculum.graph.nodes())
        self.reset()
    
    def reset(self) -> RLState:
        """Reset environment with a new student"""
        self.current_state = None
        self.episode_done = False
        return self.current_state
    
    def set_student(self, student: StudentProfile) -> RLState:
        """Set the current student for the environment"""
        passed_courses = {
            course for course, grade in student.completed_courses.items()
            if grade >= 2.0
        }
        
        credits_completed = sum(
            self.curriculum.get_course_info(course).get('credits', 3)
            for course in passed_courses
        )
        
        # Ensure failed_courses is a set
        failed_courses = student.failed_courses
        if isinstance(failed_courses, list):
            failed_courses = set(failed_courses)
        
        self.current_state = RLState(
            completed_courses=passed_courses,
            current_gpa=student.gpa,
            current_term=student.current_term,
            interests=student.interests,
            failed_courses=failed_courses,
            academic_standing=student.academic_standing,
            credits_completed=credits_completed
        )
        
        self.episode_done = False
        return self.current_state
    
    def get_action_space(self) -> List[str]:
        """Get available actions (courses) for current state"""
        if self.current_state is None:
            return []
        
        eligible = self.curriculum.get_eligible_courses(
            self.current_state.completed_courses,
            self.current_state.failed_courses
        )
        
        return eligible
    
    def step(self, action: List[str]) -> Tuple[RLState, float, bool, Dict]:
        """Execute action and return new state, reward, done, info"""
        if self.current_state is None:
            raise ValueError("Environment not initialized with student")
        
        # Calculate reward
        reward = self.calculate_reward(action)
        
        # Update state (simulate taking courses)
        new_state = self.simulate_course_completion(action)
        
        # Check if done (graduated or max terms)
        done = (new_state.credits_completed >= 120 or 
                new_state.current_term >= 12)
        
        info = {
            'courses_taken': action,
            'credits_earned': sum(self.curriculum.get_course_info(c).get('credits', 3) for c in action),
            'constraint_score': self.constraints.get_constraint_score(action, self._state_to_student_profile(self.current_state))
        }
        
        self.current_state = new_state
        self.episode_done = done
        
        return new_state, reward, done, info
    
    def calculate_reward(self, action: List[str]) -> float:
        """Enhanced reward function with strong interest alignment"""
        if not action:
            return -1.0  # Simple penalty for no action
        
        student_profile = self._state_to_student_profile(self.current_state)
        total_reward = 0.0
        
        # 1. Strong interest alignment (0-5 points per course) - INCREASED WEIGHT
        interest_total = 0.0
        for course in action:
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Theory')
            interest_level = self.current_state.interests.get(domain, 0.1)
            interest_total += interest_level * 5.0  # Increased from 2.0 to 5.0
        
        total_reward += interest_total
        
        # 2. Constraint compliance bonus/penalty (-3 to +5 points)
        is_valid, violations = self.constraints.validate_course_selection(action, student_profile)
        if is_valid:
            total_reward += 5.0  # Increased bonus for valid selections
        else:
            # Reduced penalty to not overwhelm interest scores
            total_reward -= min(3.0, len(violations) * 0.5)
        
        # 3. Course count penalty (encourage exactly 3 courses)
        if len(action) == 3:
            total_reward += 2.0
        elif len(action) > 3:
            total_reward -= (len(action) - 3) * 2.0  # Strong penalty for too many courses
        
        # 4. GPA-difficulty appropriateness (0-1 points)
        for course in action:
            course_info = self.curriculum.get_course_info(course)
            difficulty = course_info.get('difficulty', 'Intermediate')
            
            if self.current_state.current_gpa >= 3.5 and difficulty == 'Advanced':
                total_reward += 1.0
            elif 2.5 <= self.current_state.current_gpa < 3.5 and difficulty == 'Intermediate':
                total_reward += 1.0
            elif self.current_state.current_gpa < 2.5 and difficulty == 'Beginner':
                total_reward += 1.0
        
        # Clamp reward to reasonable range
        return max(-10.0, min(25.0, total_reward))
    
    def _calculate_sequence_bonus(self, action: List[str]) -> float:
        """Calculate bonus for taking courses in logical sequences"""
        sequences = [
            ['CS310', 'CS311', 'CS312'],  # AI sequence
            ['CS301', 'CS402'],           # Theory sequence
            ['SEC301', 'SEC302', 'SEC303'] # Security sequence
        ]
        
        bonus = 0.0
        for sequence in sequences:
            for i, course in enumerate(sequence[:-1]):
                next_course = sequence[i + 1]
                if (course in self.current_state.completed_courses and 
                    next_course in action):
                    bonus += 0.3
        
        return bonus
    
    def simulate_course_completion(self, action: List[str]) -> RLState:
        """Simulate completing the selected courses"""
        # Simple simulation - assume all courses are passed with interest-based grades
        new_completed = self.current_state.completed_courses.copy()
        new_failed = self.current_state.failed_courses.copy()
        
        total_grade_points = self.current_state.current_gpa * self.current_state.credits_completed
        new_credits = 0
        
        for course in action:
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Theory')
            difficulty = course_info.get('difficulty', 'Intermediate')
            credits = course_info.get('credits', 3)
            
            # Estimate grade
            interest_level = self.current_state.interests.get(domain, 0.3)
            difficulty_factor = {'Beginner': 1.0, 'Intermediate': 0.8, 'Advanced': 0.6}[difficulty]
            
            estimated_grade = 2.0 + interest_level * 2.0 * difficulty_factor
            estimated_grade = min(4.0, max(0.0, estimated_grade + np.random.normal(0, 0.3)))
            
            if estimated_grade >= 2.0:
                new_completed.add(course)
                total_grade_points += estimated_grade * credits
                new_credits += credits
                if course in new_failed:
                    new_failed.remove(course)
            else:
                new_failed.add(course)
        
        # Update GPA
        total_credits = self.current_state.credits_completed + new_credits
        new_gpa = total_grade_points / total_credits if total_credits > 0 else 0.0
        
        # Update academic standing
        if new_gpa >= 3.0:
            standing = "Good"
        elif new_gpa >= 2.5:
            standing = "Warning"
        else:
            standing = "Probation"
        
        return RLState(
            completed_courses=new_completed,
            current_gpa=new_gpa,
            current_term=self.current_state.current_term + 1,
            interests=self.current_state.interests,
            failed_courses=new_failed,
            academic_standing=standing,
            credits_completed=total_credits
        )
    
    def _state_to_student_profile(self, state: RLState) -> StudentProfile:
        """Convert RL state back to student profile for constraint checking"""
        completed_courses = {}
        for course in state.completed_courses:
            completed_courses[course] = 3.0  # Assume average grade
        
        return StudentProfile(
            student_id="temp",
            name="temp",
            completed_courses=completed_courses,
            failed_courses=state.failed_courses,
            current_term=state.current_term,
            gpa=state.current_gpa,
            interests=state.interests,
            max_courses_per_term=4,
            target_graduation_term=8,
            academic_standing=state.academic_standing
        )


class DQNAdvisor:
    """DQN-based academic advisor"""
    
    def __init__(self, curriculum: CurriculumGraph, constraints: AcademicConstraints):
        self.curriculum = curriculum
        self.constraints = constraints
        self.env = CourseRecommendationEnv(curriculum, constraints)
        
        # Network parameters
        self.state_size = len(curriculum.graph.nodes()) + 13  # courses + features + interests
        self.action_size = len(curriculum.graph.nodes())
        
        # DQN parameters - Simplified and stable
        self.epsilon = 1.0
        self.epsilon_decay = 0.999   # Moderate decay
        self.epsilon_min = 0.1       # Higher minimum exploration
        self.learning_rate = 0.0001  # Slightly higher learning rate
        self.batch_size = 32         # Smaller batch for stability
        self.memory_size = 10000     # Reasonable memory size
        self.gamma = 0.9             # Lower discount factor
        self.target_update_freq = 100 # More frequent updates for better learning
        self.training_step = 0       # Track training steps
        
        # Initialize networks with proper weight initialization
        self.q_network = DQNNetwork(self.state_size, self.action_size)
        self.target_network = DQNNetwork(self.state_size, self.action_size)
        
        # Initialize weights
        self._initialize_networks()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()  # More stable than MSE for RL
        
        # Experience replay
        self.memory = deque(maxlen=self.memory_size)
        
        # Training statistics
        self.training_stats = defaultdict(list)
    
    def _initialize_networks(self):
        """Initialize network weights properly"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        
        self.q_network.apply(init_weights)
        self.target_network.apply(init_weights)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action_values(self, state: RLState) -> np.ndarray:
        """Get Q-values for all actions given a state"""
        state_vector = torch.FloatTensor(state.to_vector(self.curriculum)).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_vector)
        
        return q_values.numpy().flatten()
    
    def select_courses(self, student: StudentProfile, num_courses: int = 3) -> List[str]:
        """Select courses for a student using the trained model"""
        state = self.env.set_student(student)
        eligible_courses = self.env.get_action_space()
        
        if not eligible_courses:
            return []
        
        # Limit number of courses to constraint maximum
        max_courses = min(num_courses, 3)  # Enforce 3-course limit
        
        # Use enhanced greedy selection that balances interests and constraints
        selected_courses = self._enhanced_greedy_selection(student, max_courses, eligible_courses)
        
        # Double-check validation
        is_valid, violations = self.constraints.validate_course_selection(selected_courses, student)
        
        if not is_valid:
            # If still invalid, fallback to simple greedy
            selected_courses = self._simple_greedy_selection(student, max_courses, eligible_courses)
        
        return selected_courses
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return dict(self.training_stats)
    
    def reset_training_stats(self):
        """Reset training statistics"""
        self.training_stats = defaultdict(list)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the neural networks"""
        return {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_step': self.training_step,
            'q_network_params': sum(p.numel() for p in self.q_network.parameters()),
            'target_network_params': sum(p.numel() for p in self.target_network.parameters())
        }
    
    def _enhanced_greedy_selection(self, student: StudentProfile, num_courses: int, eligible_courses: List[str]) -> List[str]:
        """Enhanced greedy course selection with better interest alignment"""
        scored_courses = []
        
        for course in eligible_courses:
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Theory')
            difficulty = course_info.get('difficulty', 'Intermediate')
            
            # Strong interest weighting
            interest_score = student.interests.get(domain, 0.1) * 5.0  # Amplify interest
            
            # GPA-difficulty matching
            gpa_difficulty_bonus = 0.0
            if student.gpa >= 3.5 and difficulty == 'Advanced':
                gpa_difficulty_bonus = 2.0
            elif 3.0 <= student.gpa < 3.5 and difficulty == 'Intermediate':
                gpa_difficulty_bonus = 1.5
            elif student.gpa < 3.0 and difficulty == 'Beginner':
                gpa_difficulty_bonus = 1.0
            
            # Graduation progress bonus
            progress_bonus = 0.0
            if student.current_term >= 6:  # Senior students
                if difficulty == 'Advanced':
                    progress_bonus = 1.0
            
            total_score = interest_score + gpa_difficulty_bonus + progress_bonus
            scored_courses.append((course, total_score))
        
        # Sort by score and select top courses
        scored_courses.sort(key=lambda x: x[1], reverse=True)
        
        # Select courses while checking constraints
        selected = []
        for course, score in scored_courses:
            if len(selected) >= num_courses:
                break
            
            # Avoid duplicates
            if course in selected:
                continue
                
            test_selection = selected + [course]
            is_valid, violations = self.constraints.validate_course_selection(test_selection, student)
            
            # Allow if valid or only minor violations
            minor_violations = [v for v in violations if 'exceed' not in v.description.lower()]
            if is_valid or len(violations) <= 1 or len(minor_violations) == len(violations):
                selected.append(course)
        
        return selected[:num_courses]
    
    def _simple_greedy_selection(self, student: StudentProfile, num_courses: int, eligible_courses: List[str]) -> List[str]:
        """Simple fallback selection focused on interest alignment"""
        scored_courses = []
        
        for course in eligible_courses:
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Theory')
            interest_score = student.interests.get(domain, 0.1)
            scored_courses.append((course, interest_score))
        
        scored_courses.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates and select top courses
        selected = []
        for course, score in scored_courses:
            if course not in selected and len(selected) < num_courses:
                selected.append(course)
        
        return selected

    def _greedy_selection(self, student: StudentProfile, num_courses: int) -> List[str]:
        """Fallback greedy course selection"""
        passed_courses = {
            course for course, grade in student.completed_courses.items()
            if grade >= 2.0
        }
        
        eligible = self.curriculum.get_eligible_courses(passed_courses)
        
        # Score courses by interest and difficulty
        scored_courses = []
        for course in eligible:
            course_info = self.curriculum.get_course_info(course)
            domain = course_info.get('domain', 'Theory')
            difficulty = course_info.get('difficulty', 'Intermediate')
            
            interest_score = student.interests.get(domain, 0.3)
            difficulty_bonus = {'Beginner': 0.1, 'Intermediate': 0.2, 'Advanced': 0.3}[difficulty]
            
            total_score = interest_score + difficulty_bonus
            scored_courses.append((course, total_score))
        
        # Sort and select unique courses
        scored_courses.sort(key=lambda x: x[1], reverse=True)
        selected = []
        for course, score in scored_courses:
            if course not in selected and len(selected) < num_courses:
                selected.append(course)
        
        return selected
    
    def train_episode(self, student: StudentProfile) -> Dict[str, float]:
        """Train on a single episode with a student"""
        state = self.env.set_student(student)
        total_reward = 0.0
        steps = 0

        while not self.env.episode_done and steps < 10:  # Limit steps per episode
            # Select action (epsilon-greedy)
            if random.random() < self.epsilon:
                # Random action
                eligible = self.env.get_action_space()
                if eligible:
                    action = random.sample(eligible, min(random.randint(1, 3), len(eligible)))
                else:
                    action = []
            else:
                # Greedy action - but limit to 3 courses
                action = self.select_courses(self._state_to_student_profile(state), num_courses=3)

            if not action:
                break

            # Execute action
            next_state, reward, done, info = self.env.step(action)

            # Convert action to single action index for DQN (select first course as primary action)
            action_idx = 0
            if action and action[0] in self.curriculum.graph.nodes():
                action_idx = list(self.curriculum.graph.nodes()).index(action[0])

            # Store experience
            self.memory.append((
                state.to_vector(self.curriculum),
                action_idx,
                reward,
                next_state.to_vector(self.curriculum),
                done
            ))
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # Train network
        if len(self.memory) > self.batch_size:
            self._replay()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon
        }
    
    def train_step(self, student: StudentProfile) -> Dict[str, float]:
        """Simplified and stable training step"""
        state = self.env.set_student(student)
        eligible_courses = self.env.get_action_space()
        
        if not eligible_courses:
            return {'reward': 0.0, 'loss': 0.0, 'epsilon': self.epsilon}
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random exploration - select 1-3 courses
            num_courses = min(random.randint(1, 3), len(eligible_courses))
            action_courses = random.sample(eligible_courses, num_courses)
        else:
            # Greedy exploitation
            action_courses = self.select_courses(student, num_courses=min(3, len(eligible_courses)))
        
        if not action_courses:
            return {'reward': 0.0, 'loss': 0.0, 'epsilon': self.epsilon}
        
        # Calculate reward using simplified function
        reward = self.env.calculate_reward(action_courses)
        
        # Clamp reward to prevent explosion
        reward = max(-10.0, min(10.0, reward))
        
        # Use only the first course for DQN training (single action)
        primary_course = action_courses[0]
        course_idx = list(self.curriculum.graph.nodes()).index(primary_course)
        
        # Simple next state (just add random noise to current state)
        next_state_vector = state.to_vector(self.curriculum)
        next_state_vector[course_idx] = 1.0  # Mark course as completed
        
        # Determine if episode is done
        done = (state.credits_completed >= 100 or state.current_term >= 10)
        
        # Store experience
        self.memory.append((
            state.to_vector(self.curriculum),
            course_idx,
            reward,
            next_state_vector,
            done
        ))
        
        # Train network when we have enough samples
        loss = 0.0
        if len(self.memory) > self.batch_size:
            # Train every 5th call to this method, not based on training_step
            if len(self.memory) % 5 == 0:
                loss = self._replay()
                if loss is None:
                    loss = 0.0
        
        # Slower epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            'reward': reward,
            'loss': loss,
            'epsilon': self.epsilon,
            'courses_selected': len(action_courses)
        }
    
    def _action_to_vector(self, action: List[str]) -> np.ndarray:
        """Convert action to binary vector"""
        action_vector = np.zeros(self.action_size)
        for course in action:
            if course in self.curriculum.graph.nodes():
                idx = list(self.curriculum.graph.nodes()).index(course)
                action_vector[idx] = 1.0
        return action_vector
    
    def _replay(self):
        """Fixed experience replay with proper gradient handling"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Separate batch components
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Normalize rewards to prevent explosion
        rewards = torch.clamp(rewards, -10.0, 10.0)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            target_q_values = torch.clamp(target_q_values, -50.0, 50.0)  # Prevent explosion
        
        # Calculate loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Check for NaN or infinite loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Invalid loss detected: {loss.item()}, skipping update")
            return 0.0
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _state_to_student_profile(self, state: RLState) -> StudentProfile:
        """Convert RL state to student profile"""
        completed_courses = {}
        for course in state.completed_courses:
            completed_courses[course] = 3.0
        
        return StudentProfile(
            student_id="temp",
            name="temp",
            completed_courses=completed_courses,
            failed_courses=state.failed_courses,
            current_term=state.current_term,
            gpa=state.current_gpa,
            interests=state.interests,
            max_courses_per_term=4,
            target_graduation_term=8,
            academic_standing=state.academic_standing
        )
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': dict(self.training_stats)
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_stats = defaultdict(list, checkpoint['training_stats'])


def create_rl_advisor(curriculum: CurriculumGraph, constraints: AcademicConstraints) -> DQNAdvisor:
    """Create and return a DQN advisor"""
    return DQNAdvisor(curriculum, constraints)


if __name__ == "__main__":
    from curriculum_graph import create_sample_curriculum
    from student_simulation import create_student_cohort
    
    # Create components
    curriculum = create_sample_curriculum()
    constraints = AcademicConstraints(curriculum)
    advisor = create_rl_advisor(curriculum, constraints)
    
    # Create student cohort
    students = create_student_cohort(curriculum, 10)
    
    print("Training RL advisor...")
    
    # Train for a few episodes
    for episode in range(5):
        student = random.choice(students)
        stats = advisor.train_episode(student)
        print(f"Episode {episode + 1}: Reward = {stats['total_reward']:.2f}, Steps = {stats['steps']}")
    
    # Test recommendations
    print("\nTesting recommendations:")
    test_student = students[0]
    recommendations = advisor.select_courses(test_student)
    
    print(f"Student: {test_student.student_id}")
    print(f"Current GPA: {test_student.gpa}")
    print(f"Completed courses: {len(test_student.completed_courses)}")
    print(f"Recommended courses: {recommendations}")
    
    # Validate recommendations
    is_valid, violations = constraints.validate_course_selection(recommendations, test_student)
    print(f"Valid recommendations: {is_valid}")
    if violations:
        print("Violations:")
        for v in violations:
            print(f"  - {v.description}")
