# AI Curriculum Planner: Adaptive Academic Advising System

## Overview
This project implements an AI-powered academic advising system that uses graph-based curriculum modeling and reinforcement learning to provide personalized course recommendations for 100 simulated students. The system includes a comprehensive **Streamlit web application** for interactive visualization and analysis of curriculum data, student profiles, and training metrics.

## 🚀 Live Demo
Check out the interactive web application built with Streamlit:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-curriculum-planner.streamlit.app/)

**Note**: The application is running on a community cluster and may take a moment to start up if it has been inactive. Please be patient!

## Features
- **Graph-based Curriculum Model**: Courses as nodes, prerequisites as edges
- **Student Simulation**: 100 diverse students with different academic backgrounds
- **Reinforcement Learning**: Personalized course recommendations using Deep Q-Networks
- **Constraint Modeling**: Course load limits, prerequisites, retake policies
- **Interactive Web Dashboard**: Multi-page Streamlit application with real-time visualizations
- **Comprehensive Analytics**: Student performance metrics, curriculum analysis, and training insights

## Project Structure
```
├── src/
│   ├── curriculum_graph.py      # Graph-based curriculum modeling
│   ├── student_simulation.py    # Student data generation with statistical distributions
│   ├── rl_advisor.py            # Deep Q-Network reinforcement learning advisor
│   ├── constraints.py           # Academic constraints and validation
│   └── visualization.py         # Graph and data visualization utilities
├── app/
│   ├── main.py                  # Streamlit application entry point
│   └── pages2/                  # Multi-page application modules
│       ├── home.py              # Dashboard overview and key metrics
│       ├── curriculum_graph.py  # Interactive curriculum visualization
│       ├── student_analysis.py  # Student population analytics
│       ├── student_dashboard.py # Individual student profiles
│       ├── course_recommendations.py # RL-generated recommendations
│       ├── training_metrics.py  # RL training progress and metrics
│       └── data_explorer.py     # Raw data exploration tools
├── data/
|   |__ trained_advisor.pth      # Trained Pytorch Model
│   ├── curriculum_data.json     # Course and prerequisite data
│   ├── students_data.json       # Simulated student profiles
│   ├── training_stats.json      # RL training metrics and history
│   └── recommendations.json     # Generated recommendations
├── notebooks/
│   ├── curriculum_analysis.ipynb    # Curriculum exploration
│   ├── student_simulation.ipynb     # Student data analysis
│   └── rl_training.ipynb            # RL model training and evaluation
├── reports/
│   └── ai_curriculum_planner_report.pdf
├── requirements.txt
└── main.py                      # CLI execution script
```

## Installation
```bash
# Clone the repository and install dependencies
pip install -r requirements.txt

# Additional dependencies for Streamlit app
pip install streamlit plotly
```

## Usage

### Streamlit Web Application (Recommended)
Launch the interactive dashboard:
```bash
cd app
streamlit run main.py
```

The web application provides:
- **Home**: Overview dashboard with key statistics and system metrics
- **Curriculum Graph**: Interactive visualization of course prerequisites and relationships
- **Student Analysis**: Population-level analytics with filtering and insights
- **Student Dashboard**: Individual student profiles and academic progress
- **Course Recommendations**: RL-generated personalized course suggestions
- **Training Metrics**: Deep Q-Network training progress and performance analytics
- **Data Explorer**: Raw data inspection and statistical distributions

### Command Line Interface
Generate data and train models via CLI:

#### 1. Generate Curriculum and Student Data
```bash
python main.py --generate-data
```

#### 2. Train RL Model
```bash
python main.py --train-rl --episodes 2000
```

#### 3. Generate Recommendations
```bash
python main.py --recommend
```

#### 4. Visualize Results
```bash
python main.py --visualize
```

## Key Components

### Graph Schema
- **Nodes**: Courses with attributes (code, name, credits, difficulty, domain)
- **Edges**: Prerequisites relationships with weights
- **Constraints**: Course load limits, GPA requirements, retake policies

### Student Data Generation
The system generates realistic student profiles using sophisticated statistical distributions:

#### Academic Ability Distribution
- **Student Ability**: Beta(2, 2) distribution - creates realistic bell-curve of academic performance
- **Grade Simulation**: Course-specific normal distributions based on difficulty and student factors

#### Grade Distribution Parameters
```python
Grade Means by Difficulty:
- Beginner courses: μ = 3.2, σ = 0.6
- Intermediate courses: μ = 2.9, σ = 0.8  
- Advanced courses: μ = 2.7, σ = 0.9
```

#### Interest Profile Generation
- **Primary Interests**: Each student has 1-3 primary domains (0.7-1.0 interest level)
- **Secondary Interests**: Other domains receive 0.1-0.5 interest level
- **Domains**: AI, Security, Data Science, Software Engineering, Systems, Theory

#### Academic Progression Modeling
- **Course Selection**: Interest-weighted scoring with term appropriateness
- **Failure Simulation**: Courses with grade < 2.0 marked as failed
- **GPA Calculation**: Credit-weighted average of passed courses only
- **Academic Standing**: Determined by GPA and failed course count
  - Good: GPA ≥ 3.0 and ≤1 failed course
  - Warning: GPA ≥ 2.0 
  - Probation: GPA < 2.0

### Reinforcement Learning Algorithm
The system uses **Deep Q-Networks (DQN)** for personalized course recommendations:

#### State Representation
```python
State Vector Components:
- Course completion binary vector (one-hot encoding)
- Current GPA (normalized 0-1)
- Current term (normalized 0-1) 
- Credits completed (normalized 0-1)
- Failed courses count (normalized)
- Academic standing (one-hot)
- Interest vector (6 domains, 0-1 each)
```

#### Action Space
- **Actions**: Select 3-5 courses for next term
- **Constraints**: Must satisfy prerequisites and academic policies
- **Validation**: Real-time constraint checking via AcademicConstraints class

#### Reward Function
The reward function balances multiple objectives:

```python
R(s,a) = R_interest + R_constraints + R_count + R_difficulty

Where:
- R_interest = Σ(interest_level × 5.0) for each course
- R_constraints = +5.0 if valid, -min(3.0, violations×0.5) if invalid  
- R_count = +2.0 if exactly 3 courses, -2.0×(excess) if >3
- R_difficulty = +1.0 for GPA-appropriate difficulty matching
```

#### Network Architecture
- **Input Layer**: State vector (variable size based on curriculum)
- **Hidden Layers**: [512, 256, 128] neurons with ReLU activation
- **Dropout**: 0.2 rate for regularization
- **Output Layer**: Q-values for each possible course

### Score Functions and Equations

#### Interest Alignment Score
```python
Interest_Score = Σ(course_domain_interest × weight) / num_courses
```

#### GPA Prediction
```python
New_GPA = (current_GPA × current_credits + Σ(predicted_grade × credits)) / total_credits
```

#### Graduation Efficiency
```python
Efficiency = (remaining_credits / avg_credits_per_term) / terms_remaining
```

#### Constraint Satisfaction Rate
```python
Satisfaction_Rate = valid_selections / total_recommendations
```

## Example Results
The system generates personalized recommendations for each student, considering:
- **Academic Prerequisites**: Ensures all prerequisite courses are completed
- **Individual Interests**: Matches recommendations to student's domain preferences
- **GPA Optimization**: Suggests appropriate difficulty levels based on academic standing
- **Graduation Timeline**: Balances course load with efficient degree completion
- **Learning Sequences**: Recommends logical course progressions within domains

### Sample Recommendation Output
```json
{
  "student_id": "Student_042",
  "recommended_courses": ["CS310", "SEC301", "DS201"],
  "reasoning": {
    "interest_match": 0.89,
    "difficulty_appropriate": true,
    "prerequisites_satisfied": true,
    "estimated_gpa_impact": +0.15
  }
}
```

## Quick Start Guide

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit App**:
   ```bash
   cd app && streamlit run main.py
   ```

3. **Explore the Dashboard**:
   - Navigate using the sidebar menu
   - Filter data using interactive controls
   - View detailed course and student information
   - Analyze RL training progress and metrics

4. **Generate New Data** (Optional):
   ```bash
   python main.py --generate-data
   ```

## Technologies Used
- **Frontend**: Streamlit for interactive web dashboard
- **Graph Modeling**: NetworkX for curriculum graph analysis
- **Reinforcement Learning**: PyTorch for Deep Q-Networks implementation
- **Data Processing**: Pandas/NumPy for data manipulation and analysis
- **Visualization**: Plotly for interactive charts, Matplotlib for static plots
- **Statistical Modeling**: SciPy, NumPy for probability distributions
- **Machine Learning**: Scikit-learn for additional ML utilities

## Streamlit Application Features

### Navigation
- **Sidebar Navigation**: Clean, consistent navigation across all pages
- **Real-time Updates**: Dynamic filtering and interactive visualizations
- **Responsive Design**: Optimized for different screen sizes

### Key Pages
1. **Home Dashboard**: System overview with key performance indicators
2. **Curriculum Graph**: Interactive network visualization with course details
3. **Student Analysis**: Population analytics with filtering capabilities  
4. **Individual Dashboards**: Detailed student profiles and progress tracking
5. **RL Recommendations**: Personalized course suggestions with explanations
6. **Training Metrics**: Deep Q-Network performance and learning curves
7. **Data Explorer**: Raw data inspection with statistical summaries

### Interactive Features
- **Dynamic Filtering**: Filter students by GPA, term, academic standing
- **Course Search**: Searchable course selection and detailed information
- **Real-time Metrics**: Live calculation of statistics and KPIs
- **Export Capabilities**: Download filtered data and visualizations

## Statistical Distributions Summary

### Grade Distribution by Course Difficulty
| Difficulty Level | Mean GPA | Standard Deviation | Grade Range |
|------------------|----------|-------------------|-------------|
| Beginner        | 3.2      | 0.6               | 2.0 - 4.0   |
| Intermediate    | 2.9      | 0.8               | 1.3 - 4.0   |
| Advanced        | 2.7      | 0.9               | 1.0 - 4.0   |

### Student Ability Distribution
- **Distribution**: Beta(α=2, β=2) 
- **Mean**: 0.5 (average ability)
- **Variance**: 0.05 (moderate spread)
- **Shape**: Symmetric bell curve concentrated around mean

### Interest Level Distribution
- **Primary Interests**: Uniform(0.7, 1.0) for 1-3 selected domains
- **Secondary Interests**: Uniform(0.1, 0.5) for remaining domains
- **Domain Coverage**: All students have non-zero interest in each domain

### Academic Progression Patterns
- **Course Load**: Uniform(3, 5) courses per term
- **Early Stop Probability**: 10% chance after term 4 (transfer/gap year)
- **Foundation Priority**: Terms 1-2 prioritize beginner courses
- **Specialization**: Terms 5+ focus on advanced domain-specific courses

## Performance Metrics

### Curriculum Modeling
- **Graph Density**: 42 courses, 89 prerequisite relationships
- **Average Prerequisites**: 2.1 per course
- **Longest Path**: 6 course sequence (foundation to advanced)
- **Domain Distribution**: Balanced across 6 academic areas

### Student Population Statistics  
- **Total Students**: 100 diverse academic profiles
- **Average GPA**: 2.85 (realistic university distribution)
- **Completion Rate**: 85% (15% early stop simulation)
- **Failed Courses**: 12% average failure rate per student

### RL Training Performance
- **Training Episodes**: 2000+ episodes for convergence
- **Final Reward**: Stable convergence around episode 1500
- **Interest Alignment**: 89% average match with student preferences  
- **Constraint Satisfaction**: 94% valid recommendation rate
- **Learning Efficiency**: Consistent improvement over 1000+ episodes

### System Metrics
- **Recommendation Accuracy**: 92% student satisfaction simulation
- **Processing Speed**: <100ms per recommendation
- **Data Coverage**: 100% curriculum graph traversal
- **Visualization Performance**: Real-time interactive updates

## Future Enhancements
- **Real University Integration**: Connect with actual student information systems
- **Multi-objective Optimization**: Balance competing goals (GPA, graduation time, cost)
- **Advanced RL Algorithms**: Implement PPO, A3C for improved performance  
- **Real-time Constraint Updates**: Dynamic prerequisite and policy changes
- **Mobile-responsive Design**: Optimize Streamlit app for mobile devices
- **A/B Testing Framework**: Compare recommendation strategies
- **Integration APIs**: RESTful endpoints for external system integration

## License
MIT License
