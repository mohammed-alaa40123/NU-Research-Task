# AI Curriculum Planner: Adaptive Academic Advising System

## Overview
This project implements an AI-powered academic advising system that uses graph-based curriculum modeling and reinforcement learning to provide personalized course recommendations for 100 simulated students.

## Features
- **Graph-based Curriculum Model**: Courses as nodes, prerequisites as edges
- **Student Simulation**: 100 diverse students with different academic backgrounds
- **Reinforcement Learning**: Personalized course recommendations
- **Constraint Modeling**: Course load limits, prerequisites, retake policies
- **Visualization**: Interactive curriculum graphs and student progress

## Project Structure
```
├── src/
│   ├── curriculum_graph.py      # Graph-based curriculum modeling
│   ├── student_simulation.py    # Student data generation
│   ├── rl_advisor.py            # Reinforcement learning advisor
│   ├── constraints.py           # Academic constraints
│   └── visualization.py         # Graph and data visualization
├── data/
│   ├── curriculum_data.json     # Course and prerequisite data
│   ├── students_data.json       # Simulated student profiles
│   └── recommendations.json     # Generated recommendations
├── notebooks/
│   ├── curriculum_analysis.ipynb    # Curriculum exploration
│   ├── student_simulation.ipynb     # Student data analysis
│   └── rl_training.ipynb            # RL model training
├── reports/
│   └── ai_curriculum_planner_report.pdf
├── requirements.txt
└── main.py                      # Main execution script
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Curriculum and Student Data
```bash
python main.py --generate-data
```

### 2. Train RL Model
```bash
python main.py --train-rl --episodes 2000
```

### 3. Generate Recommendations
```bash
python main.py --recommend
```

### 4. Visualize Results
```bash
python main.py --visualize
```

## Key Components

### Graph Schema
- **Nodes**: Courses with attributes (code, name, credits, difficulty, domain)
- **Edges**: Prerequisites relationships with weights
- **Constraints**: Course load limits, GPA requirements, retake policies

### Student Simulation
Each student has:
- Completed courses with grades
- Current GPA
- Academic interests (AI, Security, Data Science, etc.)
- Term number and graduation timeline

### RL Approach
- **State**: [completed_courses, GPA, term, interests_vector]
- **Action**: Select 3-5 courses for next term
- **Reward**: GPA improvement + interest alignment + graduation progress

## Example Results
The system generates personalized recommendations for each student, considering:
- Academic prerequisites
- Individual interests and strengths
- GPA optimization
- Graduation timeline efficiency

## Technologies Used
- **NetworkX**: Graph modeling and analysis
- **Stable-Baselines3**: Reinforcement learning
- **Pandas/NumPy**: Data manipulation
- **Plotly/Matplotlib**: Visualization
- **Scikit-learn**: Machine learning utilities

## Performance Metrics
- Average GPA improvement: X.XX points
- Graduation efficiency: XX% faster completion
- Interest alignment score: XX%
- Constraint satisfaction: XX% compliance

## Future Enhancements
- Integration with real university data
- Multi-objective optimization
- Real-time constraint updates
- Advanced visualization dashboard

## License
MIT License
