Of course\! Here is the updated README in Markdown format, now including the Streamlit logo next to the link.

-----

# AI Curriculum Planner: Adaptive Academic Advising System

## Overview

This project implements an AI-powered academic advising system that uses graph-based curriculum modeling and reinforcement learning to provide personalized course recommendations for 100 simulated students. The system includes a comprehensive **Streamlit web application** for interactive visualization and analysis of curriculum data, student profiles, and training metrics.

## 🚀 Live Demo

Check out the interactive web application built with Streamlit:

[](https://ai-curriculum-planner.streamlit.app/)

**Note**: The application is running on a community cluster and may take a moment to start up if it has been inactive. Please be patient\!

-----

## Features

  - **Graph-based Curriculum Model**: Courses as nodes, prerequisites as edges.
  - **Student Simulation**: 100 diverse students with different academic backgrounds.
  - **Reinforcement Learning**: Personalized course recommendations using Deep Q-Networks.
  - **Constraint Modeling**: Course load limits, prerequisites, and retake policies.
  - **Interactive Web Dashboard**: Multi-page Streamlit application with real-time visualizations.
  - **Comprehensive Analytics**: Student performance metrics, curriculum analysis, and training insights.

-----

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
│   ├── curriculum_data.json     # Course and prerequisite data
│   ├── students_data.json       # Simulated student profiles
│   ├── trained_advisor.pth      # Trained PyTorch model
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

-----

## Installation

```bash
# Clone the repository and install dependencies
pip install -r requirements.txt

# Additional dependencies for Streamlit app
pip install streamlit plotly
```

-----

## Usage

### Streamlit Web Application (Recommended)

Launch the interactive dashboard:

```bash
cd app
streamlit run main.py
```

The web application provides:

  - **Home**: Overview dashboard with key statistics and system metrics.
  - **Curriculum Graph**: Interactive visualization of course prerequisites and relationships.
  - **Student Analysis**: Population-level analytics with filtering and insights.
  - **Student Dashboard**: Individual student profiles and academic progress.
  - **Course Recommendations**: RL-generated personalized course suggestions.
  - **Training Metrics**: Deep Q-Network training progress and performance analytics.
  - **Data Explorer**: Raw data inspection and statistical distributions.

### Command Line Interface

Generate data and train models via CLI:

#### 1\. Generate Curriculum and Student Data

```bash
python main.py --generate-data
```

#### 2\. Train RL Model

```bash
python main.py --train-rl --episodes 2000
```

#### 3\. Generate Recommendations

```bash
python main.py --recommend
```

#### 4\. Visualize Results

```bash
python main.py --visualize
```

-----

## Key Components

### Graph Schema

  - **Nodes**: Courses with attributes (code, name, credits, difficulty, domain).
  - **Edges**: Prerequisite relationships with weights.
  - **Constraints**: Course load limits, GPA requirements, retake policies.

### Student Data Generation

The system generates realistic student profiles using sophisticated statistical distributions:

#### Academic Ability Distribution

  - **Student Ability**: Beta(2, 2) distribution creates a realistic bell-curve of academic performance.
  - **Grade Simulation**: Course-specific normal distributions based on difficulty and student factors.

#### Interest Profile Generation

  - **Primary Interests**: Each student has 1-3 primary domains (0.7-1.0 interest level).
  - **Secondary Interests**: Other domains receive a 0.1-0.5 interest level.
  - **Domains**: AI, Security, Data Science, Software Engineering, Systems, Theory.

-----

## Reinforcement Learning Algorithm

The system uses **Deep Q-Networks (DQN)** for personalized course recommendations:

### State Representation

The state vector includes:

  - Course completion binary vector (one-hot encoding).
  - Current GPA, term, credits completed (normalized).
  - Failed courses count and academic standing (one-hot).
  - Interest vector (6 domains, 0-1 each).

### Reward Function

The reward function balances multiple objectives:
`R(s,a) = R_interest + R_constraints + R_count + R_difficulty`

  - `R_interest`: High reward for courses matching student interests.
  - `R_constraints`: Bonus for valid selections, penalty for violations.
  - `R_count`: Reward for an optimal number of courses.
  - `R_difficulty`: Reward for GPA-appropriate difficulty matching.

### Network Architecture

  - **Input Layer**: State vector.
  - **Hidden Layers**: [512, 256, 128] neurons with ReLU activation and Dropout.
  - **Output Layer**: Q-values for each possible course.

-----

## Performance Metrics

  - **Interest Alignment**: 89% average match with student preferences.
  - **Constraint Satisfaction**: 94% valid recommendation rate.
  - **Recommendation Accuracy**: 92% simulated student satisfaction.
  - **Processing Speed**: \<100ms per recommendation.

-----

## Technologies Used

  - **Frontend**: Streamlit
  - **Graph Modeling**: NetworkX
  - **Reinforcement Learning**: PyTorch
  - **Data Processing**: Pandas/NumPy
  - **Visualization**: Plotly, Matplotlib

-----

## License

MIT License
