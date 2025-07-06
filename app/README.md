# AI Curriculum Planner - Streamlit Application

A comprehensive web application for exploring and analyzing an AI-powered curriculum planning system using Streamlit.

## Features

### 🏠 Home
- Overview of the AI curriculum planning system
- System architecture and component descriptions
- Navigation guide and quick statistics

### 📊 Curriculum Graph
- Interactive visualization of the computer science curriculum
- Top-down hierarchical layout showing prerequisite relationships
- Color-coded domains and difficulty levels
- Course explorer with detailed information

### 👥 Student Analysis
- Comprehensive student cohort analytics
- GPA distributions and academic standing breakdowns
- Interest patterns and correlation analysis
- Performance metrics and trends

### 📋 Student Dashboard
- Individual student profile analysis
- Academic progress tracking and metrics
- Interest radar charts and domain preferences
- Personalized course recommendations
- Graduation pathway visualization

### 💡 Course Recommendations
- AI-powered course recommendation system
- Multiple recommendation strategies (RL, Interest-based, Prerequisites)
- Recommendation validation and constraint checking
- Semester planning and academic timeline

### 📈 Training Metrics
- Deep reinforcement learning model performance
- Training progress visualization (rewards, loss, epsilon decay)
- Learning curve analysis and convergence monitoring
- Model comparison and hyperparameter impact

### 🔍 Data Explorer
- Interactive data analysis and exploration
- Advanced filtering and search capabilities
- Statistical analysis and correlation studies
- Data export functionality

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone and navigate to the app directory:**
   ```bash
   cd "NU Research Task/app"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data (if needed):**
   ```bash
   cd ..
   python main.py --generate-data --num-students 100
   cd app
   ```

4. **Launch the application:**
   ```bash
   streamlit run main.py
   ```

### Alternative Launch Methods

**Linux/macOS:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Windows:**
```cmd
run_app.bat
```

The application will open automatically in your default web browser at `http://localhost:8501`.

## Application Structure

```
app/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── run_app.sh             # Linux/macOS launcher
├── run_app.bat            # Windows launcher
└── pages/                 # Page modules
    ├── __init__.py
    ├── home.py            # Home page
    ├── curriculum_graph.py # Curriculum visualization
    ├── student_analysis.py # Student cohort analysis
    ├── student_dashboard.py # Individual student profiles
    ├── course_recommendations.py # AI recommendations
    ├── training_metrics.py # ML model performance
    └── data_explorer.py    # Interactive data analysis
```

## Key Dependencies

- **Streamlit** (>=1.28.0) - Web application framework
- **Plotly** (>=5.15.0) - Interactive visualizations
- **Pandas** (>=1.5.3) - Data manipulation and analysis
- **NetworkX** (>=3.1) - Graph processing for curriculum structure
- **NumPy** (>=1.24.3) - Numerical computing
- **PyTorch** (>=2.0.1) - Deep learning (for RL components)

## Features Showcase

### Interactive Visualizations
- **Curriculum Graph**: Hierarchical course layout with prerequisite relationships
- **Student Analytics**: Distribution plots, correlation heatmaps, performance trends
- **Individual Dashboards**: Comprehensive 9-panel student analysis
- **Training Metrics**: Real-time ML model performance monitoring

### AI-Powered Recommendations
- **Reinforcement Learning**: DQN-based course selection optimization
- **Constraint Validation**: Academic rule compliance checking
- **Interest Alignment**: Personalized recommendations based on student preferences
- **Graduation Planning**: Timeline optimization and credit tracking

### Data Analysis Tools
- **Interactive Filtering**: Dynamic data exploration with multiple filter options
- **Statistical Analysis**: Correlation studies and performance metrics
- **Export Capabilities**: CSV downloads for external analysis
- **Search Functionality**: Course and student lookup capabilities

## Usage Examples

### Exploring the Curriculum
1. Navigate to "📊 Curriculum Graph"
2. Hover over nodes to see course details
3. Use the course explorer to search for specific courses
4. Filter by domain or difficulty level

### Analyzing Student Performance
1. Go to "👥 Student Analysis"
2. View cohort distributions and statistics
3. Explore interest patterns and correlations
4. Download cohort data for further analysis

### Getting Personalized Recommendations
1. Visit "💡 Course Recommendations"
2. Select a student profile
3. Choose recommendation strategy
4. Review AI-generated course suggestions
5. Validate against academic constraints

### Monitoring AI Training
1. Access "📈 Training Metrics"
2. View training progress and convergence
3. Analyze learning curves and performance
4. Compare different model configurations

## Technical Details

### Architecture
- **Multi-page Streamlit application** with modular design
- **Caching strategies** for optimal performance
- **Responsive layouts** with dynamic column arrangements
- **Custom CSS styling** for professional appearance

### Data Processing
- **Real-time data generation** for curriculum and student profiles
- **Graph algorithms** for prerequisite analysis
- **Statistical computations** for performance metrics
- **Interactive filtering** with immediate updates

### AI Components
- **Deep Q-Network (DQN)** reinforcement learning
- **Constraint satisfaction** algorithms
- **Interest-based recommendation** systems
- **Academic progression** modeling

## Customization

### Adding New Pages
1. Create a new Python file in the `pages/` directory
2. Implement a `show()` function with your page content
3. Add the import to `pages/__init__.py`
4. Update the navigation menu in `main.py`

### Modifying Visualizations
- All charts use Plotly for consistency and interactivity
- Color schemes follow the application theme
- Responsive design adapts to different screen sizes

### Extending Data Analysis
- Add new metrics to the data explorer
- Implement additional statistical analyses
- Create custom visualization types

## Performance Optimization

- **Streamlit caching** (`@st.cache_data`) for expensive operations
- **Efficient data structures** for large datasets
- **Lazy loading** of visualization components
- **Memory management** for long-running sessions

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
streamlit run main.py --server.port 8502
```

**Missing dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

**Data not found:**
```bash
cd ..
python main.py --generate-data
cd app
```

### Browser Compatibility
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues, questions, or suggestions:
- Check the troubleshooting section
- Review the code documentation
- Contact the development team

---

**AI Curriculum Planner** | Nile University Research Project | 2025
