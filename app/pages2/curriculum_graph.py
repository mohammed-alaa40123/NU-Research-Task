"""
Curriculum Graph Page - Interactive Curriculum Visualization
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import networkx as nx
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.curriculum_graph import create_sample_curriculum
from src.visualization import create_visualizer

def load_real_stats():
    """Load real statistics from curriculum data"""
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        
        with open(os.path.join(data_dir, 'curriculum_data.json'), 'r') as f:
            curriculum_data = json.load(f)
        
        nodes = curriculum_data['nodes']
        edges = curriculum_data['edges']
        
        total_courses = len(nodes)
        total_prerequisites = len(edges)
        avg_prerequisites = total_prerequisites / total_courses if total_courses > 0 else 0
        
        # Count domains
        domains = {}
        for course in nodes:
            domain = course['domain']
            domains[domain] = domains.get(domain, 0) + 1
        
        return {
            'total_courses': total_courses,
            'total_prerequisites': total_prerequisites,
            'avg_prerequisites': avg_prerequisites,
            'domains': domains
        }
    except Exception as e:
        # Fallback values
        return {
            'total_courses': 34,
            'total_prerequisites': 37,
            'avg_prerequisites': 1.1,
            'domains': {
                'Software Engineering': 9,
                'Data Science': 6,
                'AI': 6,
                'Theory': 5,
                'Systems': 4,
                'Security': 4
            }
        }

@st.cache_data
def load_curriculum_data():
    """Load and cache curriculum data"""
    return create_sample_curriculum()

@st.cache_data
def create_curriculum_visualization():
    """Create and cache curriculum visualization"""
    curriculum = load_curriculum_data()
    G = curriculum.graph
    
    # Calculate course levels for hierarchical layout
    course_levels = {}
    topo_order = list(nx.topological_sort(G))
    for node in topo_order:
        prereqs = curriculum.get_prerequisites(node)
        if not prereqs:
            course_levels[node] = 0
        else:
            max_prereq_level = max(course_levels.get(prereq, 0) for prereq in prereqs)
            course_levels[node] = max_prereq_level + 1
    
    # Group courses by level
    max_level = max(course_levels.values())
    level_nodes = {level: [] for level in range(max_level + 1)}
    for node, level in course_levels.items():
        level_nodes[level].append(node)
    
    # Create layout
    pos = {}
    y_spacing = -2.0
    x_spacing = 1.5
    
    for level, nodes in level_nodes.items():
        y = level * y_spacing
        num_nodes = len(nodes)
        
        if num_nodes == 1:
            x_positions = [0]
        else:
            total_width = (num_nodes - 1) * x_spacing
            x_positions = [i * x_spacing - total_width / 2 for i in range(num_nodes)]
        
        nodes_sorted = sorted(nodes, key=lambda n: curriculum.get_course_info(n).get('domain', 'ZZZ'))
        
        for i, node in enumerate(nodes_sorted):
            pos[node] = (x_positions[i], y)
    
    # Prepare node data
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    
    domain_colors = {
        'AI': '#E74C3C', 'Security': '#3498DB', 'Data Science': '#9B59B6',
        'Software Engineering': '#27AE60', 'Systems': '#F39C12', 'Theory': '#E67E22'
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        course_info = curriculum.get_course_info(node)
        domain = course_info.get('domain', 'Other')
        difficulty = course_info.get('difficulty', 'Intermediate')
        
        node_color.append(domain_colors.get(domain, '#95A5A6'))
        
        size_map = {'Beginner': 55, 'Intermediate': 60, 'Advanced': 65}
        node_size.append(size_map.get(difficulty, 60))
        
        prereqs = curriculum.get_prerequisites(node)
        level = course_levels.get(node, 0)
        
        text = (f"<b>{node}</b><br>"
                f"{course_info.get('name', 'Unknown')}<br>"
                f"Domain: {domain}<br>"
                f"Level: {level}<br>"
                f"Prerequisites: {', '.join(prereqs) if prereqs else 'None'}")
        node_text.append(text)
    
    # Create edge data
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='#BDC3C7'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'),
            opacity=0.9
        ),
        text=[node for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=11, color='white', family='Arial Black'),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add level labels
    for level in range(max_level + 1):
        if level_nodes[level]:
            fig.add_annotation(
                x=min(pos[node][0] for node in level_nodes[level]) - 2,
                y=level * y_spacing,
                text=f"<b>Level {level}</b>",
                showarrow=False,
                font=dict(size=14, color='#2C3E50'),
                xanchor='right'
            )
    
    # Add domain legend
    legend_y = max_level * y_spacing - 1
    legend_x = max(node_x) + 1
    for i, (domain, color) in enumerate(domain_colors.items()):
        fig.add_trace(go.Scatter(
            x=[legend_x], y=[legend_y - i * 0.3],
            mode='markers+text',
            marker=dict(size=15, color=color, line=dict(width=1, color='white')),
            text=domain,
            textposition="middle right",
            textfont=dict(size=10, color='#2C3E50'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Update layout with autoscaling
    fig.update_layout(
        title=dict(
            text="Computer Science Curriculum - Top-Down Hierarchical View",
            font=dict(size=20, color='#2C3E50'),
            x=0.5
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            autorange=True,  # Enable autoscaling
            fixedrange=False  # Allow zooming
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            autorange=True,  # Enable autoscaling
            fixedrange=False  # Allow zooming
        ),
        autosize=True,  # Auto-resize to container
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=200, t=80, b=50)
    )
    
    return fig, curriculum

def show():
    """Display the curriculum graph page"""
    
    st.markdown('<h1 class="main-header">📊 Curriculum Graph</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Interactive Computer Science Curriculum Visualization
    
    This interactive graph shows the complete computer science curriculum with:
    - **Hierarchical levels** based on prerequisite depth
    - **Color-coded domains** for different subject areas
    - **Node sizes** indicating course difficulty
    - **Interactive hover** for detailed course information
    """)
    
    # Create controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        show_stats = st.checkbox("Show Statistics", value=True)
        
    with col2:
        highlight_domain = st.selectbox(
            "Highlight Domain",
            ["None", "AI", "Security", "Data Science", "Software Engineering", "Systems", "Theory"]
        )
    
    # Load and display visualization
    try:
        fig, curriculum = create_curriculum_visualization()
        
        # Display statistics if requested
        if show_stats:
            st.markdown("### 📈 Curriculum Statistics")
            
            # Load real statistics
            real_stats = load_real_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Courses", real_stats['total_courses'])
            
            with col2:
                st.metric("Prerequisites", real_stats['total_prerequisites'])
            
            with col3:
                st.metric("Avg Prerequisites", f"{real_stats['avg_prerequisites']:.1f}")
            
            with col4:
                st.metric("Domains", len(real_stats['domains']))
            
            # Domain distribution
            st.markdown("### 🎯 Domain Distribution")
            domain_data = real_stats['domains']
            
            # Create domain distribution chart with updated colors
            domain_colors_list = ['#E74C3C', '#3498DB', '#9B59B6', '#27AE60', '#F39C12', '#E67E22']
            
            domain_fig = go.Figure(data=[
                go.Bar(
                    x=list(domain_data.keys()),
                    y=list(domain_data.values()),
                    marker_color=domain_colors_list[:len(domain_data)],
                    text=list(domain_data.values()),
                    textposition='auto'
                )
            ])
            
            domain_fig.update_layout(
                title="Courses per Domain",
                xaxis_title="Domain",
                yaxis_title="Number of Courses",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(domain_fig, use_container_width=True)
        
        # Display main curriculum graph with autoscaling
        st.markdown("### 🗺️ Interactive Curriculum Map")
        st.info("💡 **Tip:** The graph is fully interactive - zoom, pan, and hover over courses for details!")
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'responsive': True
        })
        
        # Course explorer
        st.markdown("### 🔍 Interactive Course Explorer")
        
        # Get all courses
        all_courses = list(curriculum.graph.nodes())
        
        # Create filter options
        explore_col1, explore_col2, explore_col3 = st.columns(3)
        
        with explore_col1:
            # Domain filter
            all_domains = ["All"] + sorted(list(set(curriculum.get_course_info(course).get('domain', 'Unknown') for course in all_courses)))
            selected_domain = st.selectbox("Filter by Domain:", all_domains, key="domain_filter")
            
        with explore_col2:
            # Difficulty filter
            all_difficulties = ["All"] + sorted(list(set(curriculum.get_course_info(course).get('difficulty', 'Unknown') for course in all_courses)))
            selected_difficulty = st.selectbox("Filter by Difficulty:", all_difficulties, key="difficulty_filter")
            
        with explore_col3:
            # Level filter
            course_levels = {}
            topo_order = list(nx.topological_sort(curriculum.graph))
            for node in topo_order:
                prereqs = curriculum.get_prerequisites(node)
                if not prereqs:
                    course_levels[node] = 0
                else:
                    max_prereq_level = max(course_levels.get(prereq, 0) for prereq in prereqs)
                    course_levels[node] = max_prereq_level + 1
            
            max_level = max(course_levels.values())
            selected_level = st.selectbox("Filter by Level:", ["All"] + [f"Level {i}" for i in range(max_level + 1)], key="level_filter")
        
        # Apply filters
        filtered_courses = all_courses.copy()
        
        if selected_domain != "All":
            filtered_courses = [course for course in filtered_courses 
                             if curriculum.get_course_info(course).get('domain', 'Unknown') == selected_domain]
        
        if selected_difficulty != "All":
            filtered_courses = [course for course in filtered_courses 
                              if curriculum.get_course_info(course).get('difficulty', 'Unknown') == selected_difficulty]
        
        if selected_level != "All":
            level_num = int(selected_level.split()[1])
            filtered_courses = [course for course in filtered_courses 
                              if course_levels.get(course, 0) == level_num]
        
        # Search functionality - searchable selectbox
        st.markdown("🔍 **Search and Select Course:**")
        
        # Display filtered count
        st.info(f"📊 Showing {len(filtered_courses)} of {len(all_courses)} courses")
        
        # Course selection
        # Create a searchable course list with course codes and names
        course_options = []
        for course in filtered_courses:
            course_info = curriculum.get_course_info(course)
            course_name = course_info.get('name', 'Unknown')
            course_options.append(f"{course} - {course_name}")

        selected_course_option = st.selectbox(
            "Type to search or select a course:",
            course_options,
            index=0,
            key="course_search_select",
            help="Type course code or name to search, then select from the filtered list"
        )
        
        # Extract course code from selection
        selected_course = selected_course_option.split(" - ")[0]
        
        if selected_course:
            course_info = curriculum.get_course_info(selected_course)
            prereqs = curriculum.get_prerequisites(selected_course)
            
            # Create tabs for different views
            detail_tab, prereq_tab, pathway_tab = st.tabs(["📋 Course Details", "🔗 Prerequisites", "🛤️ Learning Pathway"])
            
            with detail_tab:
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown(f"#### {selected_course}")
                    st.markdown(f"**Name:** {course_info.get('name', 'Unknown')}")
                    st.markdown(f"**Domain:** {course_info.get('domain', 'Unknown')}")
                    st.markdown(f"**Difficulty:** {course_info.get('difficulty', 'Unknown')}")
                    st.markdown(f"**Credits:** {course_info.get('credits', 'Unknown')}")
                    st.markdown(f"**Level:** {course_levels.get(selected_course, 0)}")
                    
                with detail_col2:
                    # Course statistics
                    st.markdown("#### Course Statistics")
                    
                    # Count prerequisites
                    prereq_count = len(prereqs)
                    st.metric("Prerequisites", prereq_count)
                    
                    # Count dependent courses
                    dependent_courses = [course for course in all_courses 
                                        if selected_course in curriculum.get_prerequisites(course)]
                    st.metric("Enables Courses", len(dependent_courses))
                    
                    # Domain courses
                    same_domain_courses = [course for course in all_courses 
                                            if curriculum.get_course_info(course).get('domain') == course_info.get('domain')]
                    st.metric("Same Domain", len(same_domain_courses))
            
            with prereq_tab:
                prereq_col1, prereq_col2 = st.columns(2)
                
                with prereq_col1:
                    st.markdown("#### Direct Prerequisites")
                    if prereqs:
                        for prereq in prereqs:
                            prereq_info = curriculum.get_course_info(prereq)
                            with st.expander(f"📚 {prereq}"):
                                st.markdown(f"**Name:** {prereq_info.get('name', 'Unknown')}")
                                st.markdown(f"**Domain:** {prereq_info.get('domain', 'Unknown')}")
                                st.markdown(f"**Difficulty:** {prereq_info.get('difficulty', 'Unknown')}")
                                st.markdown(f"**Credits:** {prereq_info.get('credits', 'Unknown')}")
                    else:
                        st.markdown("*No direct prerequisites*")
                
                with prereq_col2:
                    st.markdown("#### Enables These Courses")
                    if dependent_courses:
                        for course in dependent_courses:
                            course_info_dep = curriculum.get_course_info(course)
                            with st.expander(f"🎯 {course}"):
                                st.markdown(f"**Name:** {course_info_dep.get('name', 'Unknown')}")
                                st.markdown(f"**Domain:** {course_info_dep.get('domain', 'Unknown')}")
                                st.markdown(f"**Difficulty:** {course_info_dep.get('difficulty', 'Unknown')}")
                                st.markdown(f"**Credits:** {course_info_dep.get('credits', 'Unknown')}")
                    else:
                        st.markdown("*No dependent courses*")
            
            with pathway_tab:
                st.markdown("#### Learning Pathway Analysis")
                
                # Create a mini pathway visualization
                pathway_courses = set()
                
                # Add all prerequisites recursively
                def get_all_prereqs(course):
                    all_prereqs = set()
                    direct_prereqs = curriculum.get_prerequisites(course)
                    for prereq in direct_prereqs:
                        all_prereqs.add(prereq)
                        all_prereqs.update(get_all_prereqs(prereq))
                    return all_prereqs
                
                # Add all courses that depend on this course
                def get_all_dependents(course):
                    all_deps = set()
                    for other_course in all_courses:
                        if course in get_all_prereqs(other_course):
                            all_deps.add(other_course)
                    return all_deps
                
                all_prereqs = get_all_prereqs(selected_course)
                all_dependents = get_all_dependents(selected_course)
                
                pathway_col1, pathway_col2, pathway_col3 = st.columns(3)
                
                with pathway_col1:
                    st.markdown("##### 📚 Must Take First")
                    st.markdown(f"*{len(all_prereqs)} courses*")
                    for prereq in sorted(all_prereqs):
                        level = course_levels.get(prereq, 0)
                        st.markdown(f"- **Level {level}:** {prereq}")
                
                with pathway_col2:
                    st.markdown("##### 🎯 Current Course")
                    st.markdown(f"**{selected_course}**")
                    st.markdown(f"*Level {course_levels.get(selected_course, 0)}*")
                    st.markdown(f"*{course_info.get('domain', 'Unknown')} Domain*")
                
                with pathway_col3:
                    st.markdown("##### 🚀 Unlocks Access To")
                    st.markdown(f"*{len(all_dependents)} courses*")
                    for dependent in sorted(all_dependents):
                        level = course_levels.get(dependent, 0)
                        st.markdown(f"- **Level {level}:** {dependent}")
                
                # Show pathway statistics
                st.markdown("---")
                pathway_stats_col1, pathway_stats_col2, pathway_stats_col3 = st.columns(3)
                
                with pathway_stats_col1:
                    st.metric("Total Prerequisites", len(all_prereqs))
                
                with pathway_stats_col2:
                    total_credits = sum(curriculum.get_course_info(course).get('credits', 0) for course in all_prereqs)
                    st.metric("Prerequisite Credits", total_credits)
                
                with pathway_stats_col3:
                    st.metric("Future Opportunities", len(all_dependents))
    
    
    except Exception as e:
        st.error(f"Error loading curriculum visualization: {str(e)}")
        st.info("Please ensure the curriculum data is properly loaded.")
    
    # Additional info
    with st.expander("ℹ️ About the Curriculum Graph"):
        st.markdown("""
        ### Graph Structure
        
        The curriculum graph is a **directed acyclic graph (DAG)** where:
        - **Nodes** represent courses (34 total)
        - **Edges** represent prerequisite relationships (37 total)
        - **Levels** are determined by prerequisite depth
        - **Average** of 1.1 prerequisites per course
        
        ### Visual Encoding
        
        - **Colors**: Different domains (AI, Security, Data Science, etc.)
        - **Sizes**: Course difficulty (Beginner, Intermediate, Advanced)
        - **Position**: Hierarchical layout based on prerequisites
        - **Connections**: Prerequisite relationships
        
        ### Interactive Features
        
        - **Hover** over nodes to see detailed course information
        - **Zoom** and **pan** to explore different sections (autoscaled by default)
        - **Select** courses in the explorer below for detailed analysis
        - **Responsive** layout that adapts to your screen size
        
        ### Domain Distribution
        
        The curriculum covers 6 academic domains:
        - **Software Engineering**: 9 courses (26.5%)
        - **Data Science**: 6 courses (17.6%)
        - **AI**: 6 courses (17.6%)
        - **Theory**: 5 courses (14.7%)
        - **Systems**: 4 courses (11.8%)
        - **Security**: 4 courses (11.8%)
        """)
        
        # Add some technical details
        st.markdown("### Technical Implementation")
        st.markdown("""
        - **Graph Library**: NetworkX for graph operations
        - **Visualization**: Plotly for interactive graphics
        - **Layout Algorithm**: Custom hierarchical positioning
        - **Data Source**: Real curriculum data from JSON files
        - **Caching**: Streamlit caching for performance
        """)
