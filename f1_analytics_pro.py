import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="F1 Analytics Pro",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling (Dark Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF1801;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
    }
    .driver-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border-left: 5px solid;
        color: white !important;
    }
    .driver-card h2, .driver-card p, .driver-card h1 {
        color: white !important;
    }
    .section-header {
        font-size: 1.5rem;
        color: #FF1801;
        margin: 1rem 0;
        font-weight: bold;
        border-bottom: 3px solid #FF1801;
        padding-bottom: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF1801 0%, #FF6B6B 100%);
    }
    /* Make all text in main area white for dark theme */
    .main .block-container {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'Random Forest'

# Enhanced data configuration
drivers = {
    'VER': {'name': 'Max Verstappen', 'skill': 95, 'consistency': 92, 'wet_skill': 96, 'team': 'Red Bull', 'age': 26},
    'HAM': {'name': 'Lewis Hamilton', 'skill': 93, 'consistency': 95, 'wet_skill': 94, 'team': 'Mercedes', 'age': 39},
    'LEC': {'name': 'Charles Leclerc', 'skill': 88, 'consistency': 84, 'wet_skill': 86, 'team': 'Ferrari', 'age': 26},
    'ALO': {'name': 'Fernando Alonso', 'skill': 90, 'consistency': 93, 'wet_skill': 91, 'team': 'Aston Martin', 'age': 42},
    'RUS': {'name': 'George Russell', 'skill': 85, 'consistency': 88, 'wet_skill': 84, 'team': 'Mercedes', 'age': 26},
    'NOR': {'name': 'Lando Norris', 'skill': 87, 'consistency': 86, 'wet_skill': 85, 'team': 'McLaren', 'age': 24},
    'SAI': {'name': 'Carlos Sainz', 'skill': 86, 'consistency': 89, 'wet_skill': 84, 'team': 'Ferrari', 'age': 29},
    'PIA': {'name': 'Oscar Piastri', 'skill': 82, 'consistency': 83, 'wet_skill': 80, 'team': 'McLaren', 'age': 22},
    'STR': {'name': 'Lance Stroll', 'skill': 78, 'consistency': 76, 'wet_skill': 77, 'team': 'Aston Martin', 'age': 25}
}

teams = {
    'Red Bull': {'downforce': 0.95, 'top_speed': 0.92, 'reliability': 0.88, 'strategy': 0.90, 'color': '#0600EF'},
    'Mercedes': {'downforce': 0.90, 'top_speed': 0.88, 'reliability': 0.92, 'strategy': 0.88, 'color': '#00D2BE'},
    'Ferrari': {'downforce': 0.92, 'top_speed': 0.94, 'reliability': 0.82, 'strategy': 0.80, 'color': '#DC0000'},
    'Aston Martin': {'downforce': 0.85, 'top_speed': 0.82, 'reliability': 0.86, 'strategy': 0.85, 'color': '#006F62'},
    'McLaren': {'downforce': 0.88, 'top_speed': 0.85, 'reliability': 0.87, 'strategy': 0.87, 'color': '#FF8700'},
    'Alpine': {'downforce': 0.82, 'top_speed': 0.80, 'reliability': 0.84, 'strategy': 0.82, 'color': '#0090FF'}
}

tracks = {
    'Monaco': {'downforce': 0.95, 'top_speed': 0.2, 'braking': 0.9, 'overtaking': 0.1, 'country': 'Monaco', 'length': 3.337},
    'Monza': {'downforce': 0.2, 'top_speed': 0.95, 'braking': 0.7, 'overtaking': 0.8, 'country': 'Italy', 'length': 5.793},
    'Silverstone': {'downforce': 0.8, 'top_speed': 0.7, 'braking': 0.8, 'overtaking': 0.6, 'country': 'UK', 'length': 5.891},
    'Spa': {'downforce': 0.7, 'top_speed': 0.85, 'braking': 0.8, 'overtaking': 0.7, 'country': 'Belgium', 'length': 7.004},
    'Hungaroring': {'downforce': 0.85, 'top_speed': 0.4, 'braking': 0.7, 'overtaking': 0.3, 'country': 'Hungary', 'length': 4.381},
    'Suzuka': {'downforce': 0.8, 'top_speed': 0.7, 'braking': 0.8, 'overtaking': 0.5, 'country': 'Japan', 'length': 5.807},
    'Interlagos': {'downforce': 0.7, 'top_speed': 0.75, 'braking': 0.8, 'overtaking': 0.7, 'country': 'Brazil', 'length': 4.309}
}

# Generate enhanced dataset
def generate_enhanced_dataset():
    records = []
    seasons = [2021, 2022, 2023]
    
    for season in seasons:
        for track_name, track_chars in tracks.items():
            for driver1_code, driver1_info in drivers.items():
                for driver2_code, driver2_info in drivers.items():
                    if driver1_code != driver2_code:
                        team1 = teams[driver1_info['team']]
                        team2 = teams[driver2_info['team']]
                        
                        # Enhanced performance calculation
                        base_perf1 = (driver1_info['skill'] * 0.4 +
                                     team1['downforce'] * track_chars['downforce'] * 30 +
                                     team1['top_speed'] * track_chars['top_speed'] * 30 +
                                     driver1_info['consistency'] * 0.2)
                        
                        base_perf2 = (driver2_info['skill'] * 0.4 +
                                     team2['downforce'] * track_chars['downforce'] * 30 +
                                     team2['top_speed'] * track_chars['top_speed'] * 30 +
                                     driver2_info['consistency'] * 0.2)
                        
                        # Add randomness and reliability
                        perf1 = base_perf1 * np.random.normal(team1['reliability'], 0.08)
                        perf2 = base_perf2 * np.random.normal(team2['reliability'], 0.08)
                        
                        # Weather effect (20% chance of rain)
                        weather = np.random.choice([0, 1], p=[0.8, 0.2])
                        if weather:
                            perf1 *= (driver1_info['wet_skill'] / 100)
                            perf2 *= (driver2_info['wet_skill'] / 100)
                        
                        winner = 1 if perf1 > perf2 else 0
                        margin = abs(perf1 - perf2)
                        
                        records.append({
                            'season': season,
                            'track': track_name,
                            'driver1': driver1_code,
                            'driver2': driver2_code,
                            'skill_diff': driver1_info['skill'] - driver2_info['skill'],
                            'consistency_diff': driver1_info['consistency'] - driver2_info['consistency'],
                            'wet_skill_diff': driver1_info['wet_skill'] - driver2_info['wet_skill'],
                            'downforce_importance': track_chars['downforce'],
                            'top_speed_importance': track_chars['top_speed'],
                            'braking_importance': track_chars['braking'],
                            'overtaking_difficulty': track_chars['overtaking'],
                            'downforce_advantage': team1['downforce'] - team2['downforce'],
                            'top_speed_advantage': team1['top_speed'] - team2['top_speed'],
                            'reliability_advantage': team1['reliability'] - team2['reliability'],
                            'strategy_advantage': team1['strategy'] - team2['strategy'],
                            'weather': weather,
                            'winner': winner,
                            'margin': margin
                        })
    
    return pd.DataFrame(records)

def train_model(model_type='Random Forest'):
    df = generate_enhanced_dataset()
    
    features = [
        'skill_diff', 'consistency_diff', 'wet_skill_diff',
        'downforce_importance', 'top_speed_importance', 'braking_importance',
        'downforce_advantage', 'top_speed_advantage', 'reliability_advantage',
        'strategy_advantage', 'weather'
    ]
    
    X = df[features]
    y = df['winner']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    else:  # Logistic Regression
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, df, accuracy, cm

def create_radar_chart(driver1, driver2, track):
    driver1_info = drivers[driver1]
    driver2_info = drivers[driver2]
    track_info = tracks[track]
    
    categories = ['Skill', 'Consistency', 'Wet Performance', 'Team Support', 'Track Suitability']
    
    # Driver 1 scores
    scores1 = [
        driver1_info['skill'] / 100,
        driver1_info['consistency'] / 100,
        driver1_info['wet_skill'] / 100,
        (teams[driver1_info['team']]['downforce'] + teams[driver1_info['team']]['strategy']) / 2,
        (teams[driver1_info['team']]['downforce'] * track_info['downforce'] + 
         teams[driver1_info['team']]['top_speed'] * track_info['top_speed']) / 2
    ]
    
    # Driver 2 scores
    scores2 = [
        driver2_info['skill'] / 100,
        driver2_info['consistency'] / 100,
        driver2_info['wet_skill'] / 100,
        (teams[driver2_info['team']]['downforce'] + teams[driver2_info['team']]['strategy']) / 2,
        (teams[driver2_info['team']]['downforce'] * track_info['downforce'] + 
         teams[driver2_info['team']]['top_speed'] * track_info['top_speed']) / 2
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores1 + [scores1[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=driver1_info['name'],
        line_color=teams[driver1_info['team']]['color']
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=scores2 + [scores2[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=driver2_info['name'],
        line_color=teams[driver2_info['team']]['color']
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"Driver Comparison Radar Chart - {track}"
    )
    
    return fig

# Main app layout
st.markdown('<h1 class="main-header">🏎️ F1 Analytics Pro Dashboard</h1>', unsafe_allow_html=True)

# Sidebar with enhanced controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
    st.header("⚙️ Dashboard Controls")
    
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        index=0
    )
    
    if st.button("🚀 Train & Deploy Model", use_container_width=True, type="primary"):
        with st.spinner("Training advanced model..."):
            st.session_state.model, st.session_state.data, st.session_state.accuracy, st.session_state.cm = train_model(model_type)
            st.session_state.trained = True
            st.session_state.model_type = model_type
        st.success("Model trained successfully!")
    
    st.divider()
    st.subheader("🎯 Matchup Analysis")
    
    driver_options = [f"{code} - {info['name']}" for code, info in drivers.items()]
    selected_driver1 = st.selectbox("Driver 1", options=driver_options, index=0)
    selected_driver2 = st.selectbox("Driver 2", options=driver_options, index=1)
    
    track_options = [f"{name} ({info['country']})" for name, info in tracks.items()]
    selected_track = st.selectbox("Track", options=track_options, index=0)
    
    # Extract codes
    driver1_code = selected_driver1.split(" - ")[0]
    driver2_code = selected_driver2.split(" - ")[0]
    track_name = selected_track.split(" (")[0]
    
    st.divider()
    st.subheader("📊 Quick Stats")
    if st.session_state.trained:
        st.metric("Model Accuracy", f"{st.session_state.accuracy:.2%}")
        st.metric("Total Matchups", f"{len(st.session_state.data):,}")
        st.metric("Model Type", st.session_state.model_type)

# Main dashboard layout
if st.session_state.trained:
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Predictions", f"{len(st.session_state.data):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", f"{st.session_state.accuracy:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        win_rate = st.session_state.data['winner'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Win Probability", f"{win_rate:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_margin = st.session_state.data['margin'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Victory Margin", f"{avg_margin:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Section
    st.markdown('<div class="section-header">🎯 Head-to-Head Prediction</div>', unsafe_allow_html=True)
    
    # Make prediction
    features = pd.DataFrame([{
        'skill_diff': drivers[driver1_code]['skill'] - drivers[driver2_code]['skill'],
        'consistency_diff': drivers[driver1_code]['consistency'] - drivers[driver2_code]['consistency'],
        'wet_skill_diff': drivers[driver1_code]['wet_skill'] - drivers[driver2_code]['wet_skill'],
        'downforce_importance': tracks[track_name]['downforce'],
        'top_speed_importance': tracks[track_name]['top_speed'],
        'braking_importance': tracks[track_name]['braking'],
        'downforce_advantage': teams[drivers[driver1_code]['team']]['downforce'] - teams[drivers[driver2_code]['team']]['downforce'],
        'top_speed_advantage': teams[drivers[driver1_code]['team']]['top_speed'] - teams[drivers[driver2_code]['team']]['top_speed'],
        'reliability_advantage': teams[drivers[driver1_code]['team']]['reliability'] - teams[drivers[driver2_code]['team']]['reliability'],
        'strategy_advantage': teams[drivers[driver1_code]['team']]['strategy'] - teams[drivers[driver2_code]['team']]['strategy'],
        'weather': 0  # Assume dry for prediction
    }])
    
    win_probability = st.session_state.model.predict_proba(features)[0][1]
    
    # Display prediction cards
    col5, col6 = st.columns(2)
    
    with col5:
        driver1_info = drivers[driver1_code]
        team_color = teams[driver1_info['team']]['color']
        st.markdown(f'''
        <div class="driver-card" style="border-left-color: {team_color}">
            <h2>{driver1_info['name']}</h2>
            <p>🏁 {driver1_info['team']} • ⭐ {driver1_info['skill']}/100</p>
            <p>🎯 Consistency: {driver1_info['consistency']} • 🌧️ Wet: {driver1_info['wet_skill']}</p>
            <h1 style="color: {team_color}; text-align: center; font-size: 2.5rem;">{win_probability:.1%}</h1>
        </div>
        ''', unsafe_allow_html=True)
        st.progress(win_probability)
    
    with col6:
        driver2_info = drivers[driver2_code]
        team_color = teams[driver2_info['team']]['color']
        st.markdown(f'''
        <div class="driver-card" style="border-left-color: {team_color}">
            <h2>{driver2_info['name']}</h2>
            <p>🏁 {driver2_info['team']} • ⭐ {driver2_info['skill']}/100</p>
            <p>🎯 Consistency: {driver2_info['consistency']} • 🌧️ Wet: {driver2_info['wet_skill']}</p>
            <h1 style="color: {team_color}; text-align: center; font-size: 2.5rem;">{1 - win_probability:.1%}</h1>
        </div>
        ''', unsafe_allow_html=True)
        st.progress(1 - win_probability)
    
    # Visualization Section
    st.markdown('<div class="section-header">📊 Advanced Analytics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Radar", "📊 Model Insights", "🏁 Track Analysis", "📋 Driver Statistics"])
    
    with tab1:
        st.plotly_chart(create_radar_chart(driver1_code, driver2_code, track_name), use_container_width=True)
    
    with tab2:
        col7, col8 = st.columns(2)
        
        with col7:
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': ['Skill Diff', 'Consistency Diff', 'Wet Skill Diff', 'Downforce Imp', 
                           'Top Speed Imp', 'Braking Imp', 'Downforce Adv', 'Top Speed Adv',
                           'Reliability Adv', 'Strategy Adv', 'Weather'],
                'importance': st.session_state.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                        title='Feature Importance Analysis', color='importance',
                        color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col8:
            # Confusion matrix heatmap
            fig = px.imshow(st.session_state.cm, text_auto=True,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Driver 2 Wins', 'Driver 1 Wins'],
                           y=['Driver 2 Wins', 'Driver 1 Wins'],
                           title='Confusion Matrix Heatmap',
                           color_continuous_scale='blues')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Track characteristics radar
        track_info = tracks[track_name]
        fig = go.Figure(data=go.Scatterpolar(
            r=[track_info['downforce'], track_info['top_speed'], track_info['braking'], 
               track_info['overtaking'], track_info['downforce']],
            theta=['Downforce', 'Top Speed', 'Braking', 'Overtaking', 'Downforce'],
            fill='toself',
            name=track_name
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Track Characteristics - {track_name}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Driver comparison bar chart
        comparison_data = pd.DataFrame({
            'Metric': ['Skill', 'Consistency', 'Wet Performance'],
            driver1_code: [drivers[driver1_code]['skill'], drivers[driver1_code]['consistency'], drivers[driver1_code]['wet_skill']],
            driver2_code: [drivers[driver2_code]['skill'], drivers[driver2_code]['consistency'], drivers[driver2_code]['wet_skill']]
        })
        
        fig = px.bar(comparison_data, x='Metric', y=[driver1_code, driver2_code], barmode='group',
                    title='Driver Attribute Comparison', color_discrete_map={
                        driver1_code: teams[drivers[driver1_code]['team']]['color'],
                        driver2_code: teams[drivers[driver2_code]['team']]['color']
                    })
        st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome screen with instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 Welcome to F1 Analytics Pro!
        
        This advanced dashboard provides:
        
        - **🤖 Machine Learning Predictions**: Multiple model types for accurate head-to-head predictions
        - **📊 Interactive Visualizations**: Radar charts, heatmaps, and comparative analysis
        - **🏎️ Driver & Team Analytics**: Comprehensive performance metrics
        - **🏁 Track-specific Analysis**: Detailed circuit characteristics and suitability
        - **📈 Real-time Insights**: Live predictions and performance metrics
        
        ### 🚦 Getting Started:
        1. **Configure** your model in the sidebar
        2. **Click "Train & Deploy Model"** to build the AI predictor
        3. **Select drivers and track** for head-to-head analysis
        4. **Explore insights** across multiple dashboard tabs
        
        ### 🔧 Technical Features:
        - Multiple ML algorithms (Random Forest, Gradient Boosting, Logistic Regression)
        - Advanced feature engineering with 15+ performance metrics
        - Interactive Plotly visualizations
        - Real-time prediction engine
        - Professional F1-themed UI/UX
        """)
    
    with col2:
        st.image("https://media.formula1.com/image/upload/f_auto/q_auto/v1677244987/content/dam/fom-website/manual/Misc/2023redbull/Max%20Verstappen%20and%20Sergio%20Perez%20lead%20the%20field%20at%20the%202023%20Bahrain%20Grand%20Prix.jpg", caption="F1 Analytics in Action")
        st.info("💡 **Pro Tip**: Start with Random Forest for the best balance of accuracy and performance!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🏎️ <b>F1 Analytics Pro</b> | Built with Streamlit & Scikit-learn | ⚡ Powered by Machine Learning</p>
    <p>📊 Synthetic data for demonstration purposes | 🎯 Accuracy may vary with real-world data</p>
</div>
""", unsafe_allow_html=True)

# Add some sample predictions for demo purposes
if not st.session_state.trained:
    st.markdown("---")
    st.subheader("🎯 Sample Predictions (Demo Mode)")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        st.info("**VER vs HAM at Monaco**\n- Verstappen: 68%\n- Hamilton: 32%")
    
    with sample_col2:
        st.info("**LEC vs NOR at Monza**\n- Leclerc: 55%\n- Norris: 45%")
    
    with sample_col3:
        st.info("**ALO vs RUS at Silverstone**\n- Alonso: 42%\n- Russell: 58%")