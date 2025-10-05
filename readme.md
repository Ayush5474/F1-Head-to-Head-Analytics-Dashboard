## 🚀 Features  

### 🔮 Machine Learning Engine  
- Supports **multiple ML algorithms**:  
  - Random Forest (default, best accuracy)  
  - Gradient Boosting  
  - Logistic Regression  
- Trains on **synthetic multi-season datasets** that factor in:  
  - Driver skill, consistency, and wet-weather ability  
  - Team strengths (downforce, top speed, reliability, strategy)  
  - Track characteristics (braking, overtaking difficulty, layout)  
  - Weather effects (20% rain chance)  

### 📊 Interactive Dashboard  
- **Head-to-Head Predictions** – Select two drivers and a track → see real-time win probabilities.  
- **Performance Radar Charts** – Compare drivers across skill, consistency, wet-weather ability, team support, and track suitability.  
- **Model Insights** – Feature importance bar charts & confusion matrix heatmaps.  
- **Track Analysis** – Radar chart showing each circuit’s unique profile.  
- **Driver Statistics** – Attribute comparisons in grouped bar charts.  

### 🎨 Professional UI/UX  
- Modern **dark theme** with custom CSS.  
- Styled **driver cards** with team colors, skill stats, and prediction percentages.  
- **Metric cards** for quick insights (accuracy, total matchups, win rate, margin).  
- Tabbed interface for clean organization of visualizations.  

---

## 🧠 How It Works  

1. **Synthetic Dataset Generation**  
   - Simulates every possible driver matchup across multiple seasons & tracks.  
   - Performance is calculated using driver ability, team strengths, track characteristics, and weather randomness.  

2. **Model Training**  
   - Dataset is split into training/testing sets.  
   - Models are trained to classify the **winner** of each matchup.  
   - Outputs accuracy scores and confusion matrix.  

3. **Prediction & Analysis**  
   - User selects drivers + track → app predicts head-to-head outcome with win probabilities.  
   - Visual analytics explain why predictions happen (feature importance, radar plots).  

---

## 🛠️ Tech Stack  

- **Frontend/UI**: Streamlit + Custom CSS (Dark Theme)  
- **Backend/ML**: Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)  
- **Data Simulation**: Python (Numpy, Pandas) for synthetic dataset generation  
- **Visualization**: Plotly (interactive) + Matplotlib + Seaborn  

---

## 📌 Notes  

- ⚠️ All data is **synthetic** (simulated for demonstration/educational purposes).  
- 📊 Accuracy will differ on **real F1 datasets**.  
- 🎯 The project demonstrates how **machine learning + sports analytics** can be integrated into an interactive dashboard.  

---

## 🖼️ Screenshots  

(Add screenshots of your Streamlit app here, for example:)  

