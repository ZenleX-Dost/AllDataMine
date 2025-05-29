import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import os
from stqdm import stqdm

# Set page config (removed theme parameter for compatibility)
st.set_page_config(page_title="Turbofan Engine RUL Prediction", layout="wide")

# Custom CSS for dark theme, enhanced to override Streamlit defaults
st.markdown("""
<style>
    .main {
        background-color: #2b2b2b;
        padding: 20px;
    }
    .stButton>button {
        background-color: #1e90ff;
        color: #ffffff;
        border-radius: 5px;
        padding: 8px 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #104e8b;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #444444;
        color: #d3d3d3;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1e90ff;
        color: #ffffff;
    }
    .card {
        background-color: #3c3c3c;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    p, li, .stMarkdown, .stText {
        color: #d3d3d3;
    }
    .sidebar .sidebar-content {
        background-color: #2b2b2b;
        padding: 15px;
    }
    .stSelectbox, .stSlider, .stCheckbox, .stMultiselect {
        color: #d3d3d3;
    }
    .stSelectbox > div > div, .stMultiselect > div > div {
        background-color: #3c3c3c;
        color: #d3d3d3;
    }
    .stSpinner .spinner {
        border-top-color: #1e90ff;
    }
    .stProgress > div > div {
        background-color: #1e90ff;
    }
    /* Override Streamlit dataframe and table styles */
    .stDataFrame, .stTable {
        color: #d3d3d3;
        background-color: #3c3c3c;
        border: 1px solid #555555;
    }
    .stDataFrame th, .stTable th {
        background-color: #444444;
        color: #ffffff;
    }
    .stDataFrame td, .stTable td {
        background-color: #3c3c3c;
        color: #d3d3d3;
    }
    /* Override metric styles */
    .stMetric, .stMetric > div {
        background-color: #3c3c3c;
        color: #d3d3d3;
    }
    .stMetric label {
        color: #d3d3d3;
    }
    .stMetric .metric-value {
        color: #ffffff;
    }
    /* Override warning/info messages */
    .stAlert {
        background-color: #444444;
        color: #d3d3d3;
        border: 1px solid #555555;
    }
    /* Override expander styles */
    .stExpander {
        background-color: #3c3c3c;
        color: #d3d3d3;
    }
    .stExpander > div > div {
        background-color: #444444;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Custom scoring function
def custom_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 10) - 1, np.exp(d / 13) - 1))

# Cache data loading
@st.cache_data
def load_data(dataset):
    columns = ['unit_number', 'time_cycles'] + \
              [f'setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    for file in [f'train_{dataset}.txt', f'test_{dataset}.txt', f'RUL_{dataset}.txt']:
        if not os.path.exists(file):
            return None, None, None, f"File {file} not found"
    
    train_data = pd.read_csv(f'train_{dataset}.txt', sep='\s+', header=None, names=columns)
    test_data = pd.read_csv(f'test_{dataset}.txt', sep='\s+', header=None, names=columns)
    true_rul = pd.read_csv(f'RUL_{dataset}.txt', header=None, names=['true_rul'])
    
    train_data = train_data[columns]
    test_data = test_data[columns]
    
    return train_data, test_data, true_rul, None

# Cache preprocessing
@st.cache_data
def preprocess_data(train_data, test_data, dataset):
    train_data['max_cycle'] = train_data.groupby('unit_number')['time_cycles'].transform('max')
    train_data['RUL'] = train_data['max_cycle'] - train_data['time_cycles']
    
    scale_cols = [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    scaler = MinMaxScaler()
    train_data[scale_cols] = scaler.fit_transform(train_data[scale_cols])
    test_data[scale_cols] = scaler.transform(test_data[scale_cols])
    
    if dataset in ['FD002', 'FD004']:
        setting_cols = [f'setting_{i}' for i in range(1, 4)]
        kmeans = KMeans(n_clusters=6, random_state=42)
        train_data['op_condition'] = kmeans.fit_predict(train_data[setting_cols])
        test_data['op_condition'] = kmeans.predict(test_data[setting_cols])
    else:
        train_data['op_condition'] = 0
        test_data['op_condition'] = 0
    
    return train_data, test_data

# Cache feature engineering
@st.cache_data
def engineer_features(train_data, test_data, dataset, use_top_features):
    window = 5
    sensors = [f'sensor_{i}' for i in range(1, 22)]
    
    if use_top_features:
        correlations = train_data[[f'sensor_{i}' for i in range(1, 22)] + ['RUL']].corr()
        top_sensors = correlations['RUL'].abs().sort_values(ascending=False).head(5).index.tolist()
        sensors = [s for s in sensors if s in top_sensors]
    
    for sensor in sensors:
        train_data[f'{sensor}_roll_mean'] = train_data.groupby(['unit_number', 'op_condition'])[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()).fillna(train_data[sensor])
        train_data[f'{sensor}_roll_std'] = train_data.groupby(['unit_number', 'op_condition'])[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()).fillna(0)
        test_data[f'{sensor}_roll_mean'] = test_data.groupby(['unit_number', 'op_condition'])[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()).fillna(test_data[sensor])
        test_data[f'{sensor}_roll_std'] = test_data.groupby(['unit_number', 'op_condition'])[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()).fillna(0)
        
        train_data[f'{sensor}_lag1'] = train_data.groupby(['unit_number', 'op_condition'])[sensor].shift(1).fillna(train_data[sensor])
        train_data[f'{sensor}_lag2'] = train_data.groupby(['unit_number', 'op_condition'])[sensor].shift(2).fillna(train_data[sensor])
        test_data[f'{sensor}_lag1'] = test_data.groupby(['unit_number', 'op_condition'])[sensor].shift(1).fillna(test_data[sensor])
        test_data[f'{sensor}_lag2'] = test_data.groupby(['unit_number', 'op_condition'])[sensor].shift(2).fillna(test_data[sensor])
    
    scale_cols = [f'setting_{i}' for i in range(1, 4)] + sensors
    feature_cols = scale_cols + ['op_condition'] + \
                   [f'{s}_roll_mean' for s in sensors] + \
                   [f'{s}_roll_std' for s in sensors] + \
                   [f'{s}_lag1' for s in sensors] + \
                   [f'{s}_lag2' for s in sensors]
    
    return train_data, test_data, feature_cols

# Cache model training
@st.cache_resource
def train_models(train_data, test_data, true_rul, feature_cols, dataset, subsample, quick_mode, n_estimators, max_depth):
    results = {}
    
    if subsample and dataset in ['FD002', 'FD004']:
        train_engines = train_data['unit_number'].unique()
        sample_engines = np.random.choice(train_engines, size=int(len(train_engines) * 0.7), replace=False)
        train_data = train_data[train_data['unit_number'].isin(sample_engines)]
    
    engines = train_data['unit_number'].unique()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    models = {'Random Forest': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)}
    if not quick_mode:
        models.update({
            'Linear Regression': LinearRegression(),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42, n_jobs=-1)
        })
    
    status_text = st.empty()
    model_results = {}
    for model_name, model in models.items():
        status_text.text(f"Training {model_name} (Cross-Validation)...")
        rmse_scores, mae_scores, r2_scores, custom_scores = [], [], [], []
        for train_idx in stqdm(kf.split(engines), total=3, desc=f"{model_name} CV"):
            train_idx, val_idx = train_idx
            train_engines = engines[train_idx]
            val_engines = engines[val_idx]
            train_df = train_data[train_data['unit_number'].isin(train_engines)]
            val_df = train_data[train_data['unit_number'].isin(val_engines)]
            
            X_train = train_df[feature_cols]
            y_train = train_df['RUL']
            X_val = val_df[feature_cols]
            y_val = val_df['RUL']
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))
            custom_scores.append(custom_score(y_val, y_pred))
        
        model_results[model_name] = {
            'RMSE': np.mean(rmse_scores),
            'MAE': np.mean(mae_scores),
            'R2': np.mean(r2_scores),
            'Custom Score': np.mean(custom_scores)
        }
    
    status_text.text("Training final Random Forest...")
    X_train = train_data[feature_cols]
    y_train = train_data['RUL']
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    test_last = test_data.groupby('unit_number').last().reset_index()
    X_test = test_last[feature_cols]
    y_test_pred = rf.predict(X_test)
    
    rmse_test = np.sqrt(mean_squared_error(true_rul['true_rul'], y_test_pred))
    mae_test = mean_absolute_error(true_rul['true_rul'], y_test_pred)
    r2_test = r2_score(true_rul['true_rul'], y_test_pred)
    score_test = custom_score(true_rul['true_rul'], y_test_pred)
    
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
    
    fig = px.scatter(x=true_rul['true_rul'], y=y_test_pred, labels={'x': 'True RUL', 'y': 'Predicted RUL'},
                     title=f'Predicted vs True RUL - {dataset}', template='plotly_dark')
    fig.add_trace(go.Scatter(x=[0, max(true_rul['true_rul'].max(), y_test_pred.max())],
                             y=[0, max(true_rul['true_rul'].max(), y_test_pred.max())],
                             mode='lines', line=dict(dash='dash', color='red'), name='y=x'))
    
    status_text.text("Training completed!")
    
    results = {
        'CV': model_results,
        'Test': {'RMSE': rmse_test, 'MAE': mae_test, 'R2': r2_test, 'Custom Score': score_test},
        'Feature Importance': feature_importance_df,
        'Predictions': (true_rul['true_rul'], y_test_pred),
        'Plot': fig
    }
    
    return results

# Sidebar
with st.sidebar:
    st.image("https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png?resize=150,150", width=100)
    st.title("Turbofan Engine RUL Prediction")
    
    with st.expander("Dataset Selection"):
        dataset = st.selectbox("Select Dataset", ['FD001', 'FD002', 'FD003', 'FD004'], help="Choose a dataset to analyze.")
    
    with st.expander("Training Settings"):
        subsample = st.checkbox("Subsample FD002/FD004 (70%)", value=True, help="Reduces training data size for faster processing.")
        quick_mode = st.checkbox("Quick Mode (Random Forest Only)", value=True, help="Trains only Random Forest for faster results.")
        use_top_features = st.checkbox("Use Top 5 Sensors Only", value=False, help="Uses features from the 5 sensors most correlated with RUL.")
        n_estimators = st.slider("Number of Estimators (Random Forest)", 20, 100, 50, help="Number of trees in Random Forest.")
        max_depth_options = ['None', 10, 20]
        max_depth = st.selectbox("Max Depth (Random Forest)", max_depth_options, help="Maximum depth of trees in Random Forest.")
        max_depth = None if max_depth == 'None' else max_depth
    
    train_button = st.button("Train Models", help="Start model training for the selected dataset.")

# Welcome message
st.markdown("""
<div class='card'>
    <h3>Welcome to the Turbofan Engine RUL Prediction App</h3>
    <p>This app analyzes NASA's Turbofan Engine Degradation Dataset to predict Remaining Useful Life (RUL).</p>
    <p><b>How to use:</b></p>
    <ul>
        <li>Select a dataset (FD001–FD004) in the sidebar.</li>
        <li>Adjust training settings (e.g., enable Quick Mode or tune Random Forest parameters).</li>
        <li>Click "Train Models" to process data and train models.</li>
        <li>Explore tabs: Data Overview, EDA, Model Training, Results, Summary, and Decision.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Overview", "EDA", "Model Training", "Results", "Summary", "Decision"])

# Load and preprocess data
train_data, test_data, true_rul, error = load_data(dataset)
if error:
    st.error(error + ". Please ensure all dataset files are in the working directory.")
    st.stop()

with tab1:
    st.header("Data Overview")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write(f"Dataset: {dataset}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data")
        st.dataframe(train_data.head())
        st.write(f"Shape: {train_data.shape}")
    
    with col2:
        st.subheader("Test Data")
        st.dataframe(test_data.head())
        st.write(f"Shape: {test_data.shape}")
    
    st.subheader("Missing Values")
    st.dataframe(train_data.isnull().sum())
    st.markdown("</div>", unsafe_allow_html=True)

# Preprocess and engineer features
train_data, test_data = preprocess_data(train_data, test_data, dataset)
train_data, test_data, feature_cols = engineer_features(train_data, test_data, dataset, use_top_features)

with tab2:
    st.header("Exploratory Data Analysis")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    sensor_options = [f'Sensor {i}' for i in range(1, 22)]
    selected_sensors = st.multiselect("Select Sensors to Plot", sensor_options, default=['Sensor 2', 'Sensor 3'])
    
    engine1 = train_data[train_data['unit_number'] == 1]
    fig = go.Figure()
    for sensor in selected_sensors:
        sensor_col = f'sensor_{sensor.split()[-1]}'
        fig.add_trace(go.Scatter(x=engine1['time_cycles'], y=engine1[sensor_col], mode='lines', name=sensor))
    fig.update_layout(title=f'Sensor Trends for Engine 1 - {dataset}', xaxis_title='Time Cycles', yaxis_title='Scaled Sensor Value',
                      template='plotly_dark')
    st.plotly_chart(fig)
    
    st.subheader("Top 5 Sensors Correlated with RUL")
    correlations = train_data[[f'sensor_{i}' for i in range(1, 22)] + ['RUL']].corr()
    sensor_rul_corr = correlations['RUL'].drop('RUL').sort_values(ascending=False).head()
    st.dataframe(sensor_rul_corr)
    
    if dataset in ['FD002', 'FD004']:
        st.subheader("Operating Condition Distribution")
        op_counts = train_data['op_condition'].value_counts().sort_index()
        fig = px.bar(x=op_counts.index, y=op_counts.values, labels={'x': 'Operating Condition', 'y': 'Count'},
                     title=f'Operating Condition Distribution - {dataset}', template='plotly_dark')
        st.plotly_chart(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.header("Model Training")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if train_button:
        with st.spinner("Training models... This may take a few minutes."):
            results = train_models(train_data, test_data, true_rul, feature_cols, dataset, subsample, quick_mode, n_estimators, max_depth)
            st.session_state[f'results_{dataset}'] = results
            st.success("Training completed!")
    else:
        st.info("Click 'Train Models' in the sidebar to start training.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.header("Results")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if f'results_{dataset}' in st.session_state:
        results = st.session_state[f'results_{dataset}']
        
        st.subheader("Cross-Validation Results")
        cv_df = pd.DataFrame(results['CV']).T
        st.dataframe(cv_df.style.format("{:.2f}"))
        
        st.subheader("Test Set Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{results['Test']['RMSE']:.2f}")
        col2.metric("MAE", f"{results['Test']['MAE']:.2f}")
        col3.metric("R2", f"{results['Test']['R2']:.2f}")
        col4.metric("Custom Score", f"{results['Test']['Custom Score']:.2f}")
        
        st.subheader("Top 10 Important Features")
        st.dataframe(results['Feature Importance'].head(10))
        
        st.subheader("Predicted vs True RUL")
        st.plotly_chart(results['Plot'])
    else:
        st.warning("No results yet. Train models in the 'Model Training' tab.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.header("Cross-Dataset Summary")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Compare model performance across all datasets (requires training each dataset).")
    
    summary_data = []
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        if f'results_{ds}' in st.session_state:
            results = st.session_state[f'results_{ds}']
            for model, metrics in results['CV'].items():
                summary_data.append({
                    'Dataset': ds,
                    'Model': model,
                    'CV RMSE': metrics['RMSE'],
                    'CV MAE': metrics['MAE'],
                    'CV R2': metrics['R2'],
                    'CV Custom Score': metrics['Custom Score']
                })
            summary_data.append({
                'Dataset': ds,
                'Model': 'Test (Random Forest)',
                'CV RMSE': results['Test']['RMSE'],
                'CV MAE': results['Test']['MAE'],
                'CV R2': results['Test']['R2'],
                'CV Custom Score': results['Test']['Custom Score']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.style.format("{:.2f}", subset=['CV RMSE', 'CV MAE', 'CV R2', 'CV Custom Score']))
    else:
        st.info("Train models for at least one dataset to view the summary.")
    
    st.subheader("Key Insights")
    st.markdown("""
    - FD001/FD003: Single operating condition leads to higher R2 due to simpler patterns.
    - FD002/FD004: Multiple conditions increase noise, reducing R2; clustering helps.
    - Random Forest: Generally outperforms other models due to non-linear handling.
    - Next Steps: Experiment with hyperparameter tuning, adjust feature selection, or try LSTM models.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with tab6:
    st.header("Decision")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("This section provides a recommendation on the best model and settings for predicting RUL based on performance and operational context.")
    
    if summary_data:
        # Analyze performance
        summary_df = pd.DataFrame(summary_data)
        avg_r2 = summary_df.groupby('Model')['CV R2'].mean()
        avg_custom_score = summary_df.groupby('Model')['CV Custom Score'].mean()
        
        # Find best model based on R2 (primary) and Custom Score (secondary)
        best_model_r2 = avg_r2.idxmax()
        best_r2_value = avg_r2.max()
        best_custom_score = avg_custom_score[best_model_r2]
        
        # Check performance by dataset type
        single_cond = summary_df[summary_df['Dataset'].isin(['FD001', 'FD003'])]
        multi_cond = summary_df[summary_df['Dataset'].isin(['FD002', 'FD004'])]
        single_r2 = single_cond[single_cond['Model'] == best_model_r2]['CV R2'].mean() if not single_cond.empty else 0
        multi_r2 = multi_cond[multi_cond['Model'] == best_model_r2]['CV R2'].mean() if not multi_cond.empty else 0
        
        st.subheader("Model Performance Summary")
        st.write(f"Average R2 Across Datasets:")
        st.dataframe(avg_r2.reset_index().style.format({"CV R2": "{:.2f}"}))
        st.write(f"Average Custom Score Across Datasets:")
        st.dataframe(avg_custom_score.reset_index().style.format({"CV Custom Score": "{:.2f}"}))
        
        st.subheader("Operational Context")
        st.markdown(f"""
        - FD001/FD003 (Single Condition): Average R2 for {best_model_r2} = {single_r2:.2f}. Simpler data patterns allow higher accuracy.
        - FD002/FD004 (Multiple Conditions): Average R2 for {best_model_r2} = {multi_r2:.2f}. Noise from varying conditions reduces accuracy.
        - Runtime Considerations: Random Forest is computationally intensive but robust. Quick Mode and top 5 sensors reduce training time by ~50–66%.
        """)
        
        st.subheader("Recommendation")
        recommendation = {
            "Model": best_model_r2,
            "Number of Estimators": n_estimators,
            "Max Depth": max_depth if max_depth else "None",
            "Use Top 5 Sensors": "Yes" if use_top_features else "No",
            "Subsample FD002/FD004": "Yes" if subsample else "No",
            "Justification": f"{best_model_r2} achieves the highest average R2 ({best_r2_value:.2f}) and competitive Custom Score ({best_custom_score:.2f}). "
                           f"For single-condition scenarios (FD001/FD003), use all features for maximum accuracy. "
                           f"For multi-condition scenarios (FD002/FD004), enable top 5 sensors to balance speed and performance. "
                           f"Set n_estimators={n_estimators} and max_depth={max_depth if max_depth else 'None'} for a good trade-off between accuracy and runtime."
        }
        st.markdown(f"""
        Final Decision: Deploy **{recommendation['Model']}** with the following settings:
        - Number of Estimators: {recommendation['Number of Estimators']}
        - Max Depth: {recommendation['Max Depth']}
        - Use Top 5 Sensors: {recommendation['Use Top 5 Sensors']}
        - Subsample FD002/FD004: {recommendation['Subsample FD002/FD004']}
        
        Justification: {recommendation['Justification']}
        """)
        
        # Downloadable summary
        decision_df = pd.DataFrame([recommendation])
        csv = decision_df.to_csv(index=False)
        st.download_button("Download Decision Summary", csv, "decision_summary.csv")
    else:
        st.warning("No results available. Train models for at least one dataset to view the decision.")
    
    st.markdown("</div>", unsafe_allow_html=True)