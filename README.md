# Turbofan Engine RUL Prediction

A comprehensive machine learning application for predicting the Remaining Useful Life (RUL) of turbofan engines using NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

## Overview

This interactive Streamlit application provides a complete platform for engine prognostics, enabling users to analyze sensor data from turbofan engines and predict their remaining operational lifetime. The application brings together data exploration, feature engineering, model training, and real-time prediction capabilities in a single, user-friendly interface.

## Features

- **Multi-Dataset Support**: Work with all four C-MAPSS datasets (FD001-FD004)
- **Interactive Data Exploration**: Visualize sensor trends, correlations, and operating conditions
- **Advanced Feature Engineering**: Automated creation of rolling statistics and lag features
- **Multiple ML Models**: Compare Linear Regression, Random Forest, and XGBoost algorithms
- **Cross-Validation**: Robust model evaluation with K-Fold cross-validation
- **Performance Optimization**: Quick mode, subsampling, and feature selection options
- **Real-Time Prediction**: Interactive tool for predicting RUL from custom sensor inputs
- **Visual Analytics**: Interactive plots using Plotly for comprehensive data visualization
- **Model Comparison**: Cross-dataset performance summary and recommendations

## Dataset

The application uses NASA's Turbofan Engine Degradation Simulation Dataset, which contains run-to-failure time series data from simulated aircraft engines. The dataset includes:

- **4 subdatasets** (FD001, FD002, FD003, FD004)
- **26 columns** per record: 1 unit number, 1 time cycle, 3 operational settings, and 21 sensor measurements
- **Multiple operating conditions**: Single condition (FD001/FD003) and six conditions (FD002/FD004)
- **Different fault modes**: HPC degradation (FD001/FD002) and HPC + Fan degradation (FD003/FD004)

### Dataset Requirements

The application expects the following files to be present in either the project root directory or a `dataset` subdirectory:

```
train_FD001.txt, test_FD001.txt, RUL_FD001.txt
train_FD002.txt, test_FD002.txt, RUL_FD002.txt
train_FD003.txt, test_FD003.txt, RUL_FD003.txt
train_FD004.txt, test_FD004.txt, RUL_FD004.txt
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Dependencies

Install all required packages using pip:

```bash
pip install streamlit pandas numpy scikit-learn plotly xgboost stqdm
```

### Detailed Package List

- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms and preprocessing
- `plotly`: Interactive visualization library
- `xgboost`: Gradient boosting framework
- `stqdm`: Progress bars for Streamlit

## Usage

### Running the Application

1. Navigate to the project directory:
```bash
cd path/to/AllDataMine
```

2. Launch the Streamlit application:
```bash
streamlit run app.py
```

3. The application will automatically open in your default web browser (typically at `http://localhost:8501`)

### Application Workflow

1. **Select Dataset**: Choose one of the four C-MAPSS datasets (FD001-FD004) from the sidebar
2. **Configure Training Settings**:
   - Enable/disable subsampling for large datasets (FD002/FD004)
   - Toggle Quick Mode to train only Random Forest
   - Select top 5 sensors only for faster processing
   - Adjust Random Forest hyperparameters (n_estimators, max_depth)
3. **Train Models**: Click the "Train Models" button to start the pipeline
4. **Explore Results**: Navigate through the tabs to analyze data, models, and predictions

### Application Tabs

- **Data Overview**: View raw data samples, shapes, and missing value statistics
- **EDA**: Explore sensor trends, correlations with RUL, and operating conditions
- **Model Training**: Monitor the training process and status
- **Results**: Review cross-validation metrics, test set performance, feature importance, and prediction plots
- **Summary**: Compare model performance across all datasets
- **Decision**: Get automated recommendations for optimal model configuration
- **RUL Prediction**: Input custom sensor values to predict remaining useful life

## Technical Architecture

### Core Functions

#### Data Loading (`load_data`)
- Loads train, test, and RUL data files
- Supports flexible file locations (root directory or dataset folder)
- Cached with `@st.cache_data` for improved performance

#### Data Preprocessing (`preprocess_data`)
- Calculates RUL target variable from time cycles
- Applies MinMaxScaler for feature normalization
- Uses KMeans clustering to identify operating conditions in complex datasets
- Cached for efficiency

#### Feature Engineering (`engineer_features`)
- Creates rolling window statistics (mean, standard deviation)
- Generates time-lagged features (lag-1, lag-2)
- Supports optional top-5 sensor selection based on correlation with RUL
- Cached to prevent redundant computation

#### Model Training (`train_models`)
- Implements K-Fold cross-validation (3 folds)
- Trains multiple models: Linear Regression, Random Forest, XGBoost
- Evaluates using RMSE, MAE, R2, and custom asymmetric scoring function
- Stores trained model in session state for predictions
- Cached with `@st.cache_resource`

### Scoring Function

The application uses an asymmetric scoring function that reflects the reality of aviation maintenance:

```python
def custom_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 10) - 1, np.exp(d / 13) - 1))
```

This function penalizes late predictions (negative errors) more severely than early predictions, as predicting failure too late is far more dangerous in aviation contexts.

## Performance Optimization

The application includes several optimization options to balance speed and accuracy:

- **Subsample FD002/FD004**: Reduces training data to 70% of engines for faster processing
- **Quick Mode**: Trains only Random Forest instead of all three models
- **Use Top 5 Sensors**: Limits features to the 5 sensors most correlated with RUL
- **Adjustable Hyperparameters**: Fine-tune Random Forest n_estimators and max_depth

Combining these options can reduce training time by 50-66% with minimal impact on accuracy.

## Project Structure

```
AllDataMine/
├── app.py                      # Main Streamlit application
├── appv0.ipynb                 # Jupyter notebook version
├── documentation_app.html      # Interactive HTML documentation
├── documentation_appv0.html    # Documentation for notebook version
├── data.zip                    # Compressed dataset files
├── Rapport.pdf                 # Project report
├── readme.txt                  # Original dataset description
└── README.md                   # This file
```

## Troubleshooting

### FileNotFoundError
**Symptom**: Error message "File train_FD001.txt not found" on startup

**Solution**: Ensure C-MAPSS data files are in the project directory or `dataset/` subdirectory. Extract `data.zip` if necessary.

### ModuleNotFoundError
**Symptom**: Application fails to launch citing missing module

**Solution**: Install all required dependencies using the pip command shown in the Installation section.

### Slow Performance or Crash
**Symptom**: Unresponsive application during training on large datasets

**Solution**: Enable performance optimization options in the sidebar:
- Check "Subsample FD002/FD004"
- Enable "Quick Mode"
- Select "Use Top 5 Sensors Only"

### RUL Prediction Error
**Symptom**: Error when clicking "Predict RUL"

**Solution**: Train a model first using the "Train Models" button before using the prediction tool.

## Scientific Background

This project is based on the research paper "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation" which introduced the C-MAPSS simulation environment. The simulation models realistic engine degradation by:

- Gradually decreasing flow and efficiency of key components (High-Pressure Compressor)
- Tracking health index based on proximity to operational safety limits
- Generating run-to-failure data under various operating conditions and fault modes

## Key Insights

- **FD001/FD003** (Single Operating Condition): Higher R2 scores due to simpler degradation patterns
- **FD002/FD004** (Multiple Conditions): Lower R2 scores due to increased noise from varying conditions
- **Random Forest**: Generally outperforms other models due to superior handling of non-linear relationships
- **Feature Engineering**: Rolling statistics and lag features significantly improve prediction accuracy
- **Operating Conditions**: KMeans clustering helps models distinguish degradation from operational changes

## Future Improvements

- Deep Learning models (LSTM, GRU) for temporal pattern recognition
- Hyperparameter tuning with GridSearchCV or Bayesian optimization
- Additional feature engineering techniques
- Real-time streaming data support
- Export trained models for deployment
- Extended prediction visualization tools

## Authors

GIIADS-StackUnderflow Project

## License

This project is provided for educational and research purposes.

## Acknowledgments

- NASA for providing the C-MAPSS dataset
- The authors of "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
- The open-source community for the excellent Python libraries used in this project
