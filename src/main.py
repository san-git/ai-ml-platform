#!/usr/bin/env python3
"""
AI/ML Application - Multi-Model Machine Learning Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="🤖 AI/ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

class MLPlatform:
    """Main ML Platform Class"""
    
    def __init__(self):
        self.models = {
            'Classification': {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42)
            },
            'Regression': {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            },
            'Clustering': {
                'K-Means': KMeans(n_clusters=3, random_state=42)
            }
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_sample_data(self, dataset_type):
        """Load sample datasets"""
        if dataset_type == "Iris Classification":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df, 'target'
        
        elif dataset_type == "Diabetes Regression":
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df, 'target'
        
        elif dataset_type == "Breast Cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df, 'target'
        
        elif dataset_type == "Random Data":
            # Generate random data for clustering
            np.random.seed(42)
            n_samples = 300
            centers = 3
            
            X = np.random.randn(n_samples, 2)
            y = np.random.randint(0, centers, n_samples)
            
            df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
            df['target'] = y
            return df, 'target'
    
    def preprocess_data(self, df, target_column, task_type):
        """Preprocess data for ML"""
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = self.label_encoder.fit_transform(X[col])
        
        # Scale features for some models
        if task_type in ['Classification', 'Regression']:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X, y
    
    def train_model(self, X, y, model_name, task_type):
        """Train the selected model"""
        model = self.models[task_type][model_name]
        
        if task_type == 'Clustering':
            # For clustering, we don't split the data
            model.fit(X)
            return model, None, None, None
        else:
            # Split data for supervised learning
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model.fit(X_train, y_train)
            return model, X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model, X_test, y_test, task_type):
        """Evaluate model performance"""
        if task_type == 'Clustering':
            return {"inertia": model.inertia_}
        
        y_pred = model.predict(X_test)
        
        if task_type == 'Classification':
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            return {
                "accuracy": accuracy,
                "precision": report['weighted avg']['precision'],
                "recall": report['weighted avg']['recall'],
                "f1_score": report['weighted avg']['f1-score']
            }
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            return {
                "mse": mse,
                "rmse": rmse,
                "r2_score": model.score(X_test, y_test)
            }
    
    def make_prediction(self, model, input_data, feature_names):
        """Make predictions on new data"""
        # Preprocess input data
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data, columns=feature_names)
        
        # Scale if needed
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'transform'):
            input_scaled = self.scaler.transform(input_df)
            input_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        prediction = model.predict(input_df)
        return prediction

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI/ML Platform</h1>', unsafe_allow_html=True)
    
    # Initialize ML platform
    ml_platform = MLPlatform()
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Navigation")
    
    # Main menu
    option = st.sidebar.selectbox(
        "Choose an option:",
        [
            "📊 Data Upload & Exploration",
            "🤖 Model Training",
            "📈 Model Evaluation",
            "🔮 Make Predictions",
            "📊 Data Visualization",
            "💾 Save/Load Models"
        ]
    )
    
    # Handle different options
    if option == "📊 Data Upload & Exploration":
        data_exploration_page(ml_platform)
    elif option == "🤖 Model Training":
        model_training_page(ml_platform)
    elif option == "📈 Model Evaluation":
        model_evaluation_page(ml_platform)
    elif option == "🔮 Make Predictions":
        prediction_page(ml_platform)
    elif option == "📊 Data Visualization":
        visualization_page(ml_platform)
    elif option == "💾 Save/Load Models":
        save_load_page(ml_platform)

def data_exploration_page(ml_platform):
    """Data upload and exploration page"""
    st.markdown("## 📊 Data Upload & Exploration")
    
    # Data upload options
    upload_option = st.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Sample Dataset"]
    )
    
    if upload_option == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    else:  # Sample dataset
        sample_datasets = [
            "Iris Classification",
            "Diabetes Regression", 
            "Breast Cancer",
            "Random Data"
        ]
        
        selected_dataset = st.selectbox("Select sample dataset:", sample_datasets)
        
        if st.button("Load Sample Dataset"):
            df, target_col = ml_platform.load_sample_data(selected_dataset)
            st.session_state.uploaded_data = df
            st.session_state.target_column = target_col
            st.success(f"✅ {selected_dataset} loaded successfully! Shape: {df.shape}")
    
    # Display data if available
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Data info
        st.markdown("### 📊 Data Information")
        st.write(df.info())
        
        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Missing values
        if df.isnull().sum().sum() > 0:
            st.markdown("### ⚠️ Missing Values")
            missing_data = df.isnull().sum()
            st.bar_chart(missing_data)

def model_training_page(ml_platform):
    """Model training page"""
    st.markdown("## 🤖 Model Training")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the Data Upload section.")
        return
    
    df = st.session_state.uploaded_data
    
    # Task type selection
    task_type = st.selectbox(
        "Select task type:",
        ["Classification", "Regression", "Clustering"]
    )
    
    # Target column selection
    if task_type in ["Classification", "Regression"]:
        target_column = st.selectbox(
            "Select target column:",
            df.columns.tolist(),
            index=df.columns.get_loc(st.session_state.target_column) if st.session_state.target_column else 0
        )
        st.session_state.target_column = target_column
    
    # Model selection
    available_models = list(ml_platform.models[task_type].keys())
    selected_model = st.selectbox(f"Select {task_type} model:", available_models)
    
    # Training parameters
    st.markdown("### ⚙️ Training Parameters")
    
    if selected_model == "Random Forest":
        n_estimators = st.slider("Number of estimators:", 10, 200, 100)
        max_depth = st.slider("Max depth:", 1, 20, 10)
        
        if task_type == "Classification":
            ml_platform.models[task_type][selected_model] = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        else:
            ml_platform.models[task_type][selected_model] = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
    
    elif selected_model == "K-Means":
        n_clusters = st.slider("Number of clusters:", 2, 10, 3)
        ml_platform.models[task_type][selected_model] = KMeans(
            n_clusters=n_clusters, random_state=42
        )
    
    # Train model
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Preprocess data
            if task_type in ["Classification", "Regression"]:
                X, y = ml_platform.preprocess_data(df, target_column, task_type)
            else:
                X, y = ml_platform.preprocess_data(df, st.session_state.target_column, task_type)
            
            # Train model
            model, X_train, X_test, y_train, y_test = ml_platform.train_model(
                X, y, selected_model, task_type
            )
            
            # Store results
            st.session_state.trained_model = {
                'model': model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X': X,
                'y': y,
                'feature_names': X.columns.tolist(),
                'task_type': task_type,
                'model_name': selected_model
            }
            
            st.session_state.model_type = task_type
            
            st.success(f"✅ {selected_model} trained successfully!")
            
            # Show training info
            if task_type != "Clustering":
                st.info(f"Training set size: {len(X_train)} samples")
                st.info(f"Test set size: {len(X_test)} samples")

def model_evaluation_page(ml_platform):
    """Model evaluation page"""
    st.markdown("## 📈 Model Evaluation")
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the Model Training section.")
        return
    
    model_data = st.session_state.trained_model
    task_type = model_data['task_type']
    
    # Evaluate model
    if st.button("📊 Evaluate Model"):
        with st.spinner("Evaluating model..."):
            metrics = ml_platform.evaluate_model(
                model_data['model'],
                model_data['X_test'],
                model_data['y_test'],
                task_type
            )
            
            # Display metrics
            st.markdown("### 📊 Model Performance")
            
            if task_type == "Classification":
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            
            elif task_type == "Regression":
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("MSE", f"{metrics['mse']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with col3:
                    st.metric("R² Score", f"{metrics['r2_score']:.4f}")
            
            else:  # Clustering
                st.metric("Inertia", f"{metrics['inertia']:.4f}")
            
            # Feature importance for Random Forest
            if "Random Forest" in model_data['model_name']:
                st.markdown("### 🌳 Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': model_data['feature_names'],
                    'Importance': model_data['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Feature Importances"
                )
                st.plotly_chart(fig, use_container_width=True)

def prediction_page(ml_platform):
    """Prediction page"""
    st.markdown("## 🔮 Make Predictions")
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the Model Training section.")
        return
    
    model_data = st.session_state.trained_model
    task_type = model_data['task_type']
    
    st.markdown("### 📝 Input Data")
    
    # Create input form
    input_data = {}
    
    for feature in model_data['feature_names']:
        if feature in model_data['X'].columns:
            # Get the data type and range for better input
            col_data = model_data['X'][feature]
            
            if col_data.dtype in ['int64', 'float64']:
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                default_val = float(col_data.mean())
                
                input_data[feature] = st.number_input(
                    f"{feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )
            else:
                input_data[feature] = st.text_input(f"{feature}:", value="0")
    
    # Make prediction
    if st.button("🔮 Make Prediction", type="primary"):
        with st.spinner("Making prediction..."):
            prediction = ml_platform.make_prediction(
                model_data['model'],
                input_data,
                model_data['feature_names']
            )
            
            st.markdown("### 🎯 Prediction Result")
            
            if task_type == "Classification":
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Class: {prediction[0]}</h3>
                </div>
                """, unsafe_allow_html=True)
            elif task_type == "Regression":
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Value: {prediction[0]:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:  # Clustering
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Assigned Cluster: {prediction[0]}</h3>
                </div>
                """, unsafe_allow_html=True)

def visualization_page(ml_platform):
    """Data visualization page"""
    st.markdown("## 📊 Data Visualization")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the Data Upload section.")
        return
    
    df = st.session_state.uploaded_data
    
    # Visualization options
    viz_option = st.selectbox(
        "Select visualization:",
        ["Correlation Matrix", "Distribution Plots", "Scatter Plot", "Box Plot", "PCA Visualization"]
    )
    
    if viz_option == "Correlation Matrix":
        st.markdown("### 🔗 Correlation Matrix")
        
        # Calculate correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Distribution Plots":
        st.markdown("### 📈 Distribution Plots")
        
        selected_column = st.selectbox("Select column:", df.select_dtypes(include=[np.number]).columns)
        
        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}",
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Scatter Plot":
        st.markdown("### 🎯 Scatter Plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis:", df.select_dtypes(include=[np.number]).columns)
        with col2:
            y_col = st.selectbox("Y-axis:", df.select_dtypes(include=[np.number]).columns)
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=f"{x_col} vs {y_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Box Plot":
        st.markdown("### 📦 Box Plot")
        
        selected_column = st.selectbox("Select column:", df.select_dtypes(include=[np.number]).columns)
        
        fig = px.box(
            df,
            y=selected_column,
            title=f"Box Plot of {selected_column}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "PCA Visualization":
        st.markdown("### 🎨 PCA Visualization")
        
        if st.button("Generate PCA"):
            # Perform PCA
            numeric_df = df.select_dtypes(include=[np.number])
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(numeric_df)
            
            pca_df = pd.DataFrame(
                data=pca_result,
                columns=['PC1', 'PC2']
            )
            
            # Add target if available
            if st.session_state.target_column and st.session_state.target_column in df.columns:
                pca_df['target'] = df[st.session_state.target_column]
                
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='target',
                    title="PCA Visualization (2D)"
                )
            else:
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    title="PCA Visualization (2D)"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show explained variance
            st.markdown("### 📊 Explained Variance")
            explained_variance = pca.explained_variance_ratio_
            st.write(f"PC1: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
            st.write(f"PC2: {explained_variance[1]:.4f} ({explained_variance[1]*100:.2f}%)")

def save_load_page(ml_platform):
    """Save/Load models page"""
    st.markdown("## 💾 Save/Load Models")
    
    # Save model
    st.markdown("### 💾 Save Model")
    
    if st.session_state.trained_model is not None:
        model_name = st.text_input("Model name:", value="my_model")
        
        if st.button("💾 Save Model"):
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save model
            model_path = f"models/{model_name}.joblib"
            joblib.dump(st.session_state.trained_model['model'], model_path)
            
            # Save metadata
            metadata = {
                'task_type': st.session_state.trained_model['task_type'],
                'model_name': st.session_state.trained_model['model_name'],
                'feature_names': st.session_state.trained_model['feature_names'],
                'saved_at': datetime.now().isoformat()
            }
            
            metadata_path = f"models/{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            st.success(f"✅ Model saved as {model_path}")
    
    # Load model
    st.markdown("### 📂 Load Model")
    
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if f.endswith('.joblib')]
        
        if model_files:
            selected_model = st.selectbox("Select model to load:", model_files)
            
            if st.button("📂 Load Model"):
                # Load model
                model_path = f"models/{selected_model}"
                loaded_model = joblib.load(model_path)
                
                # Load metadata
                metadata_path = f"models/{selected_model.replace('.joblib', '_metadata.json')}"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    st.success(f"✅ Model loaded successfully!")
                    st.info(f"Task Type: {metadata['task_type']}")
                    st.info(f"Model Name: {metadata['model_name']}")
                    st.info(f"Features: {len(metadata['feature_names'])}")
                    st.info(f"Saved: {metadata['saved_at']}")
                else:
                    st.success(f"✅ Model loaded successfully!")
        else:
            st.info("No saved models found.")
    else:
        st.info("No models directory found.")

if __name__ == "__main__":
    main() 