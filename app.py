import math
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model.preprocessdata import scale_test_data
from model.predictmodels import predict_and_generate_metrics
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide") 
st.markdown("""
    <style>
    /* 1. Metric Containers: Using a subtle gradient for depth */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e222e, #2a2f3d);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3d4455;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.2);
    }

    /* 2. Metric Labels: Using a bright Cyan for visibility */
    [data-testid="stMetricLabel"] {
        color: #00d4ff !important; 
        font-weight: 600 !important;
        letter-spacing: 0.8px;
    }

    /* 3. Metric Values: Amethyst/Purple to stand out */
    [data-testid="stMetricValue"] {
        color: #bd93f9 !important; 
    }

    /* 4. Main Headers: Clean Bold Blue with professional font */
    h1, h2, h3 {
        color: #f8f9fa !important;
        font-family: 'Inter', sans-serif;
    }

    h1 {
        border-bottom: 3px solid #00d4ff;
        padding-bottom: 12px;
        font-weight: 800;
    }

    /* 5. Subheaders: Used for Model Details section [cite: 92] */
    .stMarkdown h4 {
        color: #f8f9fa !important;
        background-color: #262730;
        padding: 8px 15px;
        border-radius: 6px;
        border-left: 6px solid #bd93f9;
    }

    /* 6. Section Headers: Deep Navy for high contrast [cite: 90] */
    .model-header {
        background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
        border-top: 2px solid #00d4ff;
        color: #00d4ff !important;
        padding: 15px;
        border-radius: 4px;
    }

    /* 7. Report Section: Using a bright accent for the Classification Report [cite: 94] */
    .main-section-header {
        color: #ffffff;
        background-color: #6272a4; /* Muted periwinkle blue */
        padding: 12px 20px;
        border-radius: 8px;
        margin-top: 25px;
        font-weight: 700;
        display: flex;
        align-items: center;
    }

    /* 8. File Uploader Styling  */
    .upload-box {
        border: 2px dashed #44475a;
        padding: 15px;
        border-radius: 12px;
        background-color: #282a36;
    }

    .upload-label {
        color: #50fa7b; /* Emerald green for "Actionable" items */
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
    }

    /* 9. Compact Secondary Buttons */
    button[kind="secondary"] {
        background-color: #44475a !important;
        color: #f8f9fa !important;
        border: 1px solid #6272a4 !important;
        border-radius: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
with st.sidebar:
    st.image("./rajulogo.png", width=100)
    st.title("Control Panel")
    
    
    model_dict = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }
    st.markdown("### Model Configuration")
    model_names = st.multiselect(
        "Select Model(s) to Evaluate", 
        list(model_dict.keys()), 
        default=list(model_dict.keys())[:2]
    )
    
    st.info("App for ML Assignment-2 submission.")
col1, col2 = st.columns(2)
with col2:
    # Split col2 into 3 parts: 2 parts spacer, 1 part content
    _, _, download_col = st.columns([1, 1, 1.5])
    
    with download_col:
        st.markdown('<p class="right-aligned-text">📁 Data Source</p>', unsafe_allow_html=True)
        
        test_file_path = "data/initial/bank.csv"
        if os.path.exists(test_file_path):
            with open(test_file_path, "rb") as file:
                st.download_button(
                    label="📥 Download CSV", # Shortened label for better sizing
                    data=file,
                    file_name="test_data.csv",
                    mime="text/csv"
                )
    
with col1:
    # Wrap in a div to apply the border and alignment
    #st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    
    # Left-aligned small text with icon
    st.markdown('<p class="upload-label">📤 Upload test data (CSV)</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.7rem; color: gray; margin-bottom: 10px;">Columns must match training data</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "hidden label", 
        type="csv", 
        label_visibility="collapsed"
    )
    
    #st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and model_names:
    
    uploaded_file.seek(0)
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop() # Stops execution if file is bad


    metrics_list = []
    confusion_matrices = {}
    predictions = {}

    # 2. LOAD SCALER ONCE OUTSIDE THE LOOP (Unless you have different scalers per model)
    # scaler_path = os.path.join("model/pkl", "scaler.pkl")
    # if os.path.exists(scaler_path):
    #     scaler = joblib.load(scaler_path)
    #     # Pre-process the data once
    #     X_scaled, y = fetch_processed_data_from_df(test_df, scaler)
    # else:
    #     st.error(f"Scaler file not found: {scaler_path}")
    #     st.stop()

    X_scaled, y = scale_test_data(test_df)
    

    for model_name in model_names:
        model_file_key = model_dict[model_name]
        model_path = os.path.join("model/pkl", f"{model_file_key}.pkl")
        scaler_path=os.path.join("model/pkl", f"scaler.pkl")
        print("------------------------------------------------------")
        print(scaler_path)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
            continue
        
        
        #X_scaled, y = fetch_processed_data_from_df(test_df,scaler)
     
        y_true = y if y is not None else None
        y_pred = model.predict(X_scaled)
        predictions[model_name] = y_pred

        if y_true is not None:
            metrics = predict_and_generate_metrics(model, X_scaled, y_true)
            metrics['Model'] = model_name
            metrics_list.append(metrics)
            cm = confusion_matrix(y_true, y_pred)
            confusion_matrices[model_name] = cm
        else:
            st.subheader(f"Predictions for {model_name}")
            st.write(y_pred)

    if y_true is not None and metrics_list:
        st.subheader("🏆 Model Highlights")
        metrics_df = pd.DataFrame(metrics_list).set_index('Model')
        top_model = metrics_df.sort_values(by='Accuracy', ascending=False).index[0]
        best_acc = metrics_df.loc[top_model, 'Accuracy']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", top_model)
        m2.metric("Highest Accuracy", f"{best_acc:.2%}")
        
        m3.metric("Best MCC", f"{metrics_df['MCC'].max():.2f}")

        st.divider()
        # Show metrics comparison table
        st.markdown('<span class="section-header">📊 Evaluation Metrics Comparison</span>', unsafe_allow_html=True)

        # Wrap the dataframe in the container WITHOUT the extra empty div
        with st.container():
            # We apply the styling directly to the dataframe display
            st.dataframe(
                metrics_df.style.highlight_max(axis=0, color='#0e4d25') # Darker green for dark mode
                                .highlight_min(axis=0, color='#4d0e0e') # Darker red for dark mode
                                .format(precision=3),
                use_container_width=True
            )

        # Show confusion matrices
        st.markdown('<div class="main-section-header">📊 Confusion Matrix</div>', unsafe_allow_html=True)
        model_cm_items = list(confusion_matrices.items())
        n_models = len(model_cm_items)
        cols = 2
        rows = math.ceil(n_models / cols) if cols > 0 else 1
        grid = [st.columns(cols) for _ in range(rows)]
        ##rows = math.ceil(n_models / cols) if cols > 0 else 1
        for idx, (model_name, cm) in enumerate(model_cm_items):
            row = idx // cols
            col = idx % cols
            if row < len(grid) and col < len(grid[row]):
                with grid[row][col]:
                    # Use the border=True container to act as a 'Card'
                    with st.container(border=True):
                        # Styled Model Title
                        #st.markdown(f'<p class="model-header">{model_name}</p>', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="model-header">
                                {model_name}
                            </div>
                        """, unsafe_allow_html=True)
                        # --- Confusion Matrix Section ---
                        #st.markdown('<p class="plot-header">📊 Confusion Matrix</p>', unsafe_allow_html=True)
                        st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
                        plt.close('all')
                        fig, ax = plt.subplots(figsize=(6, 4))
                        # Using a slightly different colormap (e.g., 'Purples' or 'GnBu') for variety
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            xticklabels=["Pred 0", "Pred 1"], 
                            yticklabels=["True 0", "True 1"], 
                            cbar=False, annot_kws={"size": 14})
                        fig.tight_layout()
                        st.pyplot(fig, width='stretch')
                        plt.close(fig)
                        
                       

        # Optionally, fill remaining cells in the last row to keep the grid neat
        last_row = len(grid) - 1
        for col in range(len(model_cm_items) % cols, cols):
            if last_row >= 0 and col < len(grid[last_row]):
                with grid[last_row][col]:
                    st.write("")

        st.markdown('<div class="main-section-header">📋 Classification Report </div>', unsafe_allow_html=True)
        rows2 = math.ceil(n_models / cols) if cols > 0 else 1
        grid2 = [st.columns(cols) for _ in range(rows2)]
        for idx2, (model_name, cm) in enumerate(model_cm_items):
            row2 = idx2 // cols
            col2 = idx2 % cols
            if row2< len(grid2) and col2 < len(grid2[row2]):
                with grid2[row2][col2]:
                    # Use the border=True container to act as a 'Card'
                    with st.container(border=True):
                        # Styled Model Title
                        #st.markdown(f'<p class="model-header">{model_name}</p>', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="model-header">
                                {model_name}
                            </div>
                        """, unsafe_allow_html=True)
                        # --- Classification Report Section ---
                        #st.markdown('<p class="plot-header">📋 Classification Report</p>', unsafe_allow_html=True)
                        report_df = pd.DataFrame(classification_report(y_true, predictions[model_name], output_dict=True)).transpose()
                        st.dataframe(report_df.style.format(precision=2), width='content')
       

        # Optionally, fill remaining cells in the last row to keep the grid neat
        last_row2 = len(grid2) - 1
        for col2 in range(len(model_cm_items) % cols, cols):
            if last_row2 >= 0 and col2 < len(grid2[last_row2]):
                with grid[last_row2][col2]:
                    st.write("")

