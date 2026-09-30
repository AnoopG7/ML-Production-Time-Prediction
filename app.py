"""
Production Time Estimator -- Streamlit Application
Deploy on Streamlit Community Cloud or run locally: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# -- Page Config --
st.set_page_config(
    page_title="Production Time Estimator",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Custom CSS --
st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E3A5F, #4C72B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
''', unsafe_allow_html=True)

# -- Resolve file paths (works both locally and deployed) --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve(filename):
    return os.path.join(BASE_DIR, filename)

# -- Load Artifacts --
@st.cache_resource
def load_artifacts():
    model = joblib.load(resolve('production_time_model.pkl'))
    scaler = joblib.load(resolve('scaler.pkl'))
    encoders = joblib.load(resolve('label_encoders.pkl'))
    feature_names = joblib.load(resolve('feature_names.pkl'))
    comparison = pd.read_csv(resolve('model_comparison.csv'))
    return model, scaler, encoders, feature_names, comparison

try:
    model, scaler, encoders, feature_names, comparison_df = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model files not found. Run the notebook first to generate .pkl files. Error: {e}")

# -- Header --
st.markdown('<p class="main-header">Production Time Estimator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-powered production time prediction for factory process optimization</p>', unsafe_allow_html=True)

# -- Tabs --
tab1, tab2, tab3 = st.tabs(["Predict", "Model Insights", "Data Explorer"])

# ==============================================
# TAB 1: PREDICTION
# ==============================================
with tab1:
    st.header("Production Time Prediction")
    st.markdown("Adjust the parameters below to estimate how long a production batch will take.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Product Details")
        product_type = st.selectbox("Product Type",
            ['Widget_A', 'Widget_B', 'Widget_C', 'Gadget_X', 'Gadget_Y'])
        batch_size = st.number_input("Batch Size (units)", 10, 500, 100, 10,
                                     help="Number of units in the batch")
        complexity_score = st.slider("Complexity Score", 1, 5, 3,
                                     help="Product complexity level (1=simple, 5=complex)")
        material_grade = st.selectbox("Material Grade", ['Standard', 'Premium', 'Economy'])

    with col2:
        st.subheader("Machine & Workers")
        machine_id = st.selectbox("Machine ID",
            [f'M{i:02d}' for i in range(1, 11)])
        num_workers = st.slider("Number of Workers", 1, 8, 4)
        operator_exp = st.number_input("Operator Experience (years)", 0.5, 25.0, 5.0, 0.5)
        setup_time = st.number_input("Setup Time (minutes)", 5.0, 60.0, 25.0, 1.0)

    with col3:
        st.subheader("Environment & Schedule")
        ambient_temp = st.number_input("Ambient Temp (C)", 15.0, 45.0, 28.0, 0.5)
        humidity = st.number_input("Humidity (%)", 20.0, 100.0, 55.0, 1.0)
        shift = st.selectbox("Shift", ['Morning', 'Afternoon', 'Night'])
        day = st.selectbox("Day of Week", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        defect_rate = st.number_input("Defect Rate (%)", 0.0, 15.0, 3.0, 0.5)
        planned_time = st.number_input("Planned Time (hours)", 0.5, 50.0, 10.0, 0.5)

    if st.button("Predict Production Time", type="primary", use_container_width=True):
        if model_loaded:
            le_product = encoders['le_product']
            le_machine = encoders['le_machine']
            le_shift = encoders['le_shift']
            le_day = encoders['le_day']

            # Encode inputs
            product_enc = le_product.transform([product_type])[0]
            machine_enc = le_machine.transform([machine_id])[0]
            shift_enc = le_shift.transform([shift])[0]
            day_enc = le_day.transform([day])[0]

            # Engineered features
            batch_per_worker = batch_size / max(num_workers, 1)
            setup_per_batch = setup_time / max(batch_size, 1)

            input_dict = {
                'batch_size': batch_size,
                'operator_experience_yrs': operator_exp,
                'setup_time_min': setup_time,
                'complexity_score': complexity_score,
                'ambient_temp_c': ambient_temp,
                'humidity_pct': humidity,
                'num_workers': num_workers,
                'defect_rate_pct': defect_rate,
                'planned_time_hrs': planned_time,
                'product_type_encoded': product_enc,
                'machine_id_encoded': machine_enc,
                'shift_encoded': shift_enc,
                'day_encoded': day_enc,
                'material_grade_Premium': 1 if material_grade == 'Premium' else 0,
                'material_grade_Standard': 1 if material_grade == 'Standard' else 0,
                'batch_per_worker': batch_per_worker,
                'setup_per_batch': setup_per_batch,
            }

            input_data = pd.DataFrame([input_dict])
            input_data = input_data.reindex(columns=feature_names, fill_value=0)

            prediction = max(0.1, model.predict(input_data)[0])

            # -- Display Results --
            st.markdown("---")
            st.subheader("Prediction Results")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Predicted Time", f"{prediction:.2f} hrs")
            with c2:
                gap = prediction - planned_time
                st.metric("Schedule Deviation", f"{gap:+.2f} hrs",
                          delta=f"{gap:+.2f}", delta_color="inverse")
            with c3:
                rate = batch_size / max(prediction, 0.01)
                st.metric("Units / Hour", f"{rate:.1f}")
            with c4:
                status = "On Time" if prediction <= planned_time else "Delayed"
                st.metric("Status", status)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={'reference': planned_time, 'increasing': {'color': '#FF6B6B'},
                       'decreasing': {'color': '#66BB6A'}},
                title={'text': "Predicted Production Time (hours)", 'font': {'size': 18}},
                number={'suffix': ' hrs', 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, max(prediction, planned_time) * 1.5], 'tickwidth': 1},
                    'bar': {'color': "#4C72B0"},
                    'steps': [
                        {'range': [0, planned_time * 0.8], 'color': '#C8E6C9'},
                        {'range': [planned_time * 0.8, planned_time], 'color': '#FFF9C4'},
                        {'range': [planned_time, max(prediction, planned_time) * 1.5], 'color': '#FFCDD2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.8,
                        'value': planned_time
                    }
                }
            ))
            fig.update_layout(height=350, margin=dict(t=80, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model not loaded. Run the notebook first to generate model files.")

# ==============================================
# TAB 2: MODEL INSIGHTS
# ==============================================
with tab2:
    st.header("Model Performance Insights")

    if model_loaded:
        st.subheader("Model Comparison")
        metric_choice = st.selectbox("Select Metric", ['Test R2', 'MAE', 'RMSE', 'CV R2 Mean'])
        ascending = 'R2' not in metric_choice
        fig = px.bar(comparison_df.sort_values(metric_choice, ascending=ascending),
                     x=metric_choice, y='Model', orientation='h',
                     color=metric_choice, color_continuous_scale='blues',
                     title=f'Model Comparison by {metric_choice}')
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full Comparison Table")
        st.dataframe(comparison_df.style.format(precision=4), use_container_width=True)

        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance (Best Model)")
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='blues',
                         title='Feature Importance -- Best Model')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# ==============================================
# TAB 3: DATA EXPLORER
# ==============================================
with tab3:
    st.header("Data Explorer")

    # Try to load default dataset
    default_csv = resolve('factory_production_data.csv')
    uploaded_file = st.file_uploader("Upload a CSV to explore", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    elif os.path.exists(default_csv):
        data = pd.read_csv(default_csv)
        st.info("Showing the factory production dataset. Upload your own CSV above to explore different data.")
    else:
        data = None
        st.info("Upload a CSV file to explore it here.")

    if data is not None:
        st.write(f"**Shape:** {data.shape[0]} rows x {data.shape[1]} columns")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.dataframe(data.head(50), use_container_width=True)
        with col2:
            st.subheader("Statistics")
            st.dataframe(data.describe().round(3), use_container_width=True)

        st.subheader("Missing Values")
        missing = data.isnull().sum()
        if missing.sum() > 0:
            st.dataframe(missing[missing > 0].to_frame('Count'), use_container_width=True)
        else:
            st.success("No missing values found.")

# -- Footer --
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#888;"><b>Production Time Estimator</b> | '
    'Machine Learning Foundations -- Sem 5 | Case Study 100</p>',
    unsafe_allow_html=True
)
