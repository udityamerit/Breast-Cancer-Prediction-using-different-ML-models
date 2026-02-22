import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Page Configuration ---
st.set_page_config(page_title="Cancer Vision AI", page_icon="🧬", layout="wide")

# --- 2. Next-Gen Medical UI CSS ---
st.markdown("""
    <style>
    /* Sleek, deep clinical background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 0%, #1a2639 0%, #0d1117 100%);
        color: #e6edf3;
    }
    
    /* Clean up headers */
    h1, h2, h3, h4, p, span {
        color: #e6edf3 !important;
    }
    
    /* Style the Predict Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: white !important;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        height: 55px;
        border: 1px solid #90e0ef;
        box-shadow: 0 0 15px rgba(0, 180, 216, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0077b6, #023e8a);
        box-shadow: 0 0 25px rgba(0, 180, 216, 0.7);
        transform: translateY(-2px);
    }
    
    /* Cards for UI elements */
    div[data-testid="stForm"] {
        background-color: rgba(22, 33, 49, 0.7);
        border: 1px solid #304156;
        border-radius: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #304156;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to style Matplotlib charts for the dark theme
def apply_dark_theme(fig, ax, is_polar=False):
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.tick_params(colors='#e6edf3')
    if not is_polar:
        ax.xaxis.label.set_color('#e6edf3')
        ax.yaxis.label.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#304156')
    ax.title.set_color('#e6edf3')

# --- 3. Load Resources (Model, Scaler, Data) ---
@st.cache_resource
def load_resources():
    # Load models
    model = joblib.load('breast_cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Load and clean data for visualizations
    data = pd.read_csv('data.csv')
    data.columns = data.columns.str.replace('x.', '', regex=False)
    data = data.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
    
    # Format diagnosis
    if 'diagnosis' in data.columns:
        data['Diagnosis_Label'] = data['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})
        
    return model, scaler, data

try:
    model, scaler, data = load_resources()
except Exception as e:
    st.error("Error loading resources. Ensure 'breast_cancer_model.pkl', 'scaler.pkl', and 'data.csv' are in the directory.")
    st.stop()

# --- 4. Sidebar Navigation ---
st.sidebar.markdown("<h2 style='text-align: center; color: #00b4d8;'>🧬 Cancer Vision AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 14px;'>Clinical Cytology Analysis</p>", unsafe_allow_html=True)
st.sidebar.divider()

menu = st.sidebar.radio("SYSTEM MODULES", ["🔬 Diagnostic Engine", "🌌 Advanced Data Visualizer", "🗂️ Clinical Database"])

st.sidebar.divider()
st.sidebar.warning("⚠️ **System Notice:** For investigational use only. Not a substitute for histopathological diagnosis.")

# Feature definitions
features_mean = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_pts_mean', 'symmetry_mean', 'fractal_dim_mean']
features_se = ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_pts_se', 'symmetry_se', 'fractal_dim_se']
features_worst = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_pts_worst', 'symmetry_worst', 'fractal_dim_worst']

# Colors for Benign and Malignant
color_b = "#00b4d8" # Teal/Blue
color_m = "#ff3366" # Neon Pink/Red


if menu == "🔬 Diagnostic Engine":
    st.markdown("<h2>Breast Cytology Analysis Engine</h2>", unsafe_allow_html=True)
    st.markdown("Input quantitative cellular measurements extracted from the digitized fine needle aspirate (FNA) image.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Mean Features", "📉 Standard Errors", "📈 Worst Features"])

    def create_input_columns(feature_names):
        cols = st.columns(3) 
        input_data = {}
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                clean_label = feature.replace('_', ' ').title()
                default_val = float(data[feature].median()) if feature in data.columns else 0.0
                input_data[feature] = st.number_input(f"{clean_label}", value=default_val, format="%.4f")
        return input_data

    user_data = {}
    with tab1: user_data.update(create_input_columns(features_mean))
    with tab2: user_data.update(create_input_columns(features_se))
    with tab3: user_data.update(create_input_columns(features_worst))

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 INITIATE AI DIAGNOSTIC SCAN"):
        feature_order = features_mean + features_se + features_worst
        input_df = pd.DataFrame([user_data], columns=feature_order)
        scaled_input = scaler.transform(input_df)
        
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        
        st.divider()
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("<h3>Scan Results:</h3>", unsafe_allow_html=True)
            if prediction == 1: 
                st.error("### 🚨 DETECTED: Malignant Signature")
                st.markdown(f"**Confidence Score:** `{probability[1]*100:.2f}%`")
                st.markdown("> *Cellular anomaly detected. High structural irregularity consistent with malignancy. Immediate Cancer logy review recommended.*")
            else:               
                st.success("### ✅ DETECTED: Benign Signature")
                st.markdown(f"**Confidence Score:** `{probability[0]*100:.2f}%`")
                st.markdown("> *Cellular structure appears stable and regular. Morphology is consistent with benign tissue.*")
                
        with col2:
            conf_val = probability[1] * 100 if prediction == 1 else probability[0] * 100
            gauge_color = color_m if prediction == 1 else color_b
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = conf_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Certainty", 'font': {'color': '#e6edf3', 'size': 20}},
                number = {'font': {'color': '#e6edf3'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#e6edf3"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#304156",
                }
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e6edf3"})
            st.plotly_chart(fig, use_container_width=True)


elif menu == "🌌 Advanced Data Visualizer":
    st.markdown("<h2>Advanced Cellular Diagnostics</h2>", unsafe_allow_html=True)
    st.markdown("Interactive multi-dimensional analysis of the tumor database using stable Matplotlib rendering.")
    st.divider()
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 1. PCA Morphological Clustering")
        st.markdown("Principal Component Analysis (PCA) compresses all cellular features into two dimensions.")
        
        pca_data = data.dropna(subset=features_mean + ['Diagnosis_Label'])
        x_pca = pca_data[features_mean]
        scaler_pca = StandardScaler()
        x_scaled = scaler_pca.fit_transform(x_pca)
        pca = PCA(n_components=2)
        components = pca.fit_transform(x_scaled)
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=components[:,0], y=components[:,1], hue=pca_data['Diagnosis_Label'], 
                        palette={'Benign': color_b, 'Malignant': color_m}, alpha=0.8, edgecolor=None, ax=ax1)
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.grid(alpha=0.2, color='#304156')
        
        # Style for Dark Mode
        apply_dark_theme(fig1, ax1)
        leg = ax1.legend(title="Diagnosis")
        plt.setp(leg.get_title(), color='#e6edf3')
        plt.setp(leg.get_texts(), color='#e6edf3')
        leg.get_frame().set_facecolor('none')
        leg.get_frame().set_edgecolor('#304156')
        
        st.pyplot(fig1)

    with col_b:
        st.markdown("### 2. Tumor Density Topography")
        st.markdown("A topographical heatmap showing the concentration zones of tumors.")
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.kdeplot(data=data, x="radius_mean", y="texture_mean", hue="Diagnosis_Label", 
                    palette={'Benign': color_b, 'Malignant': color_m}, fill=True, alpha=0.5, thresh=0.05, ax=ax2)
        ax2.set_xlabel('Mean Radius')
        ax2.set_ylabel('Mean Texture')
        
        apply_dark_theme(fig2, ax2)
        leg = ax2.legend_
        if leg:
            plt.setp(leg.get_title(), color='#e6edf3')
            plt.setp(leg.get_texts(), color='#e6edf3')
            leg.get_frame().set_facecolor('none')
            leg.get_frame().set_edgecolor('#304156')
        
        st.pyplot(fig2)
    
    st.divider()
    
    # --- PLOT 3: Radar Chart ---
    st.markdown("### 3. Clinical Profile Blueprint (Radar Chart)")
    st.markdown("Comparing the average blueprint of a Benign vs Malignant cell across multiple features.")
    
    radar_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    labels = [f.replace('_', ' ').title() for f in radar_features] 
    
    mean_b = data[data['Diagnosis_Label'] == 'Benign'][radar_features].mean()
    mean_m = data[data['Diagnosis_Label'] == 'Malignant'][radar_features].mean()
    max_vals = data[radar_features].max()
    
    norm_b = (mean_b / max_vals).values.tolist()
    norm_m = (mean_m / max_vals).values.tolist()
    norm_b += norm_b[:1]
    norm_m += norm_m[:1]
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    angles += angles[:1]
    
    fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax3.plot(angles, norm_b, color=color_b, linewidth=2, label='Benign')
    ax3.fill(angles, norm_b, color=color_b, alpha=0.25)
    ax3.plot(angles, norm_m, color=color_m, linewidth=2, label='Malignant')
    ax3.fill(angles, norm_m, color=color_m, alpha=0.25)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels, fontsize=11)
    
    apply_dark_theme(fig3, ax3, is_polar=True)
    ax3.grid(color='#304156', alpha=0.5)
    ax3.spines['polar'].set_color('#304156')
    
    leg = ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.setp(leg.get_texts(), color='#e6edf3')
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('#304156')
    
    st.pyplot(fig3)

    st.divider()

    # --- PLOT 4: Correlation Matrix Heatmap ---
    st.markdown("### 4. Feature Correlation Matrix")
    st.markdown("Darker reds indicate features that grow simultaneously (e.g., Radius and Area).")
    
    corr_matrix = data[features_mean].corr()
    clean_feature_names = [f.replace('_', ' ').title() for f in features_mean]
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", 
                xticklabels=clean_feature_names, yticklabels=clean_feature_names, 
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax4)
    
    apply_dark_theme(fig4, ax4)
    plt.xticks(rotation=45, ha='right')
    
    # Heatmap specific dark theme fixes for colorbar
    cbar = ax4.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='#e6edf3')
    cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='#e6edf3')
    cbar.outline.set_edgecolor('#304156')
    
    st.pyplot(fig4)


elif menu == "🗂️ Clinical Database":
    st.markdown("<h2>Database Explorer</h2>", unsafe_allow_html=True)
    st.markdown("Raw data table from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patient Records", len(data))
    col2.metric("Malignant Cases", len(data[data['Diagnosis_Label'] == 'Malignant']))
    col3.metric("Benign Cases", len(data[data['Diagnosis_Label'] == 'Benign']))
    
    st.divider()
    
    display_df = data.drop(columns=['diagnosis'], errors='ignore').copy()
    display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]
    
    st.dataframe(display_df, use_container_width=True, height=600)