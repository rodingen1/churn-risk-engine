import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple

# --- Page Config ---
st.set_page_config(page_title="ChurnGuard AI", layout="wide", page_icon="🛡️")

# --- Custom CSS for Enterprise Look ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ ChurnGuard: Customer Retention Engine")
st.markdown("Ett AI-system som förutspår kundbortfall (**Churn**) och simulerar åtgärder för att rädda intäkter.")
st.markdown("---")

# --- 1. Data Generation (Simulerar en Telekom-databas) ---
@st.cache_data
def generate_synthetic_data(n_rows: int = 2000) -> pd.DataFrame:
    """
    Genererar realistisk kunddata för en telekom-operatör.
    Vi skapar mönster som AI:n kan hitta (t.ex. Högt pris + Kort kontrakt = Churn).
    """
    np.random.seed(42)
    
    # Skapa features
    data = pd.DataFrame({
        'CustomerID': range(1000, 1000 + n_rows),
        'CreditScore': np.random.randint(300, 850, n_rows),
        'Age': np.random.randint(18, 80, n_rows),
        'Tenure': np.random.randint(0, 72, n_rows), # Månader som kund
        'Balance': np.random.uniform(0, 250000, n_rows).round(2),
        'NumOfProducts': np.random.randint(1, 4, n_rows),
        'HasCrCard': np.random.randint(0, 2, n_rows),
        'IsActiveMember': np.random.randint(0, 2, n_rows),
        'EstimatedSalary': np.random.uniform(10000, 150000, n_rows).round(2),
        'MonthlyCharges': np.random.uniform(30, 120, n_rows).round(2),
        'SupportCalls': np.random.randint(0, 10, n_rows) # Antal klagomål
    })
    
    # --- Skapa Logic för Target (Churn) ---
    # Vi gör så att vissa kunder har högre risk, så AI:n kan lära sig.
    # Logic: Om kunden ringer support ofta, har höga avgifter och kort tid (Tenure), så lämnar de.
    risk_score = (
        (data['SupportCalls'] * 1.5) + 
        (data['MonthlyCharges'] / 20) - 
        (data['Tenure'] / 5) - 
        (data['IsActiveMember'] * 5)
    )
    # Sigmoid function för att få sannolikhet
    prob = 1 / (1 + np.exp(-(risk_score - 2))) # Justera threshold
    data['Churn'] = (np.random.rand(n_rows) < prob).astype(int)
    
    return data

# --- 2. Model Training (XGBoost) ---
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost är state-of-the-art för tabulär data
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=4, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    
    return model, acc, auc, X_test, y_test

# --- Load & Train ---
df = generate_synthetic_data()
model, acc, auc, X_test, y_test = train_model(df)

# --- UI: Dashboard ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Totala Kunder", f"{len(df)}")
col2.metric("Churn Rate (Verklig)", f"{df['Churn'].mean():.1%}")
col3.metric("AI Accuracy", f"{acc:.1%}", delta="High Precision")
col4.metric("ROC-AUC Score", f"{auc:.3f}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Management Overview", "🕹️ Kund-Simulator", "📋 Rådata"])

with tab1:
    st.subheader("Varför lämnar kunderna?")
    st.write("AI-modellen har analyserat vilka faktorer som driver churn.")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', 
                 title="Top Drivers for Customer Churn (XGBoost Analysis)")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insikt:** Som vi ser är 'SupportCalls' (Antal samtal till support) och 'Tenure' (Hur länge man varit kund) de starkaste signalerna. Nya kunder som ringer support ofta är i riskzonen.")

with tab2:
    st.subheader("🕹️ Churn Simulator: Rädda en kund")
    st.markdown("Justera parametrarna nedan för att se hur risken förändras i realtid. Detta verktyg används av kundtjänst för att ge rätt erbjudande.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown("### Kundprofil")
        val_calls = st.slider("Antal Support-samtal (Senaste året)", 0, 10, 5)
        val_charges = st.slider("Månadskostnad ($)", 30, 120, 90)
        val_tenure = st.slider("Tid som kund (Månader)", 0, 72, 6)
        val_age = st.slider("Ålder", 18, 80, 35)
        val_active = st.checkbox("Är aktiv medlem?", value=False)
        
        # Default values for others
        input_data = pd.DataFrame({
            'CreditScore': [650], 'Age': [val_age], 'Tenure': [val_tenure], 
            'Balance': [50000], 'NumOfProducts': [1], 'HasCrCard': [1], 
            'IsActiveMember': [int(val_active)], 'EstimatedSalary': [60000], 
            'MonthlyCharges': [val_charges], 'SupportCalls': [val_calls]
        })

    with col_result:
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.markdown("### AI Riskanalys")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk för Churn (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if probability > 0.7:
            st.error("⚠️ **HÖG RISK!** Denna kund är på väg att lämna.")
            st.markdown("**Rekommenderad Åtgärd:** Erbjud 20% rabatt eller prioritera supportärendet.")
        elif probability > 0.4:
            st.warning("⚖️ **MEDEL RISK.** Håll ögonen på denna kund.")
        else:
            st.success("✅ **LÅG RISK.** Nöjd kund.")

with tab3:
    st.dataframe(df.head(50))