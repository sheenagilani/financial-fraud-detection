import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import io

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Financial Fraud Detection & Risk Scoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark FinTech Theme, Glassmorphic Cards)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #3a506b 100%);
        border-radius: 18px;
        padding: 30px 36px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px -15px rgba(11, 19, 43, 0.7);
    }
    
    .hero-title {
        color: #ffffff;
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .project-name-badge {
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid rgba(129, 140, 248, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 820px;
        line-height: 1.55;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    /* Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        padding: 14px 32px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.5) !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px -5px rgba(124, 58, 237, 0.7) !important;
    }

    .anomaly-tag {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        line-height: 1.45;
        border-left: 4px solid;
    }
    
    .anomaly-danger {
        background: rgba(239, 68, 68, 0.12);
        border-color: #ef4444;
        color: #fca5a5;
    }

    .anomaly-warning {
        background: rgba(245, 158, 11, 0.12);
        border-color: #f59e0b;
        color: #fcd34d;
    }

    .anomaly-success {
        background: rgba(16, 185, 129, 0.12);
        border-color: #10b981;
        color: #6ee7b7;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Artifact
# ---------------------------------------------------------
@st.cache_resource
def load_fraud_model():
    try:
        model_payload = joblib.load("fraud_detection_model.pkl")
        return model_payload
    except Exception:
        return None

model_payload = load_fraud_model()

# ---------------------------------------------------------
# Sidebar: Engine Telemetry & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Telemetry")
    st.markdown("""
        <div style='background: rgba(15,23,42,0.6); padding: 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 16px;'>
            <div style='color: #10b981; font-weight: 700; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;'>
                <span style='height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block;'></span>
                STATUS: INFERENCE ONLINE
            </div>
            <div style='color: #cbd5e1; font-size: 0.8rem; margin-top: 6px;'><b>Model:</b> Balanced Ensemble Forest</div>
            <div style='color: #94a3b8; font-size: 0.78rem;'><b>Framework:</b> Scikit-Learn / Streamlit</div>
            <div style='color: #94a3b8; font-size: 0.78rem;'><b>Dataset:</b> PaySim (6.36M Records)</div>
        </div>
    """, unsafe_allow_html=True)

    if model_payload is not None:
        threshold = float(model_payload.get('threshold', 0.7199))
        st.write(f"🎯 **Calibrated Threshold:** `{threshold:.4f}`")
        st.caption("Derived on Validation PR-Curve to maximize F1-score without test set leakage.")
    else:
        threshold = 0.50
        st.warning("⚠️ `fraud_detection_model.pkl` not detected.")

    st.markdown("---")
    st.markdown("#### ⚡ Viva Quick Demo Presets")
    scenario = st.selectbox(
        "Select Test Scenario",
        [
            "Custom Transaction",
            "🔴 High-Risk Account Draining (Fraud)",
            "🟢 Legitimate Merchant Payment",
            "🟡 Large Normal Transfer",
            "🚨 Balance Tampering Discrepancy"
        ]
    )
    st.markdown("---")
    st.markdown("#### 🎓 Project Details")
    st.caption("**Project Title:** AI-Based Financial Fraud Detection & Risk Scoring System")
    st.caption("**Domain:** FinTech & Applied Machine Learning")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <div style="margin-bottom: 8px;">
            <span class="project-name-badge">FinTech ML Project</span>
        </div>
        <div class="hero-title">
            AI-Based Financial Fraud Detection & Risk Scoring System
        </div>
        <p class="hero-subtitle">
            An enterprise-grade intelligent transaction surveillance and risk scoring engine. Combines behavioral 
            discrepancy heuristics, account-emptying signals, and cost-calibrated decision thresholds to intercept illicit transactions.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Preset Logic
# ---------------------------------------------------------
if scenario == "🔴 High-Risk Account Draining (Fraud)":
    default_step = 14
    default_type = "TRANSFER"
    default_amt = 181500.00
    default_old_orig = 181500.00
    default_old_dest = 0.00
    default_auto = False
    default_new_orig = 0.00
    default_new_dest = 0.00
    default_merchant = False
elif scenario == "🟢 Legitimate Merchant Payment":
    default_step = 32
    default_type = "PAYMENT"
    default_amt = 85.50
    default_old_orig = 4200.00
    default_old_dest = 0.00
    default_auto = True
    default_new_orig = 4114.50
    default_new_dest = 0.00
    default_merchant = True
elif scenario == "🟡 Large Normal Transfer":
    default_step = 88
    default_type = "TRANSFER"
    default_amt = 15000.00
    default_old_orig = 95000.00
    default_old_dest = 12000.00
    default_auto = True
    default_new_orig = 80000.00
    default_new_dest = 27000.00
    default_merchant = False
elif scenario == "🚨 Balance Tampering Discrepancy":
    default_step = 45
    default_type = "CASH_OUT"
    default_amt = 75000.00
    default_old_orig = 100000.00
    default_old_dest = 5000.00
    default_auto = False
    default_new_orig = 90000.00
    default_new_dest = 5000.00
    default_merchant = False
else:
    default_step = 1
    default_type = "TRANSFER"
    default_amt = 1000.00
    default_old_orig = 5000.00
    default_old_dest = 1200.00
    default_auto = True
    default_new_orig = 4000.00
    default_new_dest = 2200.00
    default_merchant = False

# ---------------------------------------------------------
# Tabbed Layout for Interactive Exploration
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Real-Time Transaction Scoring",
    "📁 Batch CSV Transaction Auditor",
    "📈 Model Economics & Evaluation Metrics"
])

# =========================================================
# TAB 1: Single Transaction Live Scoring
# =========================================================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📋 Transaction Parameters")
        step = st.number_input(
            "Transaction Step (Hour of Month: 1 - 744)",
            min_value=1,
            max_value=744,
            value=default_step,
            help="Simulation clock in hours. 1 step = 1 hour (744 hours = 31 days)."
        )
        transaction_type = st.selectbox(
            "Transaction Route / Channel",
            ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"],
            index=["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"].index(default_type),
            help="High-risk corridors: TRANSFER and CASH_OUT."
        )
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            value=default_amt,
            step=500.0,
            format="%.2f"
        )
        is_merchant = st.checkbox(
            "Recipient is a Merchant Account (Identifier begins with 'M')",
            value=default_merchant,
            help="In the PaySim dataset, merchant accounts have zero reported balance history."
        )

    with col2:
        st.markdown("### 💳 Account Balance Telemetry")
        oldbalanceOrg = st.number_input(
            "Sender Initial Balance ($)",
            min_value=0.0,
            value=default_old_orig,
            step=500.0,
            format="%.2f"
        )
        oldbalanceDest = st.number_input(
            "Recipient Initial Balance ($)",
            min_value=0.0,
            value=default_old_dest,
            step=500.0,
            format="%.2f"
        )

        auto_calc = st.checkbox(
            "⚡ Auto-compute End Balances (Standard Accounting)",
            value=default_auto,
            help="Uncheck this box to test anomalies such as account-draining ($0 new balance) or balance discrepancies."
        )

        if auto_calc:
            if transaction_type == "CASH_IN":
                newbalanceOrig = oldbalanceOrg + amount
                newbalanceDest = max(0.0, oldbalanceDest - amount)
            else:
                newbalanceOrig = max(0.0, oldbalanceOrg - amount)
                newbalanceDest = oldbalanceDest + amount if not is_merchant else 0.0

            st.markdown(f"""
                <div style='background: rgba(30,41,59,0.45); padding: 12px 16px; border-radius: 10px; border: 1px dashed rgba(255,255,255,0.15); margin-top: 10px;'>
                    <div style='color: #cbd5e1; font-size: 0.88rem;'>Calculated Sender End Balance: <b style='color:#38bdf8;'>${newbalanceOrig:,.2f}</b></div>
                    <div style='color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;'>Calculated Recipient End Balance: <b style='color:#38bdf8;'>${newbalanceDest:,.2f}</b></div>
                </div>
            """, unsafe_allow_html=True)
        else:
            newbalanceOrig = st.number_input(
                "Sender Post-Transaction Balance ($)",
                min_value=0.0,
                value=default_new_orig,
                step=500.0,
                format="%.2f",
                help="Set to $0.00 to simulate full fund extraction."
            )
            newbalanceDest = st.number_input(
                "Recipient Post-Transaction Balance ($)",
                min_value=0.0,
                value=default_new_dest,
                step=500.0,
                format="%.2f",
                help="Set to $0.00 to simulate mule account immediate siphon."
            )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 Analyze Transaction & Compute Risk Score")

    if analyze_clicked:
        if model_payload is None:
            st.error("Cannot proceed: `fraud_detection_model.pkl` is missing.")
            st.stop()

        # Visual progress animation
        progress_bar = st.progress(0, text="Interrogating neural pathways & tree ensembles...")
        for pct in [25, 60, 90, 100]:
            time.sleep(0.07)
            progress_bar.progress(pct, text=f"Evaluating behavioral rules ({pct}%)...")
        time.sleep(0.05)
        progress_bar.empty()

        # Feature Engineering
        orig_balance_diff = oldbalanceOrg - newbalanceOrig
        dest_balance_diff = newbalanceDest - oldbalanceDest
        orig_balance_err = (oldbalanceOrg - amount) - newbalanceOrig
        dest_balance_err = (oldbalanceDest + amount) - newbalanceDest
        orig_emptied = int((oldbalanceOrg > 0) and (newbalanceOrig == 0))
        is_merchant_dest = int(is_merchant)

        input_dict = {
            'step': step,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'orig_balance_diff': orig_balance_diff,
            'dest_balance_diff': dest_balance_diff,
            'orig_balance_err': orig_balance_err,
            'dest_balance_err': dest_balance_err,
            'orig_emptied': orig_emptied,
            'is_merchant_dest': is_merchant_dest,
            'type_CASH_IN': int(transaction_type == "CASH_IN"),
            'type_CASH_OUT': int(transaction_type == "CASH_OUT"),
            'type_DEBIT': int(transaction_type == "DEBIT"),
            'type_PAYMENT': int(transaction_type == "PAYMENT"),
            'type_TRANSFER': int(transaction_type == "TRANSFER")
        }

        model = model_payload['model']
        features = model_payload['features']
        input_data = pd.DataFrame([input_dict])
        used_features = [f for f in features if f in input_data.columns]
        input_data = input_data[used_features]

        # Model Probability
        raw_prob = float(model.predict_proba(input_data)[0][1])
        risk_score_pct = raw_prob * 100.0
        is_flagged = raw_prob >= threshold

        # Risk Tier Classification
        if risk_score_pct >= 80:
            risk_level = "CRITICAL"
            risk_color = "#ef4444"
            action = "⛔ BLOCK IMMEDIATELY"
            action_desc = "Auto-freeze funds & originate Tier-1 security incident."
            status_label = "MALICIOUS / FRAUD"
            status_color = "#ef4444"
        elif risk_score_pct >= 50:
            risk_level = "HIGH"
            risk_color = "#f97316"
            action = "⚠️ STEP-UP AUTH (OTP / Biometric)"
            action_desc = "Demand biometric passkey or out-of-band SMS OTP."
            status_label = "HIGHLY SUSPICIOUS"
            status_color = "#f97316"
        elif risk_score_pct >= 20:
            risk_level = "MEDIUM"
            risk_color = "#eab308"
            action = "🔍 QUEUE MANUAL REVIEW"
            action_desc = "Flag for secondary review; do not disrupt consumer checkout."
            status_label = "ELEVATED RISK"
            status_color = "#eab308"
        else:
            risk_level = "LOW"
            risk_color = "#10b981"
            action = "✅ AUTO-APPROVE"
            action_desc = "Transaction complies with normal historical distribution."
            status_label = "CLEAN / LEGITIMATE"
            status_color = "#10b981"

        # Display Metrics
        st.markdown("### 📊 Operational Assessment & Decision Matrix")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Fraud Probability</div>
                    <div class="metric-value" style="color: {risk_color};">{risk_score_pct:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Transaction Status</div>
                    <div class="metric-value" style="color: {status_color}; font-size: 1.35rem;">{status_label}</div>
                </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Recommended Action</div>
                    <div class="metric-value" style="font-size: 1.12rem; color: #f8fafc;">{action}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if is_flagged:
            st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                    <h4 style="color: #fca5a5; margin: 0 0 6px 0; font-weight: 700;">🚨 Security Intercept Protocol Triggered</h4>
                    <p style="color: #fecaca; margin: 0; font-size: 0.93rem;">
                        The predicted fraud probability (<b>{raw_prob:.4f}</b>) exceeds the operational threshold (<b>{threshold:.4f}</b>). 
                        Required Protocol: <b>{action_desc}</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                    <h4 style="color: #6ee7b7; margin: 0 0 6px 0; font-weight: 700;">✅ Transaction Cleared</h4>
                    <p style="color: #a7f3d0; margin: 0; font-size: 0.93rem;">
                        Transaction risk (<b>{raw_prob:.4f}</b>) remains below the fraud threshold (<b>{threshold:.4f}</b>). 
                        Protocol: <b>{action_desc}</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Anomaly Explanations
        st.markdown("### 🔎 Forensic Rule & Anomaly Diagnostics")
        c_diag1, c_diag2 = st.columns([1, 1], gap="medium")

        with c_diag1:
            st.markdown("#### Behavioral Heuristics")
            if orig_emptied:
                st.markdown("""
                    <div class="anomaly-tag anomaly-danger">
                        <span>🚨</span>
                        <div><b>Origin Account Drained:</b> Balance reduced to $0.00, matching rapid mule fund liquidation.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="anomaly-tag anomaly-success">
                        <span>✔️</span>
                        <div><b>Balance Retention:</b> Origin account retains residual liquidity.</div>
                    </div>
                """, unsafe_allow_html=True)

            if transaction_type in ["TRANSFER", "CASH_OUT"]:
                st.markdown(f"""
                    <div class="anomaly-tag anomaly-warning">
                        <span>⚠️</span>
                        <div><b>High-Risk Channel ({transaction_type}):</b> Over 99% of PaySim fraud operations exploit this corridor.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="anomaly-tag anomaly-success">
                        <span>✔️</span>
                        <div><b>Low-Risk Corridor ({transaction_type}):</b> Negligible historical fraud occurrence.</div>
                    </div>
                """, unsafe_allow_html=True)

        with c_diag2:
            st.markdown("#### Accounting Discrepancy Diagnostics")
            if abs(orig_balance_err) > 0.01:
                st.markdown(f"""
                    <div class="anomaly-tag anomaly-danger">
                        <span>🚨</span>
                        <div><b>Origin Discrepancy:</b> Mismatch of <b>${orig_balance_err:,.2f}</b> between debit amount and balance delta.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="anomaly-tag anomaly-success">
                        <span>✔️</span>
                        <div><b>Origin Math Balanced:</b> Debited funds exactly match the balance delta.</div>
                    </div>
                """, unsafe_allow_html=True)

            if abs(dest_balance_err) > 0.01 and not is_merchant:
                st.markdown("""
                    <div class="anomaly-tag anomaly-warning">
                        <span>⚠️</span>
                        <div><b>Destination Credit Discrepancy:</b> Recipient balance did not register the transaction amount.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="anomaly-tag anomaly-success">
                        <span>✔️</span>
                        <div><b>Destination Integrity Valid:</b> Destination balance matches credit expectations.</div>
                    </div>
                """, unsafe_allow_html=True)

        # Downloadable Audit Summary
        audit_report = f"""=====================================================
AI-BASED FINANCIAL FRAUD DETECTION & RISK SCORING SYSTEM
AUDIT & COMPLIANCE EVALUATION REPORT
=====================================================
Timestamp (Step): Hour {step}
Transaction Type: {transaction_type}
Transaction Amount: ${amount:,.2f}

Origin Account:
  - Old Balance: ${oldbalanceOrg:,.2f}
  - New Balance: ${newbalanceOrig:,.2f}
  - Balance Discrepancy: ${orig_balance_err:,.2f}
  - Account Completely Drained: {'YES' if orig_emptied else 'NO'}

Destination Account:
  - Old Balance: ${oldbalanceDest:,.2f}
  - New Balance: ${newbalanceDest:,.2f}
  - Is Merchant: {'YES' if is_merchant else 'NO'}

ML Evaluation & Risk Scoring:
  - Fraud Probability: {risk_score_pct:.2f}% (Raw: {raw_prob:.4f})
  - Calibrated Decision Threshold: {threshold:.4f}
  - Operational Risk Level: {risk_level}
  - Transaction Status: {status_label}
  - Recommended Operational Action: {action}
  - Protocol Note: {action_desc}
=====================================================
"""
        st.download_button(
            label="📥 Download Compliance Audit Report (.txt)",
            data=audit_report,
            file_name=f"fraud_audit_report_step{step}_{transaction_type}.txt",
            mime="text/plain"
        )

# =========================================================
# TAB 2: Batch CSV Transaction Auditor
# =========================================================
with tab2:
    st.markdown("### 📁 Batch Transaction Auditing Engine")
    st.write(
        "Upload a batch CSV file of transactions (e.g. daily payment logs) to score all transactions, "
        "detect high-risk patterns, and export flagged records for AML (Anti-Money Laundering) compliance."
    )

    uploaded_file = st.file_uploader("Upload Transaction Batch (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df):,} records successfully!")
            st.dataframe(batch_df.head(5))

            if st.button("⚡ Score Batch Transactions"):
                with st.spinner("Scoring batch transactions..."):
                    b_df = batch_df.copy()
                    if 'amount' in b_df.columns and 'oldbalanceOrg' in b_df.columns and 'newbalanceOrig' in b_df.columns:
                        b_df['orig_balance_diff'] = b_df['oldbalanceOrg'] - b_df['newbalanceOrig']
                        b_df['dest_balance_diff'] = b_df['newbalanceDest'] - b_df['oldbalanceDest']
                        b_df['orig_balance_err'] = (b_df['oldbalanceOrg'] - b_df['amount']) - b_df['newbalanceOrig']
                        b_df['dest_balance_err'] = (b_df['oldbalanceDest'] + b_df['amount']) - b_df['newbalanceDest']
                        b_df['orig_emptied'] = ((b_df['oldbalanceOrg'] > 0) & (b_df['newbalanceOrig'] == 0)).astype(int)
                        
                        if 'nameDest' in b_df.columns:
                            b_df['is_merchant_dest'] = b_df['nameDest'].astype(str).str.startswith('M').astype(int)
                        else:
                            b_df['is_merchant_dest'] = 0

                        for t in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
                            b_df[f'type_{t}'] = (b_df['type'] == t).astype(int)

                        model = model_payload['model']
                        features = model_payload['features']
                        used_cols = [c for c in features if c in b_df.columns]
                        
                        batch_probs = model.predict_proba(b_df[used_cols])[:, 1]
                        batch_df['Fraud_Probability (%)'] = np.round(batch_probs * 100.0, 2)
                        batch_df['Flagged_Suspicious'] = (batch_probs >= threshold).astype(int)
                        batch_df['Risk_Level'] = pd.cut(
                            batch_df['Fraud_Probability (%)'],
                            bins=[-1, 20, 50, 80, 101],
                            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                        )

                        flagged_count = batch_df['Flagged_Suspicious'].sum()
                        st.markdown(f"#### Batch Summary: Found **{flagged_count:,}** flagged suspicious transactions out of **{len(batch_df):,}** ({flagged_count/len(batch_df)*100:.2f}%)")
                        
                        st.dataframe(batch_df[['step', 'type', 'amount', 'Fraud_Probability (%)', 'Risk_Level', 'Flagged_Suspicious']].head(20))

                        csv_buf = io.StringIO()
                        batch_df.to_csv(csv_buf, index=False)
                        st.download_button(
                            label="📥 Download Scored Batch CSV",
                            data=csv_buf.getvalue(),
                            file_name="scored_fraud_transactions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Uploaded CSV must contain columns: 'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# =========================================================
# TAB 3: Model Economics & Evaluation Dashboard
# =========================================================
with tab3:
    st.markdown("### 📈 Evaluation Benchmarks & Business Cost-Benefit Engine")
    st.write(
        "Demonstrates why classical accuracy is misleading in imbalanced fraud detection "
        "and illustrates the trade-off between fraud losses saved vs. false alarm review costs."
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### ⚖️ The Accuracy Paradox in FinTech")
        st.markdown("""
        In the PaySim financial dataset:
        - **Legitimate Transactions (0):** 99.871% (6,354,407 records)
        - **Fraudulent Transactions (1):** 0.129% (8,213 records)
        
        *If a naive classifier simply predicts 'Legitimate' 100% of the time, its accuracy is **99.87%**, but it intercepts **$0** of theft.*
        
        **True Key Performance Indicators:**
        - **Precision:** Reduces false customer alarms and customer friction.
        - **Recall (Sensitivity):** Maximizes the capture of illegal fund draining.
        - **F1-Score / PR-AUC:** The harmonic balance of precision and recall.
        """)

    with t_col2:
        st.markdown("#### 💰 Financial Cost-Benefit Simulator")
        avg_fraud_val = st.slider("Average Fraud Amount Intercepted ($)", 1000, 200000, 45000, step=5000)
        review_cost = st.slider("Manual Review Cost per Alert ($)", 5, 200, 25, step=5)
        monthly_tx = st.number_input("Monthly Platform Volume (Transactions)", value=500000, step=50000)

        # Simulation
        expected_fraud_rate = 0.0013
        est_frauds = int(monthly_tx * expected_fraud_rate)
        
        sim_recall = 0.88
        sim_precision = 0.85
        
        caught_frauds = int(est_frauds * sim_recall)
        fraud_losses_saved = caught_frauds * avg_fraud_val
        total_alerts = int(caught_frauds / sim_precision)
        false_alarms = total_alerts - caught_frauds
        investigation_costs = total_alerts * review_cost
        net_roi = fraud_losses_saved - investigation_costs

        st.metric("Net Fraud Loss Savings / Month", f"${net_roi:,.2f}")
        st.caption(f"Estimated Fraud Captured: **{caught_frauds:,}** / {est_frauds:,} | False Positives: **{false_alarms:,}**")