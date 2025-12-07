import sys
from pathlib import Path

# Make sure the project root (which contains `src/`) is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing.parse_json import parse_patient_bundle
from src.preprocessing.build_features import build_feature_table_for_bundle
from src.preprocessing.clustering_transform import add_cluster_and_align_for_models
from src.models.inference import load_all_models, predict_all_diseases


# ============================================================
# Streamlit config
# ============================================================

st.set_page_config(
    page_title="Lab Abnormalities as Predictors",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# Constants: lab info, disease display, cluster descriptions
# ============================================================

LAB_INFO: Dict[str, Dict[str, Any]] = {
    "creatinine_latest": {
        "name": "Creatinine",
        "unit": "mg/dL",
        "low": 0.4,
        "high": 1.3,
    },
    "egfr_latest": {
        "name": "eGFR",
        "unit": "mL/min/1.73m²",
        "low": 60,
        "high": 999,
    },
    "glucose_latest": {
        "name": "Glucose",
        "unit": "mg/dL",
        "low": 70,
        "high": 140,
    },
    "hba1c_latest": {
        "name": "HbA1c",
        "unit": "%",
        "low": 4.0,
        "high": 5.6,
    },
    "ldl_latest": {
        "name": "LDL cholesterol",
        "unit": "mg/dL",
        "low": 0,
        "high": 130,
    },
    "hdl_latest": {
        "name": "HDL cholesterol",
        "unit": "mg/dL",
        "low": 40,
        "high": 999,
    },
    "triglycerides_latest": {
        "name": "Triglycerides",
        "unit": "mg/dL",
        "low": 0,
        "high": 150,
    },
    "hemoglobin_latest": {
        "name": "Hemoglobin",
        "unit": "g/dL",
        "low": 12,
        "high": 17,
    },
    "hematocrit_latest": {
        "name": "Hematocrit",
        "unit": "%",
        "low": 36,
        "high": 50,
    },
    "bun_latest": {
        "name": "BUN",
        "unit": "mg/dL",
        "low": 7,
        "high": 25,
    },
    "albumin_latest": {
        "name": "Albumin",
        "unit": "g/dL",
        "low": 3.5,
        "high": 5.0,
    },
    "ast_latest": {
        "name": "AST",
        "unit": "U/L",
        "low": 0,
        "high": 40,
    },
    "alt_latest": {
        "name": "ALT",
        "unit": "U/L",
        "low": 0,
        "high": 40,
    },
}

DISEASE_DISPLAY = {
    "ckd": "Chronic Kidney Disease (CKD)",
    "cvd": "Cardiovascular Disease (CVD)",
    "anemia": "Anemia",
    "predm": "Prediabetes",
}

CLUSTER_DESCRIPTIONS = {
    0: "Cluster 0 – Younger group with generally healthier lab profile.",
    1: "Cluster 1 – Metabolic & cholesterol risk group (higher lipids / prediabetes tendency).",
    2: "Cluster 2 – Older mixed-risk group with several moderate abnormalities.",
    3: "Cluster 3 – Older group with higher chronic disease burden.",
}


# Codes we might use to fetch height/weight from df_obs
HEIGHT_NAMES = {
    "Body height",
    "Body height (measured)",
}
WEIGHT_NAMES = {
    "Body weight",
    "Body weight (measured)",
}


# ============================================================
# Cached model loading
# ============================================================

@st.cache_resource
def get_models():
    return load_all_models()


# ============================================================
# Helper functions
# ============================================================

def classify_risk(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.5:
        return "Moderate"
    else:
        return "High"


def status_from_lab_value(key: str, value: float) -> str:
    info = LAB_INFO.get(key)
    if info is None or pd.isna(value):
        return "Unknown"

    low = info["low"]
    high = info["high"]

    if value < low * 0.9:
        return "Low"
    elif value < low:
        return "Borderline low"
    elif value <= high:
        return "In range"
    elif value <= high * 1.2:
        return "Borderline high"
    else:
        return "High"


def status_emoji(status: str) -> str:
    if status.startswith("In range"):
        return "🟢"
    if "Borderline" in status:
        return "🟡"
    if status in ("High", "Low"):
        return "🔴"
    return "⚪️"


def generate_doctor_questions(
    risks: Dict[str, Dict[str, Any]],
    latest_labs: pd.Series,
) -> List[str]:
    qs: List[str] = []

    # Prediabetes / glucose
    predm_risk = risks.get("predm", {}).get("prob", 0)
    hba1c = latest_labs.get("hba1c_latest")
    if (predm_risk >= 0.5) or (pd.notna(hba1c) and hba1c >= 5.7):
        qs.append(
            "Could we review my blood sugar control and decide whether I should be monitored "
            "for prediabetes or diabetes?"
        )

    # CVD / cholesterol
    cvd_risk = risks.get("cvd", {}).get("prob", 0)
    ldl = latest_labs.get("ldl_latest")
    if (cvd_risk >= 0.5) or (pd.notna(ldl) and ldl >= 130):
        qs.append(
            "Can we discuss my cholesterol levels and overall cardiovascular risk, including "
            "whether lifestyle changes or medications are appropriate?"
        )

    # CKD / creatinine / eGFR
    ckd_risk = risks.get("ckd", {}).get("prob", 0)
    creat = latest_labs.get("creatinine_latest")
    egfr = latest_labs.get("egfr_latest")
    if (ckd_risk >= 0.5) or (pd.notna(creat) and creat > 1.3) or (pd.notna(egfr) and egfr < 60):
        qs.append(
            "Can we talk about my kidney function tests (creatinine, eGFR) and whether I need "
            "regular monitoring for kidney disease?"
        )

    # Anemia / hemoglobin
    anemia_risk = risks.get("anemia", {}).get("prob", 0)
    hgb = latest_labs.get("hemoglobin_latest")
    if (anemia_risk >= 0.5) or (pd.notna(hgb) and hgb < 12):
        qs.append(
            "Could we go over my blood counts and check whether anemia or other blood problems "
            "might be present?"
        )

    return qs


def get_sex_display(latest_features: pd.Series, df_patients: pd.DataFrame) -> str:
    # Try from patients table first
    if "gender" in df_patients.columns and not df_patients.empty:
        g = str(df_patients["gender"].iloc[0]).strip().upper()
        if g.startswith("M"):
            return "Male"
        if g.startswith("F"):
            return "Female"

    # Fall back to numeric in latest_features.sex
    sex_val = latest_features.get("sex")
    try:
        if sex_val == 1 or sex_val == "1":
            return "Male"
        if sex_val == 0 or sex_val == "0":
            return "Female"
    except Exception:
        pass

    # Last fallback
    return "Unknown"


def get_height_weight(df_obs: pd.DataFrame) -> (Optional[float], Optional[float]):
    height_val = None
    weight_val = None

    if df_obs.empty:
        return height_val, weight_val

    # Make sure effective_datetime is datetime for sorting
    if "effective_datetime" in df_obs.columns and not pd.api.types.is_datetime64_any_dtype(
        df_obs["effective_datetime"]
    ):
        df_obs = df_obs.copy()
        df_obs["effective_datetime"] = pd.to_datetime(
            df_obs["effective_datetime"], errors="coerce"
        )

    # Height
    df_h = df_obs[df_obs["code_display"].isin(HEIGHT_NAMES)].copy()
    if not df_h.empty and "effective_datetime" in df_h.columns:
        df_h = df_h.sort_values("effective_datetime", ascending=False)
        height_val = df_h["value_quantity"].iloc[0]

    # Weight
    df_w = df_obs[df_obs["code_display"].isin(WEIGHT_NAMES)].copy()
    if not df_w.empty and "effective_datetime" in df_w.columns:
        df_w = df_w.sort_values("effective_datetime", ascending=False)
        weight_val = df_w["value_quantity"].iloc[0]

    return height_val, weight_val


def build_lab_summary(latest_features: pd.Series) -> pd.DataFrame:
    rows = []
    for key, info in LAB_INFO.items():
        if key in latest_features.index:
            val = latest_features.get(key)
            status = status_from_lab_value(key, val)
            emoji = status_emoji(status)
            ref_range = f"{info['low']} – {info['high']} {info['unit']}"
            val_str = f"{val:.2f} {info['unit']}" if pd.notna(val) else "Missing"
            rows.append(
                {
                    "Lab": info["name"],
                    "Your result": val_str,
                    "Reference range": ref_range,
                    "Status": f"{emoji} {status}",
                    "raw_status": status,
                    "lab_key": key,
                    "numeric_value": val,
                }
            )
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(
            columns=["Lab", "Your result", "Reference range", "Status", "raw_status", "lab_key", "numeric_value"]
        )


def run_full_pipeline(bundle_json: Dict[str, Any]):
    df_patients, df_obs, df_conditions = parse_patient_bundle(bundle_json)
    feature_table = build_feature_table_for_bundle(df_patients, df_obs, reference_date="2025-01-01")
    ft_with_cluster, X_ready = add_cluster_and_align_for_models(feature_table)

    models = get_models()
    risk_results = predict_all_diseases(X_ready, models, threshold=0.5)

    latest_features = ft_with_cluster.iloc[0]

    return {
        "df_patients": df_patients,
        "df_obs": df_obs,
        "df_conditions": df_conditions,
        "feature_table": feature_table,
        "ft_with_cluster": ft_with_cluster,
        "X_ready": X_ready,
        "risks": risk_results,
        "latest_features": latest_features,
    }


# ============================================================
# Sidebar navigation
# ============================================================

st.sidebar.title("🧪 Lab Abnormalities as Predictors")

page = st.sidebar.radio(
    "Navigation",
    ["Upload", "Report Analysis", "Recommendations"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disclaimer:**\n\n"
    "This tool provides **educational estimates** based on lab values. "
    "It is **not** a diagnosis and does **not** replace advice from your doctor."
)


# ============================================================
# Page 1 — Upload
# ============================================================

if page == "Upload":
    st.title("Lab Abnormalities as Predictors for Chronic Diseases")

    st.markdown(
        "This website analyzes your lab test results and estimates the likelihood of four chronic conditions:\n\n"
        "- Chronic kidney disease (CKD)\n"
        "- Cardiovascular disease (CVD)\n"
        "- Anemia\n"
        "- Prediabetes\n\n"
        "It uses machine-learning models trained on synthetic patient data to provide **educational insight** "
        "into your lab profile."
    )

    st.markdown(
        "⚠️ **Important:** This tool does **not** provide medical diagnoses. "
        "Always review your results with a licensed healthcare professional."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your lab report (FHIR JSON)", type=["json"])

    if uploaded_file is not None:
        try:
            bundle_json = json.load(uploaded_file)
            st.success(
                "File uploaded successfully. Use the sidebar to open **Report Analysis** "
                "and **Recommendations** for this report."
            )
            st.session_state["bundle_json"] = bundle_json
        except Exception as e:
            st.error(f"Could not read JSON: {e}")

    st.stop()


# ============================================================
# For other pages, require uploaded JSON
# ============================================================

if "bundle_json" not in st.session_state:
    st.info("Please go to the **Upload** page and upload a JSON file first.")
    st.stop()

bundle_json = st.session_state["bundle_json"]

with st.spinner("Processing your data..."):
    result = run_full_pipeline(bundle_json)

df_patients = result["df_patients"]
df_obs = result["df_obs"]
feature_table = result["feature_table"]
ft_with_cluster = result["ft_with_cluster"]
risks = result["risks"]
latest_features = result["latest_features"]
cluster_id = int(ft_with_cluster["cluster"].iloc[0])


# ============================================================
# Page 2 — Report Analysis
# ============================================================

if page == "Report Analysis":
    st.header("Report Analysis")

    # 2.1 Patient KPIs
    st.subheader("Patient snapshot")

    age = latest_features.get("age")
    sex_display = get_sex_display(latest_features, df_patients)
    height_val, weight_val = get_height_weight(df_obs)
    total_labs = df_obs.shape[0]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Age", f"{int(age)} years" if pd.notna(age) else "Unknown")

    with col2:
        st.metric("Sex", sex_display)

    with col3:
        cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_id, f"Cluster {cluster_id}")
        st.metric("Cluster", f"{cluster_id}")
        st.caption(cluster_desc)

    with col4:
        if height_val is not None:
            st.metric("Height", f"{height_val:.1f} cm")
        else:
            st.metric("Height", "Not available")

    with col5:
        if weight_val is not None:
            st.metric("Weight", f"{weight_val:.1f} kg")
        else:
            st.metric("Weight", "Not available")

    st.caption(f"Total lab measurements in this file: **{total_labs}**")

    st.markdown("---")

    # 2.2 Disease risk gauges (rings)
    st.subheader("Estimated disease risks (snapshot)"

    )

    gauge_cols = st.columns(4)
    disease_keys = ["ckd", "cvd", "anemia", "predm"]

    for i, dkey in enumerate(disease_keys):
        prob = risks[dkey]["prob"]
        risk_cat = classify_risk(prob)

        with gauge_cols[i]:
            # Simple donut gauge using matplotlib
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            ax.pie(
                [prob, 1 - prob],
                colors=["#ff6b6b", "#f0f0f0"],
                startangle=90,
                counterclock=False,
                wedgeprops=dict(width=0.3),
            )
            ax.set(aspect="equal")
            ax.text(
                0, 0, f"{prob*100:.0f}%",
                ha="center", va="center", fontsize=14, fontweight="bold"
            )
            ax.set_title(DISEASE_DISPLAY[dkey], fontsize=9)
            st.pyplot(fig)

            st.write(f"**{risk_cat} risk**")
            st.caption("Model estimate based on your age and lab values.")

    st.caption(
        "These percentages are **model-based estimates**, not diagnoses. "
        "Please discuss them with your healthcare provider."
    )

    st.markdown("---")

    # 2.3 Lab-by-lab overview (with ranges + emojis)
    st.subheader("Lab-by-lab overview (latest values)")

    lab_summary_df = build_lab_summary(latest_features)

    if not lab_summary_df.empty:
        # Show user-friendly subset of columns
        show_df = lab_summary_df[["Lab", "Your result", "Reference range", "Status"]].copy()
        st.dataframe(show_df, use_container_width=True)
    else:
        st.write("No mapped lab values found in this file.")

    # 2.4 Key labs (visual only, no range shading)
    st.subheader("Key labs that need attention")

    if not lab_summary_df.empty:
        # Sort by severity: High/Low > Borderline > In range
        priority_order = {
            "High": 0,
            "Low": 0,
            "Borderline high": 1,
            "Borderline low": 1,
            "In range": 2,
            "Unknown": 3,
        }

        lab_summary_df["severity_rank"] = lab_summary_df["raw_status"].map(
            lambda s: priority_order.get(s, 3)
        )

        abnormal_df = lab_summary_df[lab_summary_df["severity_rank"] < 2].copy()
        abnormal_df = abnormal_df.sort_values("severity_rank")

        # If no clearly abnormal labs, we can fall back to borderline
        if abnormal_df.empty:
            borderline_df = lab_summary_df[
                lab_summary_df["severity_rank"] == 1
            ].copy()
            abnormal_df = borderline_df

        # Take up to 6 labs
        abnormal_df = abnormal_df.head(6)

        if not abnormal_df.empty:
            labels = abnormal_df["Lab"].tolist()
            values = abnormal_df["numeric_value"].tolist()
            statuses = abnormal_df["raw_status"].tolist()

            # Color by severity for a visually meaningful chart
            color_map = {
                "High": "#e45756",           # red
                "Low": "#4c78a8",            # blue
                "Borderline high": "#f58518",
                "Borderline low": "#72b7b2",
            }
            colors = [color_map.get(s, "#54a24b") for s in statuses]

            fig, ax = plt.subplots(figsize=(7, 4))
            bar_positions = range(len(labels))

            bars = ax.bar(bar_positions, values, color=colors, edgecolor="black", linewidth=0.7)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("Value")

            # Make it look cleaner
            ax.set_title("Key labs that need attention", fontsize=14, fontweight="bold")
            ax.yaxis.grid(True, linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

            # Remove top/right spines for a modern look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                if pd.notna(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            fig.tight_layout()
            st.pyplot(fig)
            st.caption(
                "Each bar shows one key lab result. Colors reflect how far outside the typical range the "
                "value appears to be (red/blue = clearly high or low, warm colors = borderline). "
                "Use this view to see at a glance which tests may be worth discussing with your doctor."
            )
        else:
            st.write("No labs appear clearly out of the typical range.")
    else:
        st.write("No labs available for this visualization.")

    st.markdown("---")

    # 2.5 Lab trends over time (embedded)
    st.subheader("Lab trends over time")

    if df_obs.empty:
        st.write("No observations found in this file.")
    else:
        df_obs_num = df_obs.dropna(subset=["value_quantity"]).copy()
        if df_obs_num.empty:
            st.write("No numeric lab values available for trending.")
        else:
            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(df_obs_num["effective_datetime"]):
                df_obs_num["effective_datetime"] = pd.to_datetime(
                    df_obs_num["effective_datetime"], errors="coerce"
                )

            # Build choices *based on what this patient actually has*
            available_tests = (
                df_obs_num["code_display"].dropna().value_counts().index.tolist()
            )

            lab_display = st.selectbox(
                "Choose a lab test to view over time",
                available_tests,
            )

            df_lab = df_obs_num[df_obs_num["code_display"] == lab_display].copy()
            df_lab = df_lab.sort_values("effective_datetime")

            time_window = st.radio(
                "Show data from:",
                ["All time", "Last 1 year", "Last 6 months"],
                horizontal=True,
            )

            if time_window != "All time":
                last_date = df_lab["effective_datetime"].max()
                if pd.notna(last_date):
                    if time_window == "Last 1 year":
                        cutoff = last_date - pd.DateOffset(years=1)
                    else:
                        cutoff = last_date - pd.DateOffset(months=6)
                    df_lab = df_lab[df_lab["effective_datetime"] >= cutoff]

            if df_lab.shape[0] == 0:
                st.write("No values in the selected time window.")
            else:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(df_lab["effective_datetime"], df_lab["value_quantity"], marker="o")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                plt.xticks(rotation=25, ha="right")
                st.pyplot(fig)

                if df_lab.shape[0] == 1:
                    st.info("Only one value available for this test in the uploaded file.")
                else:
                    first_val = df_lab["value_quantity"].iloc[0]
                    last_val = df_lab["value_quantity"].iloc[-1]
                    diff = last_val - first_val
                    st.write(
                        f"Change from first to last value in this view: "
                        f"{diff:+.2f} units."
                    )


# ============================================================
# Page 3 — Recommendations
# ============================================================

elif page == "Recommendations":
    st.header("Recommendations")

    st.markdown(
        "These suggestions are generated from your cluster, lab values, and risk estimates. "
        "They are meant to **support discussions** with your doctor, not replace them."
    )

    # 3.2 Cluster-based recommendations
    st.subheader("Based on your cluster")

    cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_id, f"Cluster {cluster_id}")
    st.write(f"You are in: **{cluster_desc}**")

    if cluster_id == 0:
        st.markdown(
            "- Your lab profile is similar to a relatively healthy group in our dataset.\n"
            "- It can still be useful to keep monitoring your key labs periodically.\n"
            "- Focus on maintaining healthy habits (diet, exercise, regular check-ups)."
        )
    elif cluster_id == 1:
        st.markdown(
            "- Your profile is similar to a group with higher cholesterol and prediabetes risk.\n"
            "- It may be especially helpful to focus on **cholesterol** and **blood sugar**.\n"
            "- Consider asking your doctor about lifestyle changes and whether additional tests are needed."
        )
    elif cluster_id == 2:
        st.markdown(
            "- Your profile shows several moderate lab abnormalities, often seen in older adults.\n"
            "- Coordination of care across multiple risk factors (kidney, heart, blood counts, glucose) "
            "can be important.\n"
            "- Regular follow-up of labs and medication reviews may be helpful."
        )
    elif cluster_id == 3:
        st.markdown(
            "- Your profile is similar to a group with higher burden of chronic conditions in our dataset.\n"
            "- Close monitoring of **kidney function**, **cardiovascular risk**, **blood counts**, "
            "and **glucose** may be particularly important.\n"
            "- Discuss with your doctor how often labs should be repeated and whether specialist referrals "
            "are needed."
        )

    st.markdown("---")

    # 3.3 Lab-wise recommendations
    st.subheader("Lab-based suggestions and questions for your doctor")

    lab_summary_df = build_lab_summary(latest_features)

    if lab_summary_df.empty:
        st.write("No lab data available to generate lab-wise suggestions.")
    else:
        # Focus on non-normal labs
        problematic = lab_summary_df[
            lab_summary_df["raw_status"].isin(
                ["High", "Low", "Borderline high", "Borderline low"]
            )
        ].copy()

        if problematic.empty:
            st.write("All mapped labs are within typical ranges or close to them.")
        else:
            for _, row in problematic.iterrows():
                lab_name = row["Lab"]
                status = row["Status"]
                val_str = row["Your result"]
                key = row["lab_key"]
                info = LAB_INFO.get(key, {})
                st.markdown(f"#### {lab_name} – {val_str} ({status})")

                # Short generic explanation
                if key == "ldl_latest":
                    st.write(
                        "LDL cholesterol is often called 'bad cholesterol'. Higher values are associated "
                        "with increased cardiovascular risk."
                    )
                elif key == "hba1c_latest":
                    st.write(
                        "HbA1c reflects your average blood sugar over the last 2–3 months. Higher values "
                        "can indicate prediabetes or diabetes."
                    )
                elif key == "creatinine_latest" or key == "egfr_latest":
                    st.write(
                        "Creatinine and eGFR are used to assess kidney function. Abnormal values can suggest "
                        "impaired kidney function."
                    )
                elif key == "hemoglobin_latest":
                    st.write(
                        "Hemoglobin is a measure of red blood cells. Low values can indicate anemia."
                    )
                else:
                    st.write(
                        "This value is outside the typical range. Its significance depends on your overall "
                        "health and other tests."
                    )

                # Questions related to this lab
                st.write("**Questions you might ask your doctor:**")
                if key == "ldl_latest":
                    st.markdown(
                        "- What LDL cholesterol level should I aim for?\n"
                        "- Would lifestyle changes or medications help lower my cholesterol?"
                    )
                elif key == "hba1c_latest":
                    st.markdown(
                        "- Does this HbA1c level mean I have prediabetes or diabetes?\n"
                        "- How often should we repeat my HbA1c?"
                    )
                elif key in {"creatinine_latest", "egfr_latest"}:
                    st.markdown(
                        "- What do my kidney function tests mean for me right now?\n"
                        "- Do I need more frequent monitoring or further tests for kidney disease?"
                    )
                elif key == "hemoglobin_latest":
                    st.markdown(
                        "- Could this hemoglobin level indicate anemia?\n"
                        "- Do we need to look for possible causes like iron or vitamin deficiencies?"
                    )
                else:
                    st.markdown(
                        "- How concerned should we be about this lab value?\n"
                        "- Are any follow-up tests or lifestyle changes needed?"
                    )

                st.markdown("---")

    # 3.4 Educational / student view
    st.subheader("Educational view (for students / clinicians)")

    st.write(
        "From an educational perspective, this case illustrates how combinations of lab abnormalities "
        "can cluster into patterns that are associated with different chronic disease risks."
    )

    st.markdown(
        "- **Cluster context:** Patients in the same cluster share broadly similar patterns of age and lab values.\n"
        "- **Model behavior:** The risk scores are driven by features such as lipids, kidney function markers, "
        "blood counts, and glycemic markers.\n"
        "- **Real-world practice:** For actual patients, model outputs should always be integrated with "
        "detailed history, examination, and clinical guidelines."
    )

    st.caption(
        "This section is intended to help students think about how lab patterns, disease risk, and clustering "
        "can be combined into patient-friendly explanations."
    )
