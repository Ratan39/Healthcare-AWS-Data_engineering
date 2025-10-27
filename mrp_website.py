# Streamlit Reports Block – Rule Index + Evidence Guidance (app_reports.py)
# -----------------------------------------------------------------------
# This prototype implements the requirements you listed:
# 1) Rule-Based Index (penalties 0–3) → Disease subscore (0–100) → Wellness
# 2) Evidence-mapped guidance by lab-pattern clusters (no patient-behavior data)
# 3) Reports block with chips (Normal/Watch/High), key signals, confidence, next steps
# 4) Shows two (or more) disease cards if multiple are flagged
# 5) Overall improvement line graph (monthly Wellness) with healthy band + milestones
#
# To run:
#   pip install streamlit pandas numpy plotly
#   streamlit run app_reports.py
# -----------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Patient Reports", page_icon="🧭", layout="wide")

# ------------------------------------------------------
# Reference ranges & penalty bins (transparent + editable)
# ------------------------------------------------------
REFS = {
    # Cardiovascular (mg/dL)
    "LDL": {"bins": [0, 100, 130, 160, np.inf], "penalties": [0, 1, 2, 3]},
    "HDL": {"bins": [0, 40, 60, np.inf], "penalties": [3, 0, 0]},  # HDL low = worse; protective >=60
    "Triglycerides": {"bins": [0, 150, 200, np.inf], "penalties": [0, 2, 3]},
    "TotalChol": {"bins": [0, 200, 240, np.inf], "penalties": [0, 2, 3]},

    # Prediabetes / Diabetes
    "A1c": {"bins": [0, 5.7, 6.5, np.inf], "penalties": [0, 2, 3]},  # 5.7–6.4 pre = 2, >=6.5 = 3
    "GlucoseBlood": {"bins": [0, 100, 126, np.inf], "penalties": [0, 2, 3]},

    # CKD
    "eGFR": {"bins": [0, 15, 45, 60, 90, np.inf], "penalties": [3, 2, 2, 1, 0]},
    "Creatinine": {"bins": [0, 0.6, 1.3, np.inf], "penalties": [1, 0, 2]},
    "BUN": {"bins": [0, 7, 20, np.inf], "penalties": [1, 0, 1]},

    # Anemia
    "Hemoglobin": {"bins": [0, 12.0, 16.5, np.inf], "penalties": [2, 0, 2]},
    "Hematocrit": {"bins": [0, 36.0, 49.0, np.inf], "penalties": [2, 0, 2]},

    # Liver
    "ALT": {"bins": [0, 40, np.inf], "penalties": [0, 2]},
    "AST": {"bins": [0, 40, np.inf], "penalties": [0, 2]},
    "Bilirubin": {"bins": [0, 1.2, np.inf], "penalties": [0, 2]},
    "Albumin": {"bins": [0, 3.5, 5.0, np.inf], "penalties": [2, 0, 1]},
}

# Disease → analyte weights (sum to 1.0 inside a disease)
WEIGHTS = {
    "Cardiovascular": {"LDL": 0.40, "HDL": 0.20, "Triglycerides": 0.20, "TotalChol": 0.20},
    "Prediabetes": {"A1c": 0.60, "GlucoseBlood": 0.40},
    "CKD": {"eGFR": 0.60, "Creatinine": 0.25, "BUN": 0.15},
    "Anemia": {"Hemoglobin": 0.6, "Hematocrit": 0.4},
    "Liver": {"ALT": 0.30, "AST": 0.30, "Bilirubin": 0.25, "Albumin": 0.15},
}

# Global disease weights for Wellness (sum to 1.0)
WELLNESS_WEIGHTS = {
    "Cardiovascular": 0.25,
    "Prediabetes": 0.25,
    "CKD": 0.20,
    "Liver": 0.15,
    "Anemia": 0.15,
}

# Evidence-linked strategies (cluster patterns → guidance)
EVIDENCE_GUIDANCE = {
    "high_ldl_low_hdl": {
        "pattern": "High LDL + Low HDL",
        "evidence": "Cardiology guidelines support reducing saturated fats and increasing physical activity to improve lipids.",
        "tips": [
            "Swap saturated fats for unsaturated (olive oil, nuts).",
            "Add brisk walking ~30 min/day to help raise HDL.",
            "Increase soluble fiber (oats, beans) to help lower LDL.",
        ],
    },
    "elevated_a1c_glucose": {
        "pattern": "Elevated A1c & Fasting Glucose",
        "evidence": "Diabetes prevention research supports reducing refined sugars and increasing dietary fiber.",
        "lead": "Based on your cluster (insulin resistance pattern), research shows that people with similar blood sugar and triglyceride levels benefit significantly from reducing refined carbohydrates and adding light post-meal walking. Even small consistent changes can prevent diabetes progression.",
        "tips": [
            "Cut sugary drinks; choose water or unsweetened options.",
            "Aim for fiber-rich meals (veggies, legumes, whole grains).",
            "Evening walks after meals can blunt glucose spikes.",
        ],
    },
    "low_egfr_high_creatinine": {
        "pattern": "Low eGFR & High Creatinine",
        "evidence": "CKD recommendations encourage hydration and reduced sodium intake.",
        "tips": [
            "Stay well hydrated unless your clinician advises otherwise.",
            "Reduce added salt; check labels for sodium.",
            "Discuss NSAID painkiller use with your clinician.",
        ],
    },
    "low_hemoglobin": {
        "pattern": "Low Hemoglobin",
        "evidence": "WHO guidance supports iron-rich foods for some anemia types.",
        "tips": [
            "Include iron-rich foods (beans, leafy greens, lean meats).",
            "Pair plant iron with vitamin C (e.g., beans + tomatoes).",
            "If vegetarian, consider B12 sources or discuss supplements.",
        ],
    },
    "liver_stress": {
        "pattern": "Liver enzyme elevation",
        "evidence": "Lifestyle guidance includes limiting alcohol and reviewing hepatotoxic medications.",
        "tips": [
            "Limit alcohol; some people benefit from a temporary break.",
            "Review medications/supplements with your clinician.",
            "Aim for balanced meals and regular movement.",
        ],
    },
}

# Card chip styles
CHIPS = {
    "Normal": {"emoji": "✅", "color": "#22AA6A"},
    "Watch": {"emoji": "⚠️", "color": "#F2A900"},
    "High": {"emoji": "❗", "color": "#D7263D"},
    "n/a": {"emoji": "▫️", "color": "#9AA0A6"},
}

# Mapping from SQL-style overall labels → chips
SQL_TO_CHIP = {
    "Cardiovascular": {
        "At risk": "High",
        "Likely normal": "Normal",
        "Insufficient data": "Watch",
    },
    "Prediabetes": {
        "Diabetes likely (lab criteria met)": "High",
        "Prediabetes / Elevated risk": "Watch",
        "Normal": "Normal",
        "Insufficient data": "Watch",
    },
    "CKD": {
        "High CKD risk (eGFR < 30)": "High",
        "At risk (kidney impairment likely)": "Watch",
        "Likely normal": "Normal",
        "Insufficient data": "Watch",
    },
    "Anemia": {
        "Anemia likely": "High",
        "No anemia signal": "Normal",
        "Insufficient data": "Watch",
    },
    "Liver": {
        "Liver dysfunction likely (multiple abnormalities)": "High",
        "Possible liver dysfunction": "Watch",
        "No liver dysfunction signal": "Normal",
        "Insufficient data": "Watch",
    },
}

# -------------------------------------
# Helper functions: penalties and scores
# -------------------------------------

def penalty_for_value(analyte: str, value: float) -> int:
    """Return penalty 0–3 for a numeric lab value based on REFS bins.
    Bins are ordered; the index maps to penalty.
    """
    if analyte not in REFS or value is None or np.isnan(value):
        return 0
    bins = REFS[analyte]["bins"]
    penalties = REFS[analyte]["penalties"]
    # find the bin index where value < next edge
    for i in range(len(bins) - 1):
        if value < bins[i + 1]:
            return int(penalties[i])
    return int(penalties[-1])


def disease_subscore(disease: str, row: pd.Series) -> Tuple[float, Dict[str, int]]:
    """Compute subscore (0–100) for a disease from a row of analytes.
    Returns (subscore, per_analyte_penalties)
    """
    weights = WEIGHTS[disease]
    penalties: Dict[str, int] = {}
    weighted = 0.0
    total_weight = 0.0
    for analyte, w in weights.items():
        val = row.get(analyte, np.nan)
        if pd.notnull(val):
            p = penalty_for_value(analyte, float(val))
            penalties[analyte] = p
            weighted += p * w
            total_weight += w
    if total_weight == 0:
        return (np.nan, penalties)
    # Normalize to max penalty 3 (worst) per analyte, so weighted max is <=3
    max_penalty = 3.0
    sub = 100.0 - (weighted * 100.0 / max_penalty)
    return (max(0.0, min(100.0, sub)), penalties)


def wellness_score(subscores: Dict[str, float]) -> Tuple[float, float, List[str]]:
    """Combine disease subscores to Wellness 0–100 using WELLNESS_WEIGHTS.
    Returns (score, confidence (0-1), contributing diseases list)
    Confidence is the sum of weights present.
    """
    score_sum = 0.0
    weight_sum = 0.0
    used = []
    for disease, w in WELLNESS_WEIGHTS.items():
        s = subscores.get(disease, np.nan)
        if pd.notnull(s):
            score_sum += s * w
            weight_sum += w
            used.append(disease)
    if weight_sum == 0:
        return (np.nan, 0.0, [])
    # Re-normalize by the used weight mass
    wellness = score_sum / weight_sum
    return (wellness, weight_sum, used)

# -------------------------------------
# Evidence-mapped cluster pattern logic
# -------------------------------------

def detect_patterns(row: pd.Series) -> List[str]:
    patterns = []
    # High LDL + Low HDL
    if pd.notnull(row.get("LDL")) and pd.notnull(row.get("HDL")):
        if row["LDL"] >= 160 or (row["LDL"] >= 130 and row["HDL"] < 40):
            patterns.append("high_ldl_low_hdl")
    # Elevated A1c & Fasting Glucose
    if pd.notnull(row.get("A1c")) and pd.notnull(row.get("GlucoseBlood")):
        if row["A1c"] >= 5.7 and row["GlucoseBlood"] >= 100:
            patterns.append("elevated_a1c_glucose")
    # Low eGFR & High Creatinine
    if pd.notnull(row.get("eGFR")) and pd.notnull(row.get("Creatinine")):
        if row["eGFR"] < 60 or row["Creatinine"] >= 1.3:
            patterns.append("low_egfr_high_creatinine")
    # Low Hemoglobin
    if pd.notnull(row.get("Hemoglobin")) and row["Hemoglobin"] < 12.0:
        patterns.append("low_hemoglobin")
    return patterns

# -------------------------------------
# Demo / Upload
# -------------------------------------
with st.sidebar:
    st.markdown("## 📄 Upload labs (CSV)")
    st.caption("Columns can include: date, patient_id, and any of LDL, HDL, Triglycerides, TotalChol, A1c, GlucoseBlood, eGFR, Creatinine, BUN, Hemoglobin, Hematocrit, ALT, AST, Bilirubin, Albumin")
    uploaded = st.file_uploader("Choose CSV", type=["csv"]) 
    demo = st.toggle("Use demo data", value=not uploaded)

if demo:
    dates = pd.date_range(end=datetime.today(), periods=7, freq="30D")
    df = pd.DataFrame({
        "date": dates,
        "patient_id": ["P001"]*len(dates),
        "LDL": [168, 160, 155, 150, 145, 140, 135],
        "HDL": [38, 39, 40, 42, 44, 46, 48],
        "Triglycerides": [210, 205, 195, 185, 175, 165, 150],
        "TotalChol": [245, 240, 232, 225, 215, 205, 198],
        "A1c": [6.1, 6.0, 5.9, 5.8, 5.8, 5.7, 5.7],
        "GlucoseBlood": [118, 114, 110, 108, 104, 100, 98],
        "eGFR": [72, 74, 76, 78, 82, 86, 90],
        "Creatinine": [1.2, 1.18, 1.15, 1.1, 1.05, 1.02, 1.0],
        "BUN": [22, 21, 20, 19, 18, 17, 16],
        "Hemoglobin": [11.6, 11.8, 12.0, 12.3, 12.4, 12.5, 12.7],
        "Hematocrit": [35.5, 36.0, 36.5, 37.0, 38.0, 39.0, 40.0],
        "ALT": [52, 48, 44, 40, 38, 36, 34],
        "AST": [41, 40, 39, 37, 36, 35, 34],
        "Bilirubin": [1.3, 1.2, 1.1, 1.1, 1.0, 0.9, 0.9],
        "Albumin": [3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
    })
else:
    if not uploaded:
        st.stop()
    df = pd.read_csv(uploaded)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.date_range(end=datetime.today(), periods=len(df))

patients = sorted(df.get("patient_id", pd.Series(["Patient"])) .unique()) if "patient_id" in df.columns else ["Patient"]
patient = st.sidebar.selectbox("Patient", patients)

# Global toggle for details
deep_insights = st.sidebar.toggle("Deeper insights", value=False)
# Layout choice: new Rings vs classic Cards
layout_choice = st.sidebar.radio("View", ["Rings (new)", "Cards (classic)"], index=0)

pdf = df[df["patient_id"] == patient] if "patient_id" in df.columns else df.copy()
pdf = pdf.sort_values("date")
latest = pdf.iloc[-1]

st.title("🧭 Patient Reports")
st.caption("Educational insights, not a diagnosis. Talk with your clinician about your results.")

# -------------------------------------
# Subscores per disease (rule index)
# -------------------------------------
SUBSCORES: Dict[str, float] = {}
PENALTY_DETAILS: Dict[str, Dict[str, int]] = {}

for disease in WEIGHTS.keys():
    s, pens = disease_subscore(disease, latest)
    SUBSCORES[disease] = s
    PENALTY_DETAILS[disease] = pens

wellness, conf, used = wellness_score(SUBSCORES)

# Header: wellness + date + based-on count
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown("### Wellness gauge (rule-based)")
    if np.isnan(wellness):
        st.info("Not enough data to compute Wellness.")
    else:
        bg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(wellness),
            title={"text": f"Based on {len(used)}/5 areas"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#2E86AB"},
                   "steps": [
                       {"range": [0, 40], "color": "#F8D7DA"},
                       {"range": [40, 60], "color": "#FFF3CD"},
                       {"range": [60, 80], "color": "#E2F0D9"},
                       {"range": [80, 100], "color": "#D4EDDA"},
                   ]}
        ))
        bg.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(bg, use_container_width=True)
with c2:
    st.markdown("### Evidence-based tips today")
    pats = detect_patterns(latest)
    shown = 0
    if pats:
        for p in pats:
            meta = EVIDENCE_GUIDANCE[p]
            st.write(f"**{meta['pattern']}** – {meta['evidence']}")
            for t in meta["tips"]:
                st.write(f"• {t}")
                shown += 1
                if shown >= 6:
                    break
            if shown >= 6:
                break
    else:
        st.write("**General wellness** – Small changes add up.")
        for tip in [
            "Aim for 150 minutes/week of moderate activity (walks count).",
            "Prioritize whole foods: vegetables, fruits, legumes, whole grains.",
            "Reduce added sugar and ultra-processed snacks.",
        ]:
            st.write(f"• {tip}")
with c3:
    st.empty()

st.divider()

# -------------------------------------
# Reports block (cards)
# -------------------------------------

# Helper: chip renderer

def chip(label: str) -> str:
    style = CHIPS.get(label, CHIPS["n/a"])
    return f"<span style='background:{style['color']};color:white;padding:4px 10px;border-radius:16px;font-weight:600;'>{style['emoji']} {label}</span>"

# Determine SQL-like overall labels from penalties (mirror your mapping)

def overall_label_for_disease(disease: str, pens: Dict[str, int]) -> str:
    # Heuristic mirroring your SQL outcomes (simplified for the prototype)
    if not pens:
        return "Insufficient data"
    max_p = max(pens.values())
    if disease == "Cardiovascular":
        if max_p >= 3 or (pens.get("LDL",0) >= 2 and pens.get("HDL",0) >= 1):
            return "At risk"
        return "Likely normal"
    if disease == "Prediabetes":
        if pens.get("A1c",0) >= 3 or pens.get("GlucoseBlood",0) >= 3:
            return "Diabetes likely (lab criteria met)"
        if pens.get("A1c",0) >= 2 or pens.get("GlucoseBlood",0) >= 2:
            return "Prediabetes / Elevated risk"
        return "Normal"
    if disease == "CKD":
        if pens.get("eGFR",0) >= 3:
            return "High CKD risk (eGFR < 30)"
        if pens.get("eGFR",0) >= 2 or pens.get("Creatinine",0) >= 2:
            return "At risk (kidney impairment likely)"
        return "Likely normal"
    if disease == "Anemia":
        if pens.get("Hemoglobin",0) >= 2 and pens.get("Hematocrit",0) >= 2:
            return "Anemia likely"
        return "No anemia signal"
    if disease == "Liver":
        multi = sum(1 for a in ("ALT","AST","Bilirubin","Albumin") if pens.get(a,0) >= 2)
        if multi >= 2:
            return "Liver dysfunction likely (multiple abnormalities)"
        if multi == 1:
            return "Possible liver dysfunction"
        return "No liver dysfunction signal"
    return "Insufficient data"

# Confidence by recency/coverage

def confidence_for_disease(disease: str, pdf: pd.DataFrame) -> str:
    cols = list(WEIGHTS[disease].keys())
    recent = pdf.tail(4)  # last 4 records ~ last few months
    count_present = sum(any(pd.notnull(recent[c])) for c in cols if c in pdf.columns)
    if count_present >= 2:
        return "High"
    if count_present == 1:
        return "Medium"
    return "Low"

# Build cards data
cards = []
for disease in WEIGHTS.keys():
    subs = SUBSCORES.get(disease)
    pens = PENALTY_DETAILS.get(disease, {})
    overall = overall_label_for_disease(disease, pens)
    chip_label = SQL_TO_CHIP[disease].get(overall, "Watch")
    conf = confidence_for_disease(disease, pdf)
    # key signals (tiny badges): list analytes with non-zero penalties
    signals = [f"{a}: {['Normal','Borderline','High','Very high'][p]}" for a, p in pens.items() if p > 0]
    last_date = latest["date"].date() if isinstance(latest.get("date"), pd.Timestamp) else latest.get("date")
    cards.append({
        "disease": disease,
        "subscore": subs,
        "overall": overall,
        "chip": chip_label,
        "confidence": conf,
        "signals": signals[:4],
        "last": last_date,
    })

# Only show Watch/High by default
show_all = st.checkbox("Show all 5 reports", value=False)
filtered = [c for c in cards if c["chip"] in ("Watch","High")] if not show_all else cards

# ---- Overview (Rings vs Cards) ----
if layout_choice == "Rings (new)":
    st.subheader("Your Health Rings")
    import plotly.graph_objects as go

    def ring_chart(score: float, label: str, chip_label: str):
        score = max(0, min(100, float(score if pd.notnull(score) else 0)))
        remain = 100 - score
        color = CHIPS.get(chip_label, CHIPS["n/a"])['color']
        fig = go.Figure(data=[
            go.Pie(values=[score, remain], hole=0.72, sort=False, direction='clockwise',
                   marker=dict(colors=[color, '#EAEAEA']), textinfo='none', hoverinfo='skip')
        ])
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0), height=180,
                          annotations=[dict(text=f"{int(round(score))}", x=0.5, y=0.52, showarrow=False,
                                            font=dict(size=28, color='#222')),
                                       dict(text=label, x=0.5, y=0.14, showarrow=False,
                                            font=dict(size=13, color='#555'))])
        st.plotly_chart(fig, use_container_width=True)

    # Build ring items (always show all 5)
    ring_items = []
    for disease in WEIGHTS.keys():
        subs, _ = disease_subscore(disease, latest)
        # Find chip for color context
        card = next((c for c in cards if c['disease'] == disease), None)
        chip_label = card['chip'] if card else 'n/a'
        score = subs if pd.notnull(subs) else (100 if chip_label == 'Normal' else 65 if chip_label == 'Watch' else 35)
        ring_items.append({"disease": disease, "score": score, "chip": chip_label})

    cols5 = st.columns(5)
    for i, item in enumerate(ring_items):
        with cols5[i % 5]:
            ring_chart(item["score"], item["disease"], item["chip"])
    st.caption("Rings show your current wellness for each area (0–100). Colors reflect risk level.")

    # --- Key markers (Option 4 element): top drivers per Watch/High disease ---
    if filtered:
        st.markdown("### Key markers driving your risk")
        km_cols = st.columns(2)
        idx = 0
        for c in filtered:
            pens = PENALTY_DETAILS.get(c["disease"], {})
            if not pens:
                continue
            top = sorted(pens.items(), key=lambda x: x[1], reverse=True)[:3]
            rows = []
            for a, p in top:
                val = latest.get(a)
                hr = {
                    "LDL": "< 100 mg/dL", "HDL": "≥ 40 mg/dL", "Triglycerides": "< 150 mg/dL", "TotalChol": "< 200 mg/dL",
                    "A1c": "< 5.7%", "GlucoseBlood": "70–99 mg/dL",
                    "eGFR": "≥ 60", "Creatinine": "0.6–1.3 mg/dL", "BUN": "7–20 mg/dL",
                    "ALT": "< 40 U/L", "AST": "< 40 U/L", "Bilirubin": "0.3–1.2 mg/dL", "Albumin": "3.5–5.0 g/dL",
                    "Hemoglobin": "12.0–16.5 g/dL", "Hematocrit": "36–49%",
                }.get(a, "—")
                status = ["Normal ✅", "Borderline ⚠️", "High ❗", "Very high ❗"][p] if p is not None else "n/a"
                if isinstance(val, (int, float)) and not pd.isna(val):
                    vtxt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                else:
                    vtxt = str(val) if val is not None else "—"
                rows.append([a, vtxt, hr, status])
            if rows:
                df_km = pd.DataFrame(rows, columns=["Marker", "Your Value", "Healthy Range", "Status"])
                with km_cols[idx % 2]:
                    st.markdown(f"**{c['disease']}**  " + chip(c["chip"]), unsafe_allow_html=True)
                    st.table(df_km)
                idx += 1
    else:
        st.success("All looks normal today based on available results.")

else:
    # Classic cards view
    if not filtered:
        st.success("All looks normal today based on available results.")

    cols = st.columns(2)
    for i, c in enumerate(filtered):
        with cols[i % 2]:
            st.markdown(f"### {c['disease']}  " + chip(c["chip"]), unsafe_allow_html=True)
            st.caption(f"Last tested: **{c['last']}** · Confidence: **{c['confidence']}**")
            if c["signals"]:
                st.write("**Insights:** ", ", ".join(c["signals"]))
            # Trend arrow
            primary = list(WEIGHTS[c["disease"]].keys())[0]
            if primary in pdf.columns and pdf[primary].notna().sum() >= 2:
                v = pdf[primary].dropna()
                slope = (v.iloc[-1] - v.iloc[0]) / max(1, len(v)-1)
                arrow = "→ stable"
                if slope < 0: arrow = "▲ improving" if c["disease"] in ("Cardiovascular","Prediabetes","Liver","CKD") else "▼ worsening"
                if slope > 0: arrow = "▼ worsening" if c["disease"] in ("Cardiovascular","Prediabetes","Liver","CKD") else "▲ improving"
                st.caption(f"Trend: {arrow}")

            st.markdown("**Next steps**")
            steps = ["Recheck in ~3 months"]
            if c["disease"] == "Cardiovascular":
                steps.append("Discuss statin eligibility")
                steps.append("Quick nutrition tips: Mediterranean-style swaps")
            elif c["disease"] == "Prediabetes":
                steps.append("Quick nutrition tips: reduce refined sugar, add fiber")
            elif c["disease"] == "CKD":
                steps.append("Quick tips: hydration and reduce salt (unless advised otherwise)")
            elif c["disease"] == "Anemia":
                steps.append("Quick tips: iron-rich foods; pair with vitamin C")
            elif c["disease"] == "Liver":
                steps.append("Quick tips: limit alcohol; review meds with clinician")
            for s in steps:
                st.write("- ", s)
            st.markdown("---")

# -------------------------------------
# Details sections: Factors Table + Cluster-Based Wellness Guidance
# (Shown only when the disease is Watch/High)
# -------------------------------------
# Note: Added Prediabetes Forecast Line (6–12 month outlook) below factors table.

DISEASE_GUIDANCE = {
    "Cardiovascular": {
        "title": "Cholesterol Pattern Cluster",
        "lead": (
            "Based on your cholesterol pattern, research shows that people with similar lab results can often "
            "improve heart health through small, consistent lifestyle changes—especially those that support healthy "
            "cholesterol balance and reduce inflammation."
        ),
        "tips": [
            "Choose healthy fats like olive oil, nuts, and seeds instead of fried or processed foods.",
            "Try adding 20–30 minutes of walking most days to help raise HDL and lower LDL.",
            "Increase fiber intake (beans, oats, vegetables) to naturally help remove LDL.",
            "If you smoke, consider support to quit—stopping can increase HDL within weeks.",
        ],
    },
    "Prediabetes": {
        "title": "Blood Sugar Pattern Cluster",
        "lead": (
            "Based on your blood sugar pattern, research shows that people with similar lab values often see meaningful "
            "improvements by reducing refined carbohydrates and staying lightly active after meals. Even small "
            "consistent steps can reverse rising blood sugar over time."
        ),
        "tips": [
            "Replace sugary drinks with water or unsweetened beverages.",
            "Fill half your plate with vegetables and add protein to stabilize sugar levels.",
            "Take a 10–15 minute walk after meals to help your body use sugar for energy.",
            "Eat at regular times and avoid late-night snacking to support stable blood sugar.",
        ],
    },
    "CKD": {
        "title": "Kidney Function Pattern",
        "lead": (
            "Your lab pattern suggests your kidneys may be under extra stress. The good news is that many people with "
            "similar results have improved kidney function through hydration, gentle nutrition changes, and managing "
            "blood pressure."
        ),
        "tips": [
            "Stay well-hydrated unless your doctor has advised otherwise.",
            "Limit high-salt foods (packaged snacks, instant noodles, processed meats).",
            "If you use ibuprofen or similar painkillers often, discuss alternatives with your clinician.",
            "Include kidney-friendly foods like fruits, vegetables, and whole grains.",
        ],
    },
    "Liver": {
        "title": "Liver Enzyme Pattern",
        "lead": (
            "Your liver markers suggest the liver may be working harder than usual. People with this pattern often see "
            "improvements through gentle lifestyle adjustments that reduce liver load and support natural functions."
        ),
        "tips": [
            "Limit alcohol to give your liver a chance to heal and restore balance.",
            "Avoid unnecessary over-the-counter meds/supplements unless prescribed.",
            "Choose liver-friendly foods like leafy greens, berries, and lean proteins.",
            "Aim for regular movement — even light activity helps improve enzyme levels over time.",
        ],
    },
    "Anemia": {
        "title": "Blood Oxygen Pattern",
        "lead": (
            "This pattern suggests your red blood cell levels may be slightly low. Many people with similar results "
            "improve their energy and blood oxygen levels through nutrition that supports red blood cell production."
        ),
        "tips": [
            "Add iron-rich foods such as beans, spinach, and lean meats.",
            "Pair plant-based iron with vitamin C (like lentils with tomatoes or citrus).",
            "Ensure adequate vitamin B12 (through foods or as recommended).",
            "Prioritize regular sleep and moderate activity to help your body use oxygen effectively.",
        ],
    },
}

for c in filtered:
    if not deep_insights:
        continue
    if c["chip"] not in ("Watch", "High"):
        continue

    st.markdown(f"## {c['disease']} — Details")

    # Separate factors table per disease
    if c["disease"] == "Prediabetes":
        st.markdown("**What’s affecting your blood sugar**")
        a1c = latest.get("A1c"); glu = latest.get("GlucoseBlood")
        table_data = []
        if pd.notnull(a1c):
            status = "Diabetes ❗" if a1c >= 6.5 else ("Prediabetes ⚠️" if a1c >= 5.7 else "Normal ✅")
            table_data.append(["A1c", f"{a1c:.2f}%", "< 5.7%", status])
        if pd.notnull(glu):
            status = "Diabetes ❗" if glu >= 126 else ("Prediabetes ⚠️" if glu >= 100 else "Normal ✅")
            table_data.append(["Fasting Glucose", f"{glu:.0f} mg/dL", "70–99 mg/dL", status])
        if table_data:
            df_table = pd.DataFrame(table_data, columns=["Factor", "Your Value", "Healthy Range", "Status"])
            st.table(df_table)

        # ---- Prediabetes Forecast Line (6–12 month outlook) ----
        # Use A1c if available; fallback to fasting glucose.
        import plotly.graph_objects as go
        series_label, y_hist = None, None
        if "A1c" in pdf.columns and pdf["A1c"].notna().sum() >= 3:
            series_label = "A1c (%)"; y_hist = pdf[["date","A1c"]].dropna()
        elif "GlucoseBlood" in pdf.columns and pdf["GlucoseBlood"].notna().sum() >= 3:
            series_label = "Fasting Glucose (mg/dL)"; y_hist = pdf[["date","GlucoseBlood"]].dropna()

        if y_hist is not None and len(y_hist) >= 3:
            # Convert dates to ordinal for simple linear regression
            x = y_hist["date"].map(pd.Timestamp.toordinal).values
            y = y_hist.iloc[:,1].values
            # Fit simple linear model
            coeffs = np.polyfit(x, y, 1)
            m, b = coeffs[0], coeffs[1]
            # Build future timeline: next 12 months at monthly cadence
            last_date = pd.to_datetime(y_hist["date"].max())
            future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=12, freq="MS")
            x_future = future_dates.map(pd.Timestamp.toordinal).values
            y_future = m * x_future + b
            # Simple widening CI (heuristic for UI)
            sigma = max(1e-6, np.std(y - (m*x + b))) if len(y) > 2 else 0.0
            baseline = np.maximum(1.0, np.nanmean(np.abs(y)))
            ci = sigma + (np.arange(1, len(future_dates)+1) / len(future_dates)) * (0.15 * baseline)
            upper = y_future + ci
            lower = y_future - ci

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_hist["date"], y=y_hist.iloc[:,1], mode="lines+markers", name="Historical"))
            fig.add_trace(go.Scatter(x=future_dates, y=y_future, mode="lines", name="Forecast", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=np.concatenate([future_dates, future_dates[::-1]]),
                                     y=np.concatenate([upper, lower[::-1]]),
                                     fill="toself", mode="lines", line=dict(width=0),
                                     name="Confidence range", opacity=0.2))
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=300,
                              title="6–12 Month Outlook (Prediabetes)",
                              xaxis_title="Date", yaxis_title=series_label)
            st.plotly_chart(fig, use_container_width=True)

    elif c["disease"] == "Cardiovascular":
        st.markdown("**What’s affecting your heart risk**")
        ldl = latest.get("LDL"); hdl = latest.get("HDL"); tg = latest.get("Triglycerides"); tc = latest.get("TotalChol")
        table_data = []
        if pd.notnull(ldl):
            status = "High ❗" if ldl >= 160 else ("Borderline ⚠️" if ldl >= 130 else ("Near optimal ✅" if ldl >= 100 else "Optimal ✅"))
            table_data.append(["LDL", f"{ldl:.0f} mg/dL", "< 100 mg/dL", status])
        if pd.notnull(hdl):
            status = "Low ⚠️" if hdl < 40 else ("Protective ✅" if hdl >= 60 else "Normal ✅")
            table_data.append(["HDL", f"{hdl:.0f} mg/dL", "≥ 40 mg/dL (≥60 protective)", status])
        if pd.notnull(tg):
            status = "High ❗" if tg >= 200 else ("Borderline ⚠️" if tg >= 150 else "Normal ✅")
            table_data.append(["Triglycerides", f"{tg:.0f} mg/dL", "< 150 mg/dL", status])
        if pd.notnull(tc):
            status = "High ❗" if tc >= 240 else ("Borderline ⚠️" if tc >= 200 else "Desirable ✅")
            table_data.append(["Total Cholesterol", f"{tc:.0f} mg/dL", "< 200 mg/dL", status])
        if table_data:
            df_table = pd.DataFrame(table_data, columns=["Factor", "Your Value", "Healthy Range", "Status"])
            st.table(df_table)

    elif c["disease"] == "CKD":
        st.markdown("**What’s affecting your kidney health**")
        egfr = latest.get("eGFR"); cr = latest.get("Creatinine"); bun = latest.get("BUN")
        table_data = []
        if pd.notnull(egfr):
            status = "High risk ❗" if egfr < 30 else ("Watch ⚠️" if egfr < 60 else "Normal ✅")
            table_data.append(["eGFR", f"{egfr:.0f}", "≥ 60", status])
        if pd.notnull(cr):
            status = "High ⚠️" if cr > 1.3 else ("Low ⚠️" if cr < 0.6 else "Normal ✅")
            table_data.append(["Creatinine", f"{cr:.2f} mg/dL", "0.6–1.3 mg/dL", status])
        if pd.notnull(bun):
            status = "High ⚠️" if bun > 20 else ("Low ⚠️" if bun < 7 else "Normal ✅")
            table_data.append(["BUN", f"{bun:.0f} mg/dL", "7–20 mg/dL", status])
        if table_data:
            df_table = pd.DataFrame(table_data, columns=["Factor", "Your Value", "Healthy Range", "Status"])
            st.table(df_table)

    elif c["disease"] == "Liver":
        st.markdown("**What’s affecting your liver health**")
        alt = latest.get("ALT"); ast = latest.get("AST"); bili = latest.get("Bilirubin"); alb = latest.get("Albumin")
        table_data = []
        if pd.notnull(alt):
            status = "High ❗" if alt >= 40 else "Normal ✅"
            table_data.append(["ALT", f"{alt:.0f} U/L", "< 40 U/L", status])
        if pd.notnull(ast):
            status = "High ❗" if ast >= 40 else "Normal ✅"
            table_data.append(["AST", f"{ast:.0f} U/L", "< 40 U/L", status])
        if pd.notnull(bili):
            status = "High ⚠️" if bili > 1.2 else "Normal ✅"
            table_data.append(["Total Bilirubin", f"{bili:.1f} mg/dL", "0.3–1.2 mg/dL", status])
        if pd.notnull(alb):
            status = "Low ⚠️" if alb < 3.5 else ("High ⚠️" if alb > 5.0 else "Normal ✅")
            table_data.append(["Albumin", f"{alb:.1f} g/dL", "3.5–5.0 g/dL", status])
        if table_data:
            df_table = pd.DataFrame(table_data, columns=["Factor", "Your Value", "Healthy Range", "Status"])
            st.table(df_table)

    elif c["disease"] == "Anemia":
        st.markdown("**What’s affecting your blood oxygen**")
        hb = latest.get("Hemoglobin"); hct = latest.get("Hematocrit")
        table_data = []
        if pd.notnull(hb):
            status = "Low ⚠️" if hb < 12.0 else ("High ⚠️" if hb > 16.5 else "Normal ✅")
            table_data.append(["Hemoglobin", f"{hb:.1f} g/dL", "12.0–16.5 g/dL", status])
        if pd.notnull(hct):
            status = "Low ⚠️" if hct < 36.0 else ("High ⚠️" if hct > 49.0 else "Normal ✅")
            table_data.append(["Hematocrit", f"{hct:.0f}%", "36–49%", status])
        if table_data:
            df_table = pd.DataFrame(table_data, columns=["Factor", "Your Value", "Healthy Range", "Status"])
            st.table(df_table)

    # Cluster-Based Wellness Guidance (separate block)
    guide = DISEASE_GUIDANCE.get(c["disease"])
    if guide:
        st.markdown("### Cluster-Based Wellness Guidance")
        st.info(guide["lead"])  # friendly evidence-based insight
        for tip in guide["tips"]:
            st.write("- ", tip)

    st.markdown("---")

# -------------------------------------
# Overall improvement line graph (monthly Wellness)
# -------------------------------------
st.subheader("Your progress over time")

# Compute monthly wellness from historical rows
monthly = pdf.copy()
monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()

rows = []
for m, grp in monthly.groupby("month"):
    last = grp.iloc[-1]
    subscores_m = {}
    for disease in WEIGHTS.keys():
        s, _ = disease_subscore(disease, last)
        subscores_m[disease] = s
    w, conf_m, used_m = wellness_score(subscores_m)
    rows.append({"month": m, "wellness": w, "used": len(used_m)})

hist = pd.DataFrame(rows).dropna()

if hist.empty:
    st.caption("Not enough history to plot.")
else:
    fig = go.Figure()
    # Plain line only (no background band)
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["wellness"], mode="lines+markers", name="Wellness"))
    # Trend annotation
    if len(hist) >= 2:
        delta = hist["wellness"].iloc[-1] - hist["wellness"].iloc[0]
        fig.add_annotation(x=hist["month"].iloc[-1], y=hist["wellness"].iloc[-1],
                           text=f"Δ {delta:+.0f} in {len(hist)} months", showarrow=True)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.caption("Strategies shown are general wellness guidance mapped from lab patterns and medical literature—not treatment.")
