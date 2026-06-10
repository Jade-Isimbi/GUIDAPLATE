"""
GuidaPlate — Streamlit MVP dashboard (proof-of-concept demo).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FOOD_DB_PATH = ROOT / "backend" / "data" / "food_database.csv"

REQUIRED_COLUMNS = [
    "english",
    "kinyarwanda",
    "category",
    "potassium_mg",
    "phosphorus_mg",
    "protein_g",
    "ckd_stage_safe",
    "notes",
]

STAGE_TO_NUM = {"G2": 2, "G3": 3, "G4": 4}

DISPLAY_COLUMNS = [
    "english",
    "kinyarwanda",
    "category",
    "potassium_mg",
    "phosphorus_mg",
    "protein_g",
    "ckd_stage_safe",
    "notes",
]


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        color: #1B4332;
        margin-bottom: 0.25rem;
    }
    .tagline {
        font-size: 1.15rem;
        color: #40916C;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1B4332;
        border-bottom: 2px solid #95D5B2;
        padding-bottom: 0.35rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #D8F3DC 0%, #B7E4C7 100%);
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        border: 1px solid #95D5B2;
        box-shadow: 0 2px 8px rgba(27, 67, 50, 0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1B4332;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #2D6A4F;
        margin-top: 0.25rem;
    }
    .step-card {
        background: #F8FFF9;
        border-left: 4px solid #40916C;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-radius: 0 8px 8px 0;
    }
    .risk-high {
        background: #FEE2E2;
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .risk-moderate {
        background: #FEF3C7;
        border: 2px solid #D97706;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .risk-low {
        background: #D1FAE5;
        border: 2px solid #059669;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .risk-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .rec-card {
        background: #FFFFFF;
        border: 1px solid #95D5B2;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(27, 67, 50, 0.06);
    }
    .rec-title {
        font-weight: 600;
        color: #1B4332;
        font-size: 1.05rem;
    }
    .rec-meta {
        color: #52796F;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    .warning-box {
        background: #FFF7ED;
        border: 1px solid #FDBA74;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #9A3412;
        margin: 1.5rem 0;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return list of missing required columns."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


@st.cache_data
def load_food_data() -> pd.DataFrame:
    """Load and validate the Rwanda food database CSV."""
    if not FOOD_DB_PATH.exists():
        raise FileNotFoundError(
            f"Food database not found at `{FOOD_DB_PATH}`. "
            "Place `food_database.csv` in `backend/data/`."
        )
    df = pd.read_csv(FOOD_DB_PATH)
    missing = validate_columns(df)
    if missing:
        raise ValueError(
            f"Food database is missing required columns: {', '.join(missing)}"
        )
    return df


def potassium_color(value: float) -> str:
    """Return hex color for potassium risk level."""
    if pd.isna(value):
        return "#FFFFFF"
    if value < 200:
        return "#D1FAE5"  # green
    if value <= 300:
        return "#FEF3C7"  # orange
    return "#FEE2E2"  # red


def apply_potassium_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply conditional background color to potassium_mg column."""

    def _style_cell(val):
        return f"background-color: {potassium_color(val)}"

    return df.style.map(_style_cell, subset=["potassium_mg"])


def parse_stage_safe(ckd_stage_safe: str) -> tuple[int, int] | None:
    """Parse ckd_stage_safe strings like '1-3' or '1-5'."""
    if pd.isna(ckd_stage_safe):
        return None
    text = str(ckd_stage_safe).strip()
    if "-" not in text:
        return None
    low, high = text.split("-", 1)
    try:
        return int(low), int(high)
    except ValueError:
        return None


def is_safe_for_stage(ckd_stage_safe: str, stage_num: int) -> bool:
    """True if food is safe for the given numeric CKD stage."""
    parsed = parse_stage_safe(ckd_stage_safe)
    if parsed is None:
        return False
    low, high = parsed
    return low <= stage_num <= high


def filter_by_ckd_safety(df: pd.DataFrame, safety_filter: str) -> pd.DataFrame:
    """Filter dataframe by CKD stage safety selectbox value."""
    if safety_filter == "All":
        return df
    stage_map = {
        "Safe for G2": 2,
        "Safe for G3": 3,
        "Safe for G4": 4,
    }
    stage_num = stage_map.get(safety_filter)
    if stage_num is None:
        return df
    mask = df["ckd_stage_safe"].apply(lambda x: is_safe_for_stage(x, stage_num))
    return df[mask]


def get_risk_thresholds() -> dict[str, dict[str, float]]:
    """Return KDOQI-aligned daily thresholds by CKD stage."""
    return {
        "G2": {
            "potassium": 3500,
            "phosphorus": 1000,
            "protein_per_kg": 0.8,
            "sodium": 2300,
        },
        "G3": {
            "potassium": 3000,
            "phosphorus": 800,
            "protein_per_kg": 0.6,
            "sodium": 2300,
        },
        "G4": {
            "potassium": 2500,
            "phosphorus": 700,
            "protein_per_kg": 0.55,
            "sodium": 2300,
        },
    }


def calculate_risk(
    stage: str,
    potassium: float,
    phosphorus: float,
    protein: float,
    body_weight: float,
    sodium: float,
) -> tuple[str, int, float, list[dict]]:
    """
    Compute dietary risk level and comparison rows.

    Returns: (risk_label, exceeded_count, protein_per_kg, comparison_rows)
    """
    thresholds = get_risk_thresholds()[stage]
    protein_per_kg = protein / body_weight if body_weight > 0 else 0.0

    checks = [
        ("Potassium", potassium, thresholds["potassium"], "mg/day"),
        ("Phosphorus", phosphorus, thresholds["phosphorus"], "mg/day"),
        ("Protein", protein_per_kg, thresholds["protein_per_kg"], "g/kg/day"),
        ("Sodium", sodium, thresholds["sodium"], "mg/day"),
    ]

    exceeded = 0
    rows = []
    for name, actual, limit, unit in checks:
        over = actual > limit
        if over:
            exceeded += 1
        if name == "Protein":
            actual_display = f"{actual:.2f} {unit}"
            limit_display = f"{limit:.2f} {unit}"
        else:
            actual_display = f"{actual:.0f} {unit}"
            limit_display = f"{limit:.0f} {unit}"
        rows.append(
            {
                "Nutrient": name,
                "Actual Intake": actual_display,
                "Safe Limit": limit_display,
                "Status": "Exceeds limit" if over else "Within limit",
            }
        )

    if exceeded >= 2:
        risk = "HIGH"
    elif exceeded == 1:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return risk, exceeded, protein_per_kg, rows


def get_food_recommendations(
    df: pd.DataFrame, stage: str, n: int = 3
) -> pd.DataFrame:
    """Return top n lowest-potassium foods safe for the selected stage."""
    stage_num = STAGE_TO_NUM[stage]
    safe = df[df["ckd_stage_safe"].apply(lambda x: is_safe_for_stage(x, stage_num))]
    return safe.nsmallest(n, "potassium_mg")


def plot_potassium_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of potassium by food, colored by risk level."""
    chart_df = df.sort_values("potassium_mg", ascending=False).copy()
    chart_df["color"] = chart_df["potassium_mg"].apply(potassium_color)

    fig = go.Figure(
        go.Bar(
            x=chart_df["english"],
            y=chart_df["potassium_mg"],
            marker_color=chart_df["color"],
            marker_line_color="#52796F",
            marker_line_width=0.5,
        )
    )
    fig.update_layout(
        title="Potassium Content by Food (mg)",
        xaxis_title="Food",
        yaxis_title="Potassium (mg)",
        xaxis_tickangle=-45,
        height=450,
        margin=dict(b=120),
        plot_bgcolor="#F8FFF9",
        paper_bgcolor="#FFFFFF",
    )
    fig.add_hline(
        y=300,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text="300 mg threshold",
    )
    return fig


def render_risk_box(risk: str) -> None:
    """Display styled risk result box."""
    messages = {
        "HIGH": (
            "risk-high",
            "🔴 HIGH RISK",
            "Your dietary intake today exceeds multiple safe limits for your CKD stage.",
        ),
        "MODERATE": (
            "risk-moderate",
            "🟡 MODERATE RISK",
            "One nutrient exceeds your safe limit.",
        ),
        "LOW": (
            "risk-low",
            "🟢 LOW RISK",
            "Your dietary intake is within safe limits for your CKD stage today.",
        ),
    }
    css_class, title, body = messages[risk]
    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="risk-title">{title}</div>'
        f"<p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def render_recommendation_card(row: pd.Series) -> None:
    """Render a single food recommendation card."""
    st.markdown(
        f'<div class="rec-card">'
        f'<div class="rec-title">{row["english"]} — {row["kinyarwanda"]}</div>'
        f'<div class="rec-meta">'
        f"Category: {row['category']} &nbsp;|&nbsp; "
        f"Potassium: {row['potassium_mg']:.0f} mg"
        f"</div>"
        f'<div class="rec-meta" style="margin-top:0.5rem;">{row["notes"]}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    st.markdown('<p class="main-header">GuidaPlate</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tagline">AI-Powered Dietary Guidance for CKD Patients in Rwanda</p>',
        unsafe_allow_html=True,
    )
    st.write(
        "GuidaPlate helps CKD patients understand food safety, nutrient risk, "
        "and personalized dietary recommendations grounded in clinical guidelines "
        "and Rwanda-specific food composition data."
    )

    c1, c2, c3 = st.columns(3)
    metrics = [
        ("50", "Rwandan Foods"),
        ("4", "CKD Stages"),
        ("3", "Risk Levels"),
    ]
    for col, (value, label) in zip([c1, c2, c3], metrics):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="section-header">How GuidaPlate Works</p>', unsafe_allow_html=True
    )
    steps = [
        "Explore commonly consumed Rwandan foods",
        "Enter daily nutrient intake",
        "Receive risk level and safer food suggestions",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(
            f'<div class="step-card"><strong>Step {i}.</strong> {step}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="warning-box">'
        "<strong>Disclaimer:</strong> This is a proof-of-concept system. "
        "Recommendations are based on KDOQI 2020 clinical guidelines and "
        "simplified rule-based logic. Always consult a healthcare provider."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("About this project"):
        st.markdown(
            """
**Data sources**
- NHANES 2017–2018, CDC
- Kenya Food Composition Tables 2018
- USDA FoodData Central
- Rwanda Food Balance Sheet

**Clinical guidelines**
- KDOQI 2020
- KDIGO 2024
            """
        )


def page_food_explorer(df: pd.DataFrame) -> None:
    st.title("Rwanda CKD Food Explorer")
    st.caption("Explore commonly consumed Rwandan foods and their CKD safety ratings")

    st.sidebar.markdown("### Filters")

    search = st.sidebar.text_input("Search by food name", placeholder="e.g. beans, ibijumba")
    categories = sorted(df["category"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
    safety_filter = st.sidebar.selectbox(
        "CKD stage safety",
        ["All", "Safe for G2", "Safe for G3", "Safe for G4"],
    )
    k_min, k_max = st.sidebar.slider(
        "Potassium range (mg)", min_value=0, max_value=1800, value=(0, 1800)
    )

    filtered = df.copy()
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    filtered = filter_by_ckd_safety(filtered, safety_filter)
    filtered = filtered[
        (filtered["potassium_mg"] >= k_min) & (filtered["potassium_mg"] <= k_max)
    ]
    if search.strip():
        q = search.strip().lower()
        filtered = filtered[
            filtered["english"].str.lower().str.contains(q, na=False)
            | filtered["kinyarwanda"].str.lower().str.contains(q, na=False)
        ]

    st.markdown(
        f"**Showing {len(filtered)} of {len(df)} foods**",
    )

    display_df = filtered[DISPLAY_COLUMNS].reset_index(drop=True)
    st.dataframe(apply_potassium_style(display_df), use_container_width=True, height=400)

    if not filtered.empty:
        st.plotly_chart(plot_potassium_chart(filtered), use_container_width=True)
    else:
        st.info("No foods match the current filters.")


def page_risk_assessment(df: pd.DataFrame) -> None:
    st.title("Dietary Risk Assessment")
    st.caption("Enter today's nutrient intake to estimate dietary risk")

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            stage = st.selectbox("CKD Stage", ["G2", "G3", "G4"])
            potassium = st.number_input(
                "Potassium intake today (mg)",
                min_value=0,
                max_value=6000,
                value=2000,
                step=100,
            )
            phosphorus = st.number_input(
                "Phosphorus intake today (mg)",
                min_value=0,
                max_value=2000,
                value=600,
                step=50,
            )
        with col2:
            protein = st.number_input(
                "Total protein (g)",
                min_value=0,
                max_value=200,
                value=50,
                step=5,
            )
            body_weight = st.number_input(
                "Body weight (kg)",
                min_value=30,
                max_value=150,
                value=65,
                step=1,
            )
            sodium = st.number_input(
                "Sodium intake today (mg)",
                min_value=0,
                max_value=5000,
                value=1500,
                step=100,
            )
        submitted = st.form_submit_button("CALCULATE RISK", use_container_width=True)

    if submitted:
        risk, exceeded, protein_per_kg, comparison = calculate_risk(
            stage, potassium, phosphorus, protein, body_weight, sodium
        )
        render_risk_box(risk)

        st.markdown(
            '<p class="section-header">Nutrient Comparison</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(comparison),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            '<p class="section-header">Top 3 Safer Food Recommendations</p>',
            unsafe_allow_html=True,
        )
        recs = get_food_recommendations(df, stage, n=3)
        if recs.empty:
            st.warning(f"No foods marked safe for {stage} in the database.")
        else:
            for _, row in recs.iterrows():
                render_recommendation_card(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="GuidaPlate",
        page_icon="🫘",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.sidebar.title("GuidaPlate")
    st.sidebar.caption("CKD Dietary Guidance — MVP Demo")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Food Explorer", "Risk Assessment"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "*Proof-of-concept demo for BSc capstone assessment.*"
    )

    try:
        food_df = load_food_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Food Explorer":
        page_food_explorer(food_df)
    else:
        page_risk_assessment(food_df)


if __name__ == "__main__":
    main()
