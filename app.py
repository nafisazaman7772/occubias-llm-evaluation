import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# App Config
# ------------------------------------------------------------
st.set_page_config(page_title="OccuBias: Occupational LLM Evaluation", layout="wide")

st.title("OccuBias: Occupational LLM Evaluation")
st.caption("Interactive exploration of occupational gender bias in Llama 3- Instruct & Gemma 7B- it  outputs .")

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
LLAMA_PATH = "data/llama_results.csv"
GEMMA_PATH = "data/gemma_results.csv"

MODEL_LLAMA = "Llama-3-8B-Instruct"
MODEL_GEMMA = "Gemma-7B-IT"

MALE = {"he", "him", "his"}
FEMALE = {"she", "her", "hers"}
NEUTRAL = {"they", "them", "their"}

TEMPLATE_CANON = {
    "neutral": "neutral",
    "ambig": "ambiguous",
    "ambiguous": "ambiguous",
    "stereo": "stereotype",
    "stereotype": "stereotype",
}

EXPECTED_OUTPUT_COLS = ["Model", "Profession", "Template", "Gender"]


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_template(x: str) -> str:
    s = _norm_str(x).lower()
    for key, val in TEMPLATE_CANON.items():
        if key in s:
            return val
    # if already clean like "neutral"
    if s in {"neutral", "ambiguous", "stereotype"}:
        return s
    return s


def pronoun_to_gender(p: str) -> str:
    s = _norm_str(p).lower()
    if s in MALE:
        return "Male"
    if s in FEMALE:
        return "Female"
    if s in NEUTRAL:
        return "Neutral"
    # If your CSV already has Male/Female/Neutral in a column, handle that:
    if s in {"male", "female", "neutral"}:
        return s.capitalize()
    return "Neutral"


def pct_from_gender(series: pd.Series) -> dict:
    total = len(series)
    if total == 0:
        return {"Male %": 0.0, "Female %": 0.0, "Neutral %": 0.0}
    counts = series.value_counts().to_dict()
    return {
        "Male %": round(counts.get("Male", 0) * 100 / total, 2),
        "Female %": round(counts.get("Female", 0) * 100 / total, 2),
        "Neutral %": round(counts.get("Neutral", 0) * 100 / total, 2),
    }


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def load_and_normalise(path: str, model_name: str) -> pd.DataFrame:
    """
    Loads a CSV and normalises it into a canonical schema:
    Model, Profession, Template, Gender (+ optional extra columns retained)
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Try to find profession column
    prof_col = pick_first_existing(df, ["Profession", "profession", "job", "occupation"])
    tmpl_col = pick_first_existing(df, ["Template", "template", "template_type", "prompt_type"])
    gender_col = pick_first_existing(df, ["Gender", "gender"])
    pron_col = pick_first_existing(df, ["Selected Pronoun", "selected pronoun", "pronoun", "Pronoun", "selected_pronoun"])

    # Profession is required
    if prof_col is None:
        raise ValueError(f"[{model_name}] Missing Profession column. Found columns: {list(df.columns)}")

    # Template is optional but recommended
    if tmpl_col is None:
        # create a fallback column if not present
        df["_TemplateTmp"] = "all"
        tmpl_col = "_TemplateTmp"

    # Gender can come from Gender column or Pronoun column
    if gender_col is None and pron_col is None:
        raise ValueError(
            f"[{model_name}] Missing Gender/Pronoun column. Need either 'Gender' or a pronoun column like "
            f"'Selected Pronoun'/'Pronoun'. Found columns: {list(df.columns)}"
        )

    out = df.copy()
    out["Model"] = model_name
    out["Profession"] = out[prof_col].astype(str).str.strip()
    out["Template"] = out[tmpl_col].apply(norm_template)

    # Build Gender
    if gender_col is not None:
        out["Gender"] = out[gender_col].apply(pronoun_to_gender)
    else:
        out["Gender"] = out[pron_col].apply(pronoun_to_gender)

    # Keep useful context columns if present (optional)
    # Bio/text columns sometimes exist:
    bio_col = pick_first_existing(out, ["bio", "Bio", "prompt", "Prompt", "text", "Text"])
    if bio_col is not None and bio_col != "bio":
        out["bio"] = out[bio_col]

    # Return canonical subset + keep original columns for download/audit
    # but ensure canonical columns appear first
    canonical_first = ["Model", "Profession", "Template", "Gender"]
    remaining = [c for c in out.columns if c not in canonical_first]
    return out[canonical_first + remaining]


@st.cache_data(show_spinner=False)
def load_all_data() -> pd.DataFrame:
    llama = load_and_normalise(LLAMA_PATH, MODEL_LLAMA)
    gemma = load_and_normalise(GEMMA_PATH, MODEL_GEMMA)
    return pd.concat([llama, gemma], ignore_index=True)


def filter_and_sample(df: pd.DataFrame, profession: str, template: str, n: int, seed: int, model: str | None) -> pd.DataFrame:
    sub = df[df["Profession"] == profession].copy()
    if template != "all":
        sub = sub[sub["Template"] == template]
    if model is not None:
        sub = sub[sub["Model"] == model]

    if len(sub) == 0:
        return sub

    sub = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sub.iloc[: min(n, len(sub))].copy()


def make_bar_chart(labels, values, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title(title)
    return fig


# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
with st.spinner("Loading results…"):
    try:
        df_all = load_all_data()
    except Exception as e:
        st.error("Could not load your results files.")
        st.code(str(e))
        st.info(
            "Make sure these files exist in your repo:\n"
            "- data/llama_results.csv\n"
            "- data/gemma_results.csv"
        )
        st.stop()

# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
st.sidebar.header("Controls")

compare = st.sidebar.checkbox("Compare both models", value=True)

model_choice = st.sidebar.selectbox(
    "Model",
    options=[MODEL_LLAMA, MODEL_GEMMA],
    disabled=compare
)

all_professions = sorted(df_all["Profession"].dropna().unique().tolist())
profession = st.sidebar.selectbox("Profession", options=all_professions)

templates_present = sorted([t for t in df_all["Template"].dropna().unique().tolist() if t])
template = st.sidebar.selectbox("Prompt type", options=["all"] + templates_present)

n = st.sidebar.slider("Sample size", min_value=1, max_value=20, value=20)
seed = st.sidebar.number_input("Shuffle seed", min_value=0, max_value=50_000_000, value=42, step=1)

run_summary = st.sidebar.button("Run across all professions (summary)")

# ------------------------------------------------------------
# Main View
# ------------------------------------------------------------
colA, colB = st.columns([1, 1])

if run_summary:
    st.subheader("Summary across all professions")

    if compare:
        rows = []
        for model in [MODEL_LLAMA, MODEL_GEMMA]:
            for prof in all_professions:
                sub = filter_and_sample(df_all, prof, template, n=20, seed=int(seed), model=model)
                if len(sub) == 0:
                    continue
                p = pct_from_gender(sub["Gender"])
                rows.append({
                    "Model": model,
                    "Profession": prof,
                    **p,
                    "Available": int(len(df_all[(df_all["Profession"] == prof) & ((df_all["Template"] == template) if template != "all" else True) & (df_all["Model"] == model)])),
                    "Used": int(len(sub)),
                })

        if not rows:
            st.warning("No rows found for this filter.")
        else:
            summary_df = pd.DataFrame(rows).sort_values(["Model", "Male %"], ascending=[True, False])
            st.dataframe(summary_df, use_container_width=True)

            st.download_button(
                "Download summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="occubias_summary_compare.csv",
                mime="text/csv",
            )
    else:
        rows = []
        for prof in all_professions:
            sub = filter_and_sample(df_all, prof, template, n=20, seed=int(seed), model=model_choice)
            if len(sub) == 0:
                continue
            p = pct_from_gender(sub["Gender"])
            rows.append({
                "Profession": prof,
                **p,
                "Available": int(len(df_all[(df_all["Profession"] == prof) & ((df_all["Template"] == template) if template != "all" else True) & (df_all["Model"] == model_choice)])),
                "Used": int(len(sub)),
            })

        if not rows:
            st.warning("No rows found for this filter.")
        else:
            summary_df = pd.DataFrame(rows).sort_values("Male %", ascending=False)
            st.dataframe(summary_df, use_container_width=True)

            st.download_button(
                "Download summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="occubias_summary_single.csv",
                mime="text/csv",
            )

else:
    # Single profession view
    st.subheader("Evaluation (filtered view)")

    if compare:
        m1, m2 = MODEL_LLAMA, MODEL_GEMMA

        s1 = filter_and_sample(df_all, profession, template, n=int(n), seed=int(seed), model=m1)
        s2 = filter_and_sample(df_all, profession, template, n=int(n), seed=int(seed), model=m2)

        with colA:
            st.markdown(f"### {m1}")
            st.write(f"Rows used: **{len(s1)}**")
            p1 = pct_from_gender(s1["Gender"]) if len(s1) else {"Male %": 0.0, "Female %": 0.0, "Neutral %": 0.0}
            st.write(p1)
            fig1 = make_bar_chart(["Male", "Female", "Neutral"], [p1["Male %"], p1["Female %"], p1["Neutral %"]], f"{m1} — {profession} ({template})")
            st.pyplot(fig1)

        with colB:
            st.markdown(f"### {m2}")
            st.write(f"Rows used: **{len(s2)}**")
            p2 = pct_from_gender(s2["Gender"]) if len(s2) else {"Male %": 0.0, "Female %": 0.0, "Neutral %": 0.0}
            st.write(p2)
            fig2 = make_bar_chart(["Male", "Female", "Neutral"], [p2["Male %"], p2["Female %"], p2["Neutral %"]], f"{m2} — {profession} ({template})")
            st.pyplot(fig2)

        st.markdown("### Sampled rows (combined)")
        combined = []
        if len(s1):
            t = s1.copy()
            t["Model"] = m1
            combined.append(t)
        if len(s2):
            t = s2.copy()
            t["Model"] = m2
            combined.append(t)

        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            st.dataframe(combined_df.head(50), use_container_width=True)

            st.download_button(
                "Download sampled CSV",
                data=combined_df.to_csv(index=False).encode("utf-8"),
                file_name="occubias_sampled_compare.csv",
                mime="text/csv",
            )
        else:
            st.warning("No rows found for these filters.")

    else:
        s = filter_and_sample(df_all, profession, template, n=int(n), seed=int(seed), model=model_choice)
        st.write(f"Model: **{model_choice}**")
        st.write(f"Rows used: **{len(s)}**")

        if len(s) == 0:
            st.warning("No rows found for these filters.")
        else:
            p = pct_from_gender(s["Gender"])
            fig = make_bar_chart(["Male", "Female", "Neutral"], [p["Male %"], p["Female %"], p["Neutral %"]], f"{model_choice} — {profession} ({template})")
            st.pyplot(fig)

            st.markdown("### Sampled rows")
            st.dataframe(s.head(50), use_container_width=True)

            st.download_button(
                "Download sampled CSV",
                data=s.to_csv(index=False).encode("utf-8"),
                file_name="occubias_sampled_single.csv",
                mime="text/csv",
            )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption(" Mini Version of Occupational Bias Evaluation.")
