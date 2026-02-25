import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="OccuBias: Occupational LLM Evaluation", layout="wide")

st.title("OccuBias: Occupational LLM Evaluation")
st.markdown("Interactive exploration of occupational gender bias in LLM outputs.")

# ----------------------------
# Demo Data (Replace later with full dataset if needed)
# ----------------------------
np.random.seed(42)

professions = ["Doctor", "Engineer", "Teacher", "Nurse", "Lawyer"]
models = ["Llama-3-8B-Instruct", "Gemma-7B-IT"]
templates = ["neutral", "ambiguous", "stereotype"]

data = []
for model in models:
    for prof in professions:
        for tmpl in templates:
            for i in range(20):
                gender = np.random.choice(["Male", "Female", "Neutral"], p=[0.5, 0.4, 0.1])
                data.append([model, prof, tmpl, gender])

df = pd.DataFrame(data, columns=["Model", "Profession", "Template", "Gender"])

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Controls")

compare = st.sidebar.checkbox("Compare both models", value=True)

selected_model = st.sidebar.selectbox("Model", models)
selected_profession = st.sidebar.selectbox("Profession", professions)
selected_template = st.sidebar.selectbox("Prompt Type", ["all"] + templates)

seed = st.sidebar.number_input("Shuffle Seed", value=42)

# ----------------------------
# Filtering
# ----------------------------
def filter_data(dataframe, model, profession, template):
    sub = dataframe[dataframe["Profession"] == profession]
    if template != "all":
        sub = sub[sub["Template"] == template]
    if not compare:
        sub = sub[sub["Model"] == model]
    return sub.sample(frac=1, random_state=int(seed))

filtered = filter_data(df, selected_model, selected_profession, selected_template)

# ----------------------------
# Results
# ----------------------------
st.subheader("Results")

if compare:
    summary = filtered.groupby(["Model", "Gender"]).size().unstack(fill_value=0)
else:
    summary = filtered.groupby("Gender").size()

st.dataframe(summary)

# ----------------------------
# Chart
# ----------------------------
st.subheader("Gender Distribution")

fig, ax = plt.subplots()

if compare:
    summary.plot(kind="bar", ax=ax)
else:
    summary.plot(kind="bar", ax=ax)

ax.set_ylabel("Count")
st.pyplot(fig)

# ----------------------------
# Download
# ----------------------------
st.download_button(
    "Download Filtered Data",
    data=filtered.to_csv(index=False),
    file_name="occubias_filtered_results.csv",
    mime="text/csv"
)
