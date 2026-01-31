import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ag-IoT IDS Tool", layout="centered")

st.title("🌱 Ag-IoT Intrusion Detection System")
st.write("GAN-based Multiclass IDS for Agricultural IoT Networks")

st.subheader("Detection Mode")
mode = st.radio(
    "Select analysis mode:",
    ["DEMO (Ground Truth)", "IDS (Model Prediction)"]
)

selected_mode = "DEMO" if "DEMO" in mode else "IDS"


uploaded_file = st.file_uploader("Upload FarmFlow CSV file", type=["csv"])

if uploaded_file:
    st.success("CSV uploaded")

    if st.button("Analyze Network Traffic"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={"file": uploaded_file},
                data={"mode": selected_mode}
            )



            if response.status_code != 200:
                st.error("Backend error")
                st.json(response.json())
            else:
                result = response.json()

                st.info(f"Detection Mode: {result.get('mode')}")

                st.write(f"**Total Records:** {result['total_records']}")

                df = pd.DataFrame(
                    result["attack_summary"].items(),
                    columns=["Attack Type", "Count"]
                )

                st.dataframe(df)

                st.subheader("Attack Distribution")
                fig, ax = plt.subplots()
                ax.bar(df["Attack Type"], df["Count"])
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

                fig2, ax2 = plt.subplots()
                ax2.pie(df["Count"], labels=df["Attack Type"], autopct="%1.1f%%")
                st.pyplot(fig2)

                st.info(result.get("note", ""))
