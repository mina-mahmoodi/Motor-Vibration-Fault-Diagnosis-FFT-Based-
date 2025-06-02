import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Motor RMS Fault Diagnosis", layout="wide")
st.title("🔍 Motor Fault Diagnosis using RMS Vibration Data")

# Upload Excel file
uploaded_file = st.file_uploader("📂 Upload your Excel vibration dataset (RMS data)", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("📑 Select the asset sheet from Excel", xls.sheet_names)

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    with st.form("diagnosis_form"):
        # Orientation explanation
        st.markdown("""
        ### 🧭 Machine Orientation Explanation:
        - Choose **Horizontal** for machines like centrifugal pumps, horizontally-mounted motors.
        - Choose **Vertical** for vertical pumps or vertically-mounted motors.
        """)
        orientation = st.radio("Select Machine Orientation", ['Horizontal', 'Vertical'])

        # Axis explanation
        st.markdown("""
        ### 📌 Axis Direction Explanation:
        - Typically:
            - **Z** = Axial (along shaft)
            - **X, Y** = Radial (perpendicular to shaft)
        - Choose based on how your sensors are mounted on the machine.
        """)
        x_label = st.selectbox("Select column for X axis (Radial)", df.columns)
        y_label = st.selectbox("Select column for Y axis (Radial)", df.columns)
        z_label = st.selectbox("Select column for Z axis (Axial)", df.columns)
        t_label = st.selectbox("Select column for Timestamp", df.columns)

        # RPM
        rpm = st.number_input("🔁 Enter motor RPM", min_value=100, max_value=3600, step=10)

        # Submit
        submitted = st.form_submit_button("✅ Run Diagnosis")

    if submitted:
        df_use = df[[t_label, x_label, y_label, z_label]].dropna()
        df_use.columns = ['t', 'x', 'y', 'z']
        df_use['t'] = pd.to_datetime(df_use['t'], errors='coerce')
        df_use = df_use.dropna(subset=['t']).sort_values('t')

        # Rolling std deviation
        window_size = 3
        for axis in ['x', 'y', 'z']:
            df_use[f'{axis}_std'] = df_use[axis].rolling(window=window_size).std()

        # Diagnosis function
        def diagnose(row):
            faults = []
            if row['x'] > 0.5 or row['y'] > 0.5:
                faults.append('🔧 Possible Unbalance or Misalignment')
            if row['z'] > 0.35:
                faults.append('📏 Axial Load or Axial Misalignment')
            if row['x_std'] > 0.05 or row['y_std'] > 0.05 or row['z_std'] > 0.05:
                faults.append('🔩 Looseness or Variable Load')
            return ', '.join(faults) if faults else '✅ Normal'

        df_use['diagnosis'] = df_use.apply(diagnose, axis=1)

        st.subheader("📋 Diagnosed Results")
        st.dataframe(df_use[['t', 'x', 'y', 'z', 'diagnosis']])

        st.line_chart(df_use.set_index('t')[['x', 'y', 'z']])
