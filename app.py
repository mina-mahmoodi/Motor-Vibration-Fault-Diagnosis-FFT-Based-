import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Motor RMS Fault Diagnosis", layout="wide")
st.title("🔍 Motor Fault Diagnosis using RMS Vibration Data")

uploaded_file = st.file_uploader("📂 Upload your Excel vibration dataset", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("📑 Select the asset sheet from Excel", xls.sheet_names)

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    # Explanation for choosing orientation
    st.markdown("""
    **🧭 Machine Orientation Explanation:**
    - Choose *Horizontal* for machines like centrifugal pumps, horizontally-mounted motors.
    - Choose *Vertical* for machines like vertical pumps or vertically-mounted motors.
    """)

    orientation = st.radio("Select Machine Orientation", ['Horizontal', 'Vertical'])

    # Axis explanation
    st.markdown("""
    **📌 Axis Direction Explanation:**
    - Typically:
        - **Z** = Axial (along shaft)
        - **X, Y** = Radial (perpendicular to shaft)
    - Use naming based on your sensor placement.
    """)

    x_label = st.selectbox("Select column for X axis (Radial)", df.columns)
    y_label = st.selectbox("Select column for Y axis (Radial)", df.columns)
    z_label = st.selectbox("Select column for Z axis (Axial)", df.columns)
    t_label = st.selectbox("Select column for Timestamp", df.columns)

    rpm = st.number_input("🔁 Enter motor RPM", min_value=100, max_value=3600, step=10)

    df_use = df[[t_label, x_label, y_label, z_label]].dropna()
    df_use.columns = ['t', 'x', 'y', 'z']
    df_use['t'] = pd.to_datetime(df_use['t'])

     # Sample rate calculation with median delta
    df_use = df_use.sort_values('t')
    try:
        df_use['t'] = pd.to_datetime(df_use['t'], errors='coerce')
        df_use = df_use.dropna(subset=['t'])  # drop rows where timestamp failed to parse

        time_deltas = df_use['t'].diff().dt.total_seconds()
        median_interval = time_deltas.median()
        sample_rate = 1 / median_interval if median_interval and median_interval > 0 else 0

        if sample_rate:
            st.info(f"⏱️ Sample Rate: {sample_rate:.5f} Hz ({1/sample_rate:.2f} seconds/sample)")
        else:
            st.warning("⚠️ Sample rate calculated as 0. This might be due to insufficient or invalid timestamps.")
    except Exception as e:
        st.error(f"❌ Error parsing timestamps: {e}")
        sample_rate = 0

    st.info(f"⏱️ Sample Rate: {sample_rate:.5f} Hz ({1/sample_rate:.2f} seconds/sample)" if sample_rate else "❗ Sample rate could not be determined.")

    # Rolling std deviation
    window_size = 3
    for axis in ['x', 'y', 'z']:
        df_use[f'{axis}_std'] = df_use[axis].rolling(window=window_size).std()

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
