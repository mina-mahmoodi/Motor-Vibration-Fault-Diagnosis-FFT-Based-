import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Motor RMS Fault Diagnosis", layout="wide")
st.title("🧠 Motor Fault Diagnosis using RMS Vibration Data")

uploaded_file = st.file_uploader("📂 Upload your Excel vibration dataset", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("📑 Select the asset sheet from Excel", xls.sheet_names)

    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    # Orientation explanation
    st.markdown("""
    ### 🧭 Machine Orientation
    - **Horizontal**: Common for pumps, fans, blowers.
    - **Vertical**: Often seen in vertical shaft pumps, mixers.
    """)

    orientation = st.radio("Select Machine Orientation", ['Horizontal', 'Vertical'])

    # Axis explanation
    st.markdown("""
    ### 📌 Axis Direction Selection
    - Choose columns that match sensor directions:
        - **Z** → Axial (along motor shaft)
        - **X/Y** → Radial (perpendicular to shaft)
    - Make sure selected columns contain numeric RMS values.
    """)

    x_label = st.selectbox("Select column for X (Radial)", df.columns)
    y_label = st.selectbox("Select column for Y (Radial)", df.columns)
    z_label = st.selectbox("Select column for Z (Axial)", df.columns)
    t_label = st.selectbox("Select column for Timestamp", df.columns)

    rpm = st.number_input("🔁 Enter motor RPM", min_value=100, max_value=3600, step=10)

    # Prepare data
    df_use = df[[t_label, x_label, y_label, z_label]].dropna()
    df_use.columns = ['t', 'x', 'y', 'z']

    try:
        df_use['t'] = pd.to_datetime(df_use['t'], errors='coerce')
        df_use = df_use.dropna(subset=['t'])
        df_use = df_use.sort_values('t')

        # Sample rate calculation using full dataset
        time_deltas = df_use['t'].diff().dt.total_seconds().dropna()
        median_interval = time_deltas.median()
        if median_interval and median_interval > 0:
            sample_rate = round(1 / median_interval, 5)
            sample_interval = round(median_interval, 2)
            st.success(f"📈 Sample Rate ≈ {sample_rate} Hz (1 sample every {sample_interval} seconds)")
        else:
            sample_rate = 0
            st.warning("⚠️ Could not compute a valid sample rate.")
    except Exception as e:
        sample_rate = 0
        st.error(f"❌ Failed to parse timestamp or calculate sample rate: {e}")

    # RMS-based rolling std deviation
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

    # Display results
    st.subheader("📋 Diagnosis Results Based on RMS Data")
    st.dataframe(df_use[['t', 'x', 'y', 'z', 'diagnosis']])

    st.subheader("📊 Vibration Trend Chart")
    st.line_chart(df_use.set_index('t')[['x', 'y', 'z']])

    st.markdown("""
    ---
    ### ℹ️ Notes:
    - RMS-based diagnosis is suitable for **slow-developing faults**:
        - Unbalance, looseness, soft foot, misalignment, degradation
    - For precise bearing fault or cavitation diagnosis, **FFT waveform analysis is needed**
    - Add more context (e.g., bearing geometry) in future updates for deeper diagnosis
    """)
