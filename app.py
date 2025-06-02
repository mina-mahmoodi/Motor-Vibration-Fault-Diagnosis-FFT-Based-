import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.fft import fft, fftfreq

st.set_page_config(page_title="Motor Vibration Diagnosis", layout="wide")
st.title("🛠️ Motor Vibration Fault Diagnosis (FFT-Based)")

# File uploader
uploaded_file = st.file_uploader("Upload Excel vibration data", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        required_cols = ['T(X)', 'X', 'T(Y)', 'Y', 'T(Z)', 'Z']
        if not all(col in df.columns for col in required_cols):
            st.error("Excel file must include these columns: T(X), X, T(Y), Y, T(Z), Z")
            st.stop()

        # Rename for simplicity
        df = df.rename(columns={'T(X)': 't', 'X': 'x', 'T(Y)': 'ty', 'Y': 'y', 'T(Z)': 'tz', 'Z': 'z'})

        # Drop rows with missing data
        df = df[['t', 'x', 'y', 'z']].dropna()

        # Estimate sample rate from time column
        df['t'] = pd.to_datetime(df['t'])
        time_deltas = df['t'].diff().dt.total_seconds().dropna()
        avg_dt = time_deltas.mean()
        sample_rate = 1 / avg_dt  # Hz
        st.info(f"Estimated sample rate: {sample_rate:.2f} Hz")

        # Ask for user inputs
        rpm = st.number_input("Motor speed (RPM)", min_value=100, max_value=10000, value=1500)
        orientation = st.selectbox("Motor installation orientation", ['Horizontal', 'Vertical'])
        axial_axis = st.selectbox("Which axis is Axial?", ['x', 'y', 'z'])

        # Perform FFT per axis
        N = len(df)
        fft_freqs = fftfreq(N, d=1/sample_rate)
        fft_freqs = fft_freqs[:N//2]  # One-sided spectrum

        axis_data = {}
        for axis in ['x', 'y', 'z']:
            signal = df[axis] - np.mean(df[axis])
            fft_vals = fft(signal)
            magnitude = 2.0 / N * np.abs(fft_vals[:N//2])
            axis_data[axis] = (fft_freqs, magnitude)

        st.subheader("🔍 FFT Results and Diagnosis")

        for axis in ['x', 'y', 'z']:
            freqs, mags = axis_data[axis]

            fig = px.line(x=freqs, y=mags, title=f"{axis.upper()} Axis FFT", labels={'x': 'Frequency (Hz)', 'y': 'Amplitude'})
            st.plotly_chart(fig, use_container_width=True)

            # Find peak frequency
            peak_freq = freqs[np.argmax(mags)]
            peak_rpm = peak_freq * 60
            st.write(f"**{axis.upper()} Axis peak frequency:** {peak_freq:.2f} Hz ({peak_rpm:.0f} RPM)")

            # Diagnosis logic
            diagnosis = "Unknown"
            if abs(peak_rpm - rpm) < 5:
                diagnosis = "Likely Unbalance"
            elif abs(peak_rpm - 2 * rpm) < 5:
                diagnosis = "Possible Misalignment"
            elif peak_freq > 500:
                diagnosis = "Possible Bearing Fault"
            elif 0 < peak_freq < 10 and mags.max() > 0.1:
                diagnosis = "Possible Looseness"
            else:
                diagnosis = "No dominant fault detected"

            dir_note = " (Axial)" if axis == axial_axis else " (Radial)"
            st.markdown(f"📌 **Diagnosis for {axis.upper()}{dir_note}:** {diagnosis}")

        st.markdown("---")
        st.markdown("### ℹ️ Guidance")

        st.markdown("""
        - **Motor Orientation**: Helps identify direction-sensitive faults like unbalance (more severe horizontally).
        - **Axis Labeling**: Typically, axial vibrations (along shaft) show signs of misalignment or thrust issues.
        - **Fault Patterns**:
            - 1× RPM → **Unbalance**
            - 2× RPM → **Misalignment**
            - High frequencies (500+ Hz) → **Bearing issues**
            - Sub-10 Hz with high amplitude → **Looseness**
        """)

    except Exception as e:
        st.error(f"Error processing file: {e}")
