import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("🧠 Motor Vibration Fault Diagnosis (FFT-Based)")

uploaded_file = st.file_uploader("📂 Upload Excel file (with T(X), X, T(Y), Y, T(Z), Z columns)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.rename(columns={'T(X)': 't', 'X': 'x', 'T(Y)': 't_y', 'Y': 'y', 'T(Z)': 't_z', 'Z': 'z'})
    df = df[['t', 'x', 'y', 'z']].dropna()
    df['t'] = pd.to_datetime(df['t'])

    # Sample rate
    time_deltas = df['t'].diff().dt.total_seconds().dropna()
    sample_interval = time_deltas.median()
    fs = 1 / sample_interval
    st.success(f"📏 Sampling Rate: {fs:.2f} Hz")

    # User Inputs
    rpm = st.number_input("🔧 Enter motor RPM", min_value=1.0, step=1.0)
    orientation = st.selectbox("🏗️ Installation Orientation", ["Horizontal", "Vertical"])
    axial_axis = st.selectbox("📐 Which axis is axial?", ['x', 'y', 'z'])
    bearing_info = st.text_input("🔩 Optional bearing geometry (e.g., BPFO=300,BPFI=250)")

    # Parse bearing info
    bearing_freqs = {}
    if bearing_info:
        try:
            for part in bearing_info.replace(' ', '').split(','):
                k, v = part.split('=')
                bearing_freqs[k.upper()] = float(v)
        except:
            st.warning("⚠️ Invalid bearing info format. Skipping bearing diagnosis.")

    rpm_freq = rpm / 60

    # Fault detection and plotting
    def analyze_fft(signal, axis):
        N = len(signal)
        yf = np.abs(rfft(signal - np.mean(signal)))
        xf = rfftfreq(N, 1 / fs)
        peaks, _ = find_peaks(yf, height=np.max(yf) * 0.1)

        fault_labels = []
        for f in xf[peaks]:
            if abs(f - rpm_freq) < 0.1 * rpm_freq:
                fault_labels.append(("Unbalance", f))
            elif abs(f - 2 * rpm_freq) < 0.1 * rpm_freq:
                fault_labels.append(("Misalignment", f))
            elif any(abs(f - n * rpm_freq) < 0.1 * rpm_freq for n in [3, 4]):
                fault_labels.append(("Looseness", f))
            for name, val in bearing_freqs.items():
                if abs(f - val) < 0.1 * val:
                    fault_labels.append((f"Bearing ({name})", f))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xf, yf)
        for label, freq in fault_labels:
            ax.axvline(freq, color='r', linestyle='--')
            ax.text(freq, max(yf)*0.8, f"{label} ({freq:.1f}Hz)", rotation=90, color='red')
        ax.set_title(f"FFT - {axis.upper()} axis")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)

        # Text summary
        if fault_labels:
            for label, freq in fault_labels:
                st.markdown(f"- **{axis.upper()} axis** → ⚠️ {label} at {freq:.1f} Hz")
        else:
            st.markdown(f"- **{axis.upper()} axis** → ✅ No clear fault patterns detected.")

    st.header("📊 FFT Analysis & Fault Diagnosis")

    for axis in ['x', 'y', 'z']:
        st.subheader(f"🔎 {axis.upper()} Axis Analysis ({'Axial' if axis == axial_axis else 'Radial'})")
        analyze_fft(df[axis], axis)

    # Explanation
    with st.expander("ℹ️ Why orientation and axis labels matter"):
        st.write("""
        - **Installation orientation** affects how faults like looseness or misalignment manifest.  
          For example, vertical motors experience gravity differently from horizontal ones.
        - **Axial axis** (e.g., Z) helps identify axial vs radial loads. Bearing and misalignment issues often show in axial direction.
        """)

else:
    st.info("⬆️ Please upload a vibration Excel file to start.")
