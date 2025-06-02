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

    st.markdown("""
    ### Column Names Detected for Vibration Data:
    - Timestamp X: `T(X)`, X axis vibration: `X`
    - Timestamp Y: `T(Y)`, Y axis vibration: `Y`
    - Timestamp Z: `T(Z)`, Z axis vibration: `Z`
    """)

    # Find if columns exist, else show warning
    expected_cols = ['T(X)', 'X', 'T(Y)', 'Y', 'T(Z)', 'Z']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing expected columns in sheet: {missing_cols}")
    else:
        # Ask user to select which axis is axial
        axial_axis = st.selectbox(
            "Select which axis is AXIAL vibration (along shaft):",
            options=['X', 'Y', 'Z'],
            index=2,
            help="""
            Axial axis is along the shaft direction.
            Usually Z is axial.
            If your sensor setup differs, choose accordingly.
            The other two axes will be considered radial (perpendicular to shaft).
            """
        )

        # Determine corresponding timestamp columns for chosen axial and radials
        axis_map = {
            'X': ('T(X)', 'X'),
            'Y': ('T(Y)', 'Y'),
            'Z': ('T(Z)', 'Z')
        }
        axial_t_col, axial_v_col = axis_map[axial_axis]
        radial_axes = [a for a in ['X','Y','Z'] if a != axial_axis]
        radial_t_cols = [axis_map[a][0] for a in radial_axes]
        radial_v_cols = [axis_map[a][1] for a in radial_axes]

        # Confirm user that timestamps must be consistent (ideally)
        st.markdown("""
        **Note:** Timestamps should be consistent across axes. 
        The app will use the axial timestamp (`{}`) for timing analysis.
        """.format(axial_t_col))

        # Use axial timestamp and axial + radial vibration data
        data_cols = [axial_t_col, axial_v_col] + radial_v_cols
        df_use = df[data_cols].dropna()

        # Rename for simplicity
        df_use.columns = ['t', 'z'] + ['x', 'y']

        # Convert time column to datetime
        df_use['t'] = pd.to_datetime(df_use['t'], errors='coerce')
        df_use = df_use.dropna(subset=['t']).sort_values('t')

        # Calculate sample rate based on median delta
        time_deltas = df_use['t'].diff().dt.total_seconds()
        median_interval = time_deltas.median()
        sample_rate = 1 / median_interval if median_interval and median_interval > 0 else 0
        if sample_rate > 0:
            st.info(f"⏱️ Sample Rate: {sample_rate:.5f} Hz ({median_interval:.2f} seconds/sample)")
        else:
            st.warning("⚠️ Could not reliably calculate sample rate from timestamps.")

        # Machine orientation explanation
        st.markdown("""
        **🧭 Machine Orientation Explanation:**
        - *Horizontal*: machines like centrifugal pumps, horizontally-mounted motors.
        - *Vertical*: vertical pumps or vertically-mounted motors.
        """)
        orientation = st.radio("Select Machine Orientation", ['Horizontal', 'Vertical'])

        rpm = st.number_input("🔁 Enter motor RPM", min_value=100, max_value=3600, step=10)

        # Diagnosis duration choice
        st.markdown("### Choose diagnosis duration relative to the latest timestamp in data:")
        duration_choice = st.radio("Duration to analyze:", ['Last 24 hours', 'Last 7 days', 'All data'])

        latest_time = df_use['t'].max()
        if duration_choice == 'Last 24 hours':
            start_time = latest_time - pd.Timedelta(days=1)
        elif duration_choice == 'Last 7 days':
            start_time = latest_time - pd.Timedelta(days=7)
        else:
            start_time = df_use['t'].min()

        df_filtered = df_use[df_use['t'] >= start_time].copy()
        st.write(f"⏳ Data points in selected duration: {len(df_filtered)}")

        # Run diagnosis only on button press to avoid rerun lag
        if st.button("▶️ Run Diagnosis"):

            if df_filtered.empty:
                st.error("No data available in the selected duration.")
            else:
                # Calculate RMS values over rolling windows (e.g. 10 samples)
                window_samples = max(1, int(sample_rate * 60))  # ~1 min window or fallback 1 sample
                st.write(f"Using rolling RMS window: {window_samples} samples (~{window_samples * median_interval:.1f} seconds)")

                for axis in ['x','y','z']:
                    df_filtered[f'{axis}_rms'] = df_filtered[axis].rolling(window=window_samples, min_periods=1).apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True)

                # Simple fault diagnosis based on RMS thresholds
                def diagnose_rms(row):
                    faults = []
                    # Thresholds based on typical RMS vibration values - these can be tuned
                    if row['x_rms'] > 0.5 or row['y_rms'] > 0.5:
                        faults.append("🔧 Possible Unbalance or Misalignment (Radial axes high vibration)")
                    if row['z_rms'] > 0.35:
                        faults.append("📏 Axial Load or Axial Misalignment")
                    # Variability (difference between radial RMS)
                    radial_diff = abs(row['x_rms'] - row['y_rms'])
                    if radial_diff > 0.2:
                        faults.append("🔩 Possible Looseness or Variable Load (Radial axes imbalance)")
                    if not faults:
                        return "✅ Normal"
                    else:
                        return ", ".join(faults)

                df_filtered['Diagnosis'] = df_filtered.apply(diagnose_rms, axis=1)

                st.subheader("📋 Diagnosis Results Sample (Last 50 rows)")
                st.dataframe(df_filtered[['t', 'x_rms', 'y_rms', 'z_rms', 'Diagnosis']].tail(50))

                st.subheader("📈 Vibration RMS over Time")
                st.line_chart(df_filtered.set_index('t')[['x_rms', 'y_rms', 'z_rms']])

                st.markdown("""
                ---
                **Notes:**

                - Axial axis selected: `{}`

                - Other two axes considered radial.

                - Diagnosis thresholds are based on typical RMS values and may need calibration for your specific machine.

                - Rolling window size for RMS is based on sample rate to approximate ~1 minute window.

                """.format(axial_axis.upper()))

