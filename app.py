import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Motor RMS Fault Diagnosis", layout="wide")
st.title("🔍 Motor Fault Diagnosis using RMS Vibration Data")

uploaded_file = st.file_uploader("📂 Upload your Excel vibration dataset", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("📑 Select the asset sheet from Excel", xls.sheet_names)
    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    # Create a lowercase column mapping dictionary
    lower_col_map = {col.lower(): col for col in df_raw.columns}
    df = df_raw.rename(columns={v: k.lower() for k, v in lower_col_map.items()})

    expected_cols = ['t(x)', 'x', 't(y)', 'y', 't(z)', 'z']
    missing_cols = [col for col in expected_cols if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Missing expected columns in sheet: {missing_cols}")
    else:
        axial_axis = st.selectbox(
            "Select which axis is AXIAL vibration (along shaft):",
            options=['x', 'y', 'z'],
            index=2,
            help="The other two axes will be considered radial."
        )

        axis_map = {
            'x': ('t(x)', 'x'),
            'y': ('t(y)', 'y'),
            'z': ('t(z)', 'z')
        }

        axial_t_col, axial_v_col = axis_map[axial_axis]
        radial_axes = [a for a in ['x', 'y', 'z'] if a != axial_axis]
        radial_t_cols = [axis_map[a][0] for a in radial_axes]
        radial_v_cols = [axis_map[a][1] for a in radial_axes]

        selected_cols = [axial_t_col, axial_v_col] + radial_v_cols
        df_use = df[selected_cols].dropna()
        df_use.columns = ['t', 'z'] + ['x', 'y']  # standard order

        df_use['t'] = pd.to_datetime(df_use['t'], errors='coerce')
        df_use = df_use.dropna(subset=['t']).sort_values('t')

        time_deltas = df_use['t'].diff().dt.total_seconds()
        median_interval = time_deltas.median()
        sample_rate = 1 / median_interval if median_interval and median_interval > 0 else 0

        if sample_rate > 0:
            st.info(f"⏱️ Sample Rate: {sample_rate:.3f} Hz")
        else:
            st.warning("⚠️ Sample rate could not be calculated.")

        orientation = st.radio("Select Machine Orientation", ['Horizontal', 'Vertical'])
        rpm = st.number_input("🔁 Enter motor RPM", min_value=100, max_value=3600, step=10)

        duration_choice = st.radio("Select diagnosis period:", ['Last 24 hours', 'Last 7 days', 'All data'])
        latest_time = df_use['t'].max()

        if duration_choice == 'Last 24 hours':
            start_time = latest_time - pd.Timedelta(days=1)
        elif duration_choice == 'Last 7 days':
            start_time = latest_time - pd.Timedelta(days=7)
        else:
            start_time = df_use['t'].min()

        df_filtered = df_use[df_use['t'] >= start_time].copy()
        st.write(f"⏳ Data points selected: {len(df_filtered)}")

        if st.button("▶️ Run Diagnosis"):
            if df_filtered.empty:
                st.error("No data in selected range.")
            else:
                window_samples = max(1, int(sample_rate * 60))
                st.write(f"Using RMS window: {window_samples} samples (~{window_samples * median_interval:.1f} sec)")

                for axis in ['x', 'y', 'z']:
                    df_filtered[f'{axis}_rms'] = df_filtered[axis].rolling(
                        window=window_samples, min_periods=1
                    ).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)

                def diagnose_rms(row):
                    faults = []
                    if row['x_rms'] > 0.5 or row['y_rms'] > 0.5:
                        faults.append("🔧 Possible Unbalance or Misalignment (Radial)")
                    if row['z_rms'] > 0.35:
                        faults.append("📏 Axial Load or Axial Misalignment")
                    if abs(row['x_rms'] - row['y_rms']) > 0.2:
                        faults.append("🔩 Possible Looseness")
                    return "✅ Normal" if not faults else ", ".join(faults)

                df_filtered['Diagnosis'] = df_filtered.apply(diagnose_rms, axis=1)

                st.subheader("📋 Sample Diagnosis (last 50 rows)")
                st.dataframe(df_filtered[['t', 'x_rms', 'y_rms', 'z_rms', 'Diagnosis']].tail(50))

                st.subheader("📈 Vibration Trend")
                st.line_chart(df_filtered.set_index('t')[['x_rms', 'y_rms', 'z_rms']])
