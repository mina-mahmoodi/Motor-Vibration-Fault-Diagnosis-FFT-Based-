import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

st.set_page_config(page_title="Motor RMS Fault Diagnosis", layout="wide")
st.title("🔍 Motor Fault Diagnosis using RMS Vibration Data")

uploaded_file = st.file_uploader("📂 Upload your Excel vibration dataset", type=["xlsx"])

def generate_pdf(df, sheet_name):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph(f"Motor RMS Vibration Diagnosis Report - Asset Sheet: {sheet_name}", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    explanation = """
    <b>Diagnosis Logic:</b><br/>
    - Radial RMS > 0.5 indicates possible <i>unbalance or radial misalignment</i>.<br/>
    - Axial RMS > 0.35 may signal <i>axial load or axial misalignment</i>.<br/>
    - Significant difference between X and Y RMS (>0.2) may indicate <i>mechanical looseness</i>.<br/>
    These thresholds are derived from general industrial practice and indicative trends.<br/>
    """
    elements.append(Paragraph(explanation, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Prepare table data with headers
    data = [['Timestamp', 'X RMS', 'Y RMS', 'Z RMS', 'Diagnosis']]
    for _, row in df.iterrows():
        data.append([
            row['t'].strftime("%Y-%m-%d %H:%M:%S"),
            f"{row['x_rms']:.3f}",
            f"{row['y_rms']:.3f}",
            f"{row['z_rms']:.3f}",
            row['Diagnosis']
        ])

    # Create table
    table = Table(data, repeatRows=1)
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#d3d3d3')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ])
    table.setStyle(style)
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("📑 Select asset sheet", xls.sheet_names)
    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    lower_map = {c.lower(): c for c in df_raw.columns}
    df = df_raw.rename(columns={orig: lower for lower, orig in lower_map.items()})

    expected = ['t(x)', 'x', 't(y)', 'y', 't(z)', 'z']
    miss = [c for c in expected if c not in df.columns]
    if miss:
        st.warning(f"⚠️ Missing columns: {miss}")
        st.stop()

    st.markdown("""
    ### ⚙️ Select Axial Axis
    Please select the axis that aligns with the motor shaft (axial direction).
    This helps distinguish between axial and radial vibrations.
    """)

    axial_axis = st.selectbox("Select AXIAL axis", ['x', 'y', 'z'], index=2)
    axis_map = {'x': ('t(x)', 'x'), 'y': ('t(y)', 'y'), 'z': ('t(z)', 'z')}

    axial_t, axial_v = axis_map[axial_axis]
    radials = [a for a in ['x', 'y', 'z'] if a != axial_axis]
    df_use = df[[axial_t, axial_v] + [axis_map[a][1] for a in radials]].dropna()
    df_use.columns = ['t', 'z', 'x', 'y']
    df_use['t'] = pd.to_datetime(df_use['t'], errors='coerce')
    df_use = df_use.dropna(subset=['t']).sort_values('t')

    orientation = st.radio("Machine orientation", ['Horizontal', 'Vertical'])

    st.markdown("""
    ### 🗓️ Select Diagnosis Period
    The diagnosis period is based on the calendar time inferred from the dataset.<br/>
    For example, "Last 24 hours" means the 24-hour period leading up to the <i>most recent timestamp</i> in the data.
    """, unsafe_allow_html=True)

    period = st.radio("Diagnosis period", ['Last 24 hours', 'Last 7 days', 'All data'])
    end_t = df_use['t'].max()
    start_t = end_t - pd.Timedelta(days=1) if period == 'Last 24 hours' \
        else end_t - pd.Timedelta(days=7) if period == 'Last 7 days' \
        else df_use['t'].min()

    df_filt = df_use[df_use['t'] >= start_t].copy()
    st.write(f"Points in selected period: **{len(df_filt)}**")

    if st.button("▶️ Run Diagnosis"):
        if df_filt.empty:
            st.error("No data in selected period.")
            st.stop()

        # Use fixed rolling window size (10 samples)
        win = 10
        for axis in ['x', 'y', 'z']:
            df_filt[f'{axis}_rms'] = df_filt[axis].rolling(
                window=win, min_periods=1
            ).apply(lambda v: np.sqrt(np.mean(v**2)), raw=True)

        def diag(r):
            findings = []
            if r['x_rms'] > 0.5 or r['y_rms'] > 0.5:
                findings.append("🔧 Radial high (unbalance / misalignment)")
            if r['z_rms'] > 0.35:
                findings.append("📏 Axial high (axial load / misalignment)")
            if abs(r['x_rms'] - r['y_rms']) > 0.2:
                findings.append("🔩 Looseness (radial diff)")
            return "✅ Normal" if not findings else ", ".join(findings)

        df_filt['Diagnosis'] = df_filt.apply(diag, axis=1)

        st.subheader("📋 Diagnosis (last 50 rows)")
        st.dataframe(df_filt[['t', 'x_rms', 'y_rms', 'z_rms', 'Diagnosis']].tail(50))

        pdf_buffer = generate_pdf(df_filt.tail(50), sheet_name)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"rms_diagnosis_{sheet_name}.pdf",
            mime="application/pdf"
        )
