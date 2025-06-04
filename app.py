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

def diagnose_faults(df, sample_rate):
    win = max(1, int(sample_rate * 60)) if sample_rate else 10
    for axis in ['x', 'y', 'z']:
        df[f'{axis}_rms'] = df[axis].rolling(
            window=win, min_periods=1
        ).apply(lambda v: np.sqrt(np.mean(v**2)), raw=True)

    def diag(r):
        f = []
        if r['x_rms'] > 0.5 or r['y_rms'] > 0.5: f.append("Radial High")
        if r['z_rms'] > 0.35: f.append("Axial High")
        if abs(r['x_rms'] - r['y_rms']) > 0.2: f.append("Looseness")
        return ", ".join(f) if f else None

    df['Diagnosis'] = df.apply(diag, axis=1)
    return df[df['Diagnosis'].notna()][['t', 'Diagnosis']]

def generate_summary_pdf(diagnosis_summary):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("⚠️ Summary of Issues Found Across All Sheets", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    data = [['Asset Sheet', 'Timestamp', 'Issue Detected']]
    for entry in diagnosis_summary:
        for _, row in entry['faults'].iterrows():
            data.append([entry['sheet'], row['t'].strftime("%Y-%m-%d %H:%M:%S"), row['Diagnosis']])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def prepare_sheet(df_raw, axial_axis='z'):
    lower_map = {c.lower(): c for c in df_raw.columns}
    df = df_raw.rename(columns={orig: lower for lower, orig in lower_map.items()})
    expected = ['t(x)', 'x', 't(y)', 'y', 't(z)', 'z']
    if any(c not in df.columns for c in expected):
        return None

    axis_map = {'x': ('t(x)', 'x'), 'y': ('t(y)', 'y'), 'z': ('t(z)', 'z')}
    axial_t, axial_v = axis_map[axial_axis]
    radials = [a for a in ['x', 'y', 'z'] if a != axial_axis]
    df_use = df[[axial_t, axial_v] + [axis_map[a][1] for a in radials]].dropna()
    df_use.columns = ['t', 'z', 'x', 'y']
    df_use['t'] = pd.to_datetime(df_use['t'], errors='coerce')
    df_use = df_use.dropna(subset=['t']).sort_values('t')

    dt = df_use['t'].diff().dt.total_seconds().median()
    sr = 1 / dt if dt and dt > 0 else 0
    return df_use, sr

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_sheets_mode = st.checkbox("🌀 Diagnose all sheets at once")
    
    if all_sheets_mode:
        if st.button("▶️ Run Diagnosis for All Sheets"):
            diagnosis_summary = []
            for sheet in xls.sheet_names:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)
                processed = prepare_sheet(df_raw)
                if not processed:
                    continue
                df_use, sr = processed
                faults = diagnose_faults(df_use.copy(), sr)
                if not faults.empty:
                    diagnosis_summary.append({'sheet': sheet, 'faults': faults})

            if not diagnosis_summary:
                st.success("✅ No issues detected in any sheet.")
            else:
                st.subheader("⚠️ Issues Detected")
                for entry in diagnosis_summary:
                    st.write(f"📑 **Sheet:** {entry['sheet']}")
                    st.dataframe(entry['faults'])

                pdf_buffer = generate_summary_pdf(diagnosis_summary)
                st.download_button(
                    label="📥 Download PDF Summary Report",
                    data=pdf_buffer,
                    file_name="summary_faults_report.pdf",
                    mime="application/pdf"
                )
    else:
        sheet_name = st.selectbox("📑 Select asset sheet", xls.sheet_names)
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        processed = prepare_sheet(df_raw)
        if not processed:
            st.warning("Sheet is missing required columns.")
        else:
            df_use, sr = processed
            st.info(f"Sample rate ≈ {sr:.2f} Hz")

            duration = st.radio("Select data window", ['Last 24 hours', 'Last 7 days', 'All data'])
            latest = df_use['t'].max()
            start = latest - pd.Timedelta(days=1) if duration == 'Last 24 hours' \
                else latest - pd.Timedelta(days=7) if duration == 'Last 7 days' \
                else df_use['t'].min()

            df_filtered = df_use[df_use['t'] >= start]
            st.write(f"Data points selected: **{len(df_filtered)}**")

            if st.button("▶️ Run Diagnosis"):
                faults = diagnose_faults(df_filtered.copy(), sr)
                st.subheader("📋 Diagnosis (last 50 rows with issues)")
                st.dataframe(faults.tail(50))

                def generate_pdf_one_sheet(faults, sheet_name):
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elements = []

                    title = Paragraph(f"Motor Diagnosis Report - {sheet_name}", styles['Title'])
                    elements.append(title)
                    elements.append(Spacer(1, 12))

                    data = [['Timestamp', 'Diagnosis']]
                    for _, row in faults.iterrows():
                        data.append([row['t'].strftime("%Y-%m-%d %H:%M:%S"), row['Diagnosis']])

                    table = Table(data, repeatRows=1)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ]))
                    elements.append(table)
                    doc.build(elements)
                    buffer.seek(0)
                    return buffer

                pdf_buffer = generate_pdf_one_sheet(faults.tail(50), sheet_name)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"diagnosis_{sheet_name}.pdf",
                    mime="application/pdf"
                )
