# motor_rms_diagnosis_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit page config & title
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Motor RMS Fault Diagnosis", layout="wide")
st.title("🔍 Motor Fault Diagnosis using RMS Vibration Data")

# ────────────────────────────────────────────────────────────────────────────────
# File upload
# ────────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload your Excel vibration dataset", type=["xlsx"])

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────
def diagnose_faults(df: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    """Add rolling-window RMS columns and return only the rows with diagnoses."""
    win = max(1, int(sample_rate * 60)) if sample_rate else 10  # 60-second window
    for axis in ["x", "y", "z"]:
        df[f"{axis}_rms"] = (
            df[axis]
            .rolling(window=win, min_periods=1)
            .apply(lambda v: np.sqrt(np.mean(v ** 2)), raw=True)
        )

    def diag(row):
        issues = []
        if row["x_rms"] > 0.5 or row["y_rms"] > 0.5:
            issues.append("Radial High")
        if row["z_rms"] > 0.35:
            issues.append("Axial High")
        if abs(row["x_rms"] - row["y_rms"]) > 0.2:
            issues.append("Looseness")
        return ", ".join(issues) if issues else None

    df["Diagnosis"] = df.apply(diag, axis=1)
    return df.loc[df["Diagnosis"].notna(), ["t", "Diagnosis"]]


def generate_summary_pdf(diagnosis_summary: list[dict]) -> BytesIO:
    """Create a single-page PDF summary for all sheets with issues."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("⚠️ Summary of Issues Found Across All Sheets", styles["Title"]))
    elements.append(Spacer(1, 12))

    data = [["Asset Sheet", "Timestamp", "Issue Detected"]]
    for entry in diagnosis_summary:
        for _, row in entry["faults"].iterrows():
            data.append(
                [
                    entry["sheet"],
                    row["t"].strftime("%Y-%m-%d %H:%M:%S"),
                    row["Diagnosis"],
                ]
            )

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


def prepare_sheet(df_raw: pd.DataFrame, axial_axis: str = "z"):
    """Validate/rename columns, set dtypes & return (df_use, sample_rate)."""
    lower_map = {c.lower(): c for c in df_raw.columns}
    df = df_raw.rename(columns={orig: lower for lower, orig in lower_map.items()})
    expected = ["t(x)", "x", "t(y)", "y", "t(z)", "z"]
    if any(col not in df.columns for col in expected):
        return None  # invalid sheet

    axis_map = {"x": ("t(x)", "x"), "y": ("t(y)", "y"), "z": ("t(z)", "z")}
    axial_t, axial_v = axis_map[axial_axis]
    radials = [a for a in ["x", "y", "z"] if a != axial_axis]
    df_use = df[[axial_t, axial_v] + [axis_map[a][1] for a in radials]].dropna()
    df_use.columns = ["t", "z", "x", "y"]  # reorder so z is axial
    df_use["t"] = pd.to_datetime(df_use["t"], errors="coerce")
    df_use = df_use.dropna(subset=["t"]).sort_values("t")

    # Sample-rate estimation
    dt = df_use["t"].diff().dt.total_seconds().median()
    sr = 1 / dt if dt and dt > 0 else 0
    return df_use, sr


# ────────────────────────────────────────────────────────────────────────────────
# Main app logic
# ────────────────────────────────────────────────────────────────────────────────
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    all_sheets_mode = st.checkbox("🌀 Diagnose all sheets at once")

    # Allow the user to define N only when needed (separate control avoids clutter)
    if all_sheets_mode:
        max_rows = st.number_input(
            "🔢 Limit to most recent N rows per sheet",
            min_value=10,
            value=500,
            step=10,
        )

        if st.button("▶️ Run Diagnosis for All Sheets"):
            diagnosis_summary = []
            progress_bar = st.progress(0, text="Processing sheets…")
            total = len(xls.sheet_names)

            for idx, sheet in enumerate(xls.sheet_names, 1):
                try:
                    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)
                    prepared = prepare_sheet(df_raw)
                    if not prepared:
                        continue  # silently skip invalid sheets

                    df_use, sr = prepared
                    # ─ Limit to most-recent N rows
                    df_use = df_use.sort_values("t").tail(max_rows)

                    faults = diagnose_faults(df_use.copy(), sr)
                    if not faults.empty:
                        diagnosis_summary.append({"sheet": sheet, "faults": faults})
                except Exception:
                    continue  # skip unexpected errors silently

                # Update progress bar
                progress_bar.progress(idx / total, text=f"Processed {idx}/{total} sheets")

            progress_bar.empty()  # remove progress bar once done

            if not diagnosis_summary:
                st.success("✅ No issues detected in any sheet.")
            else:
                st.subheader("⚠️ Issues Detected")
                for entry in diagnosis_summary:
                    st.write(f"📑 **Sheet:** {entry['sheet']}")
                    st.dataframe(entry["faults"])

                pdf_buffer = generate_summary_pdf(diagnosis_summary)
                st.download_button(
                    label="📥 Download PDF Summary Report",
                    data=pdf_buffer,
                    file_name="summary_faults_report.pdf",
                    mime="application/pdf",
                )

    # ───────────────────────────── single-sheet mode ───────────────────────────
    else:
        sheet_name = st.selectbox("📑 Select asset sheet", xls.sheet_names)
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        prepared = prepare_sheet(df_raw)

        if not prepared:
            st.warning("⚠️ Sheet is missing required columns.")
        else:
            df_use, sr = prepared
            st.info(f"ℹ️ Estimated sample rate ≈ **{sr:.2f} Hz**")

            # Window selector (24h / 7d / all)
            duration = st.radio(
                "Select data window",
                ["Last 24 hours", "Last 7 days", "All data"],
                horizontal=True,
            )
            latest_time = df_use["t"].max()
            start_time = (
                latest_time - pd.Timedelta(days=1)
                if duration == "Last 24 hours"
                else latest_time - pd.Timedelta(days=7)
                if duration == "Last 7 days"
                else df_use["t"].min()
            )
            df_filtered = df_use[df_use["t"] >= start_time]

            # Optional per-sheet row limit (match behaviour of multi-sheet mode)
            max_rows = st.number_input(
                "🔢 Limit to most recent N rows (this sheet)",
                min_value=10,
                value=len(df_filtered),
                step=10,
            )
            df_filtered = df_filtered.sort_values("t").tail(max_rows)

            st.write(f"Data points selected: **{len(df_filtered)}**")

            if st.button("▶️ Run Diagnosis"):
                faults = diagnose_faults(df_filtered.copy(), sr)

                st.subheader("📋 Diagnosis (last 50 rows with issues)")
                st.dataframe(faults.tail(50))

                # Generate per-sheet PDF
                def generate_pdf_one_sheet(faults_df: pd.DataFrame, sheet: str) -> BytesIO:
                    buf = BytesIO()
                    doc = SimpleDocTemplate(buf, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elements = []

                    elements.append(Paragraph(f"Motor Diagnosis Report – {sheet}", styles["Title"]))
                    elements.append(Spacer(1, 12))

                    data = [["Timestamp", "Diagnosis"]]
                    for _, r in faults_df.iterrows():
                        data.append([r["t"].strftime("%Y-%m-%d %H:%M:%S"), r["Diagnosis"]])

                    tbl = Table(data, repeatRows=1)
                    tbl.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ]
                        )
                    )
                    elements.append(tbl)
                    doc.build(elements)
                    buf.seek(0)
                    return buf

                pdf = generate_pdf_one_sheet(faults.tail(50), sheet_name)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf,
                    file_name=f"diagnosis_{sheet_name}.pdf",
                    mime="application/pdf",
                )
