"""Exportacion de reportes XLSX y PDF para la app Streamlit.

Cubre el resultado oficial 2 (Automatizacion de reportes) de la Categoria I.
Las funciones devuelven `bytes` listos para `st.download_button`.

Diseño:
    - `ReportContext` es el contrato de entrada. Lo llena la pagina de Reportes
      a partir de los mismos caches que usan las otras paginas.
    - XLSX: multiples hojas con formato minimo (header colored, auto-width),
      hecho con `xlsxwriter` (ya es dep).
    - PDF: layout A4 simple con `fitz` (PyMuPDF, ya es dep). Sin dependencias
      extras a reportlab.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportContext:
    """Payload declarativo para armar reportes institucionales."""

    team_code: str
    categoria: str
    title: str
    subtitle: str
    generated_at: datetime
    kpis: list[tuple[str, str]]  # [(label, value), ...]
    alerts_table: pl.DataFrame
    forecast_table: pl.DataFrame
    chronic_stations: pl.DataFrame
    model_metrics: dict[str, object] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)


def _strip_timezones(pdf):
    """Excel no soporta tz-aware datetimes. Los convertimos a tz-naive."""
    import pandas as pd

    for col in pdf.columns:
        s = pdf[col]
        if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
            pdf[col] = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return pdf


def _df_to_sheet(writer, df: pl.DataFrame, sheet_name: str, header_format) -> None:
    """Escribe un polars.DataFrame a una hoja con formato minimo."""
    if df.is_empty():
        ws = writer.book.add_worksheet(sheet_name)
        ws.write(0, 0, "(sin datos)")
        return
    pdf = _strip_timezones(df.to_pandas())
    pdf.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
    ws = writer.sheets[sheet_name]
    for col_idx, col_name in enumerate(pdf.columns):
        ws.write(0, col_idx, str(col_name), header_format)
        max_len = max(len(str(col_name)), pdf[col_name].astype(str).map(len).max() if len(pdf) else 0)
        ws.set_column(col_idx, col_idx, min(max(max_len + 2, 10), 40))


def build_report_xlsx(ctx: ReportContext) -> bytes:
    """Construye un XLSX multi-hoja y devuelve los bytes."""
    import xlsxwriter  # noqa: F401 — import validado

    buffer = io.BytesIO()
    with pl.Config(tbl_formatting="NOTHING"):
        # Usamos pandas.ExcelWriter engine=xlsxwriter.
        import pandas as pd

        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            workbook = writer.book
            header_fmt = workbook.add_format(
                {
                    "bold": True,
                    "bg_color": "#0d3b66",
                    "font_color": "white",
                    "border": 1,
                    "align": "left",
                    "valign": "vcenter",
                }
            )
            title_fmt = workbook.add_format(
                {"bold": True, "font_size": 16, "font_color": "#0d3b66"}
            )
            meta_fmt = workbook.add_format({"italic": True, "font_color": "#6c757d"})
            kpi_label_fmt = workbook.add_format(
                {"bold": True, "bg_color": "#f4f7fb", "border": 1}
            )
            kpi_value_fmt = workbook.add_format({"border": 1, "align": "right"})

            # Hoja Resumen.
            ws = workbook.add_worksheet("Resumen")
            ws.set_column(0, 0, 38)
            ws.set_column(1, 1, 28)
            ws.write("A1", ctx.title, title_fmt)
            ws.write("A2", ctx.subtitle, meta_fmt)
            ws.write(
                "A3",
                f"Equipo {ctx.team_code} · Categoria {ctx.categoria} · "
                f"Generado {ctx.generated_at.strftime('%Y-%m-%d %H:%M')}",
                meta_fmt,
            )
            row = 5
            ws.write(row, 0, "Indicador clave", header_fmt)
            ws.write(row, 1, "Valor", header_fmt)
            row += 1
            for label, value in ctx.kpis:
                ws.write(row, 0, label, kpi_label_fmt)
                ws.write(row, 1, value, kpi_value_fmt)
                row += 1
            row += 2
            if ctx.insights:
                ws.write(row, 0, "Insights clave", header_fmt)
                row += 1
                for insight in ctx.insights:
                    ws.write(row, 0, f"• {insight}")
                    row += 1

            # Hoja Alertas.
            _df_to_sheet(writer, ctx.alerts_table, "Alertas activas", header_fmt)
            # Hoja Forecasting.
            _df_to_sheet(writer, ctx.forecast_table, "Forecasting", header_fmt)
            # Hoja Estaciones criticas.
            _df_to_sheet(
                writer, ctx.chronic_stations, "Estaciones criticas", header_fmt
            )

            # Hoja Modelo.
            ws = workbook.add_worksheet("Modelo XGBoost")
            ws.set_column(0, 0, 40)
            ws.set_column(1, 1, 32)
            ws.write(0, 0, "Metrica", header_fmt)
            ws.write(0, 1, "Valor", header_fmt)
            for i, (k, v) in enumerate(ctx.model_metrics.items(), start=1):
                ws.write(i, 0, str(k))
                ws.write(i, 1, str(v))

    data = buffer.getvalue()
    logger.info("build_report_xlsx: %d bytes, hojas=%d", len(data), 5)
    return data


def build_report_pdf(ctx: ReportContext) -> bytes:
    """Construye un PDF A4 simple con los bloques principales."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 portrait
    margin = 40
    x = margin
    y = margin

    def draw_text(text: str, size: int = 11, bold: bool = False, color=(0, 0, 0)) -> float:
        """Dibuja un bloque de texto y devuelve la nueva Y."""
        nonlocal y, page
        fontname = "helv-b" if bold else "helv"
        rect = fitz.Rect(x, y, 595 - margin, y + size + 4)
        tw = fitz.TextWriter(page.rect, color=color)
        tw.fill_textbox(
            rect,
            text,
            fontsize=size,
            font=fitz.Font(fontname=fontname),
            align=fitz.TEXT_ALIGN_LEFT,
        )
        tw.write_text(page)
        return y + size + 6

    def hr() -> None:
        nonlocal y, page
        page.draw_line((x, y), (595 - margin, y), color=(0.85, 0.9, 0.95), width=0.8)
        y += 8

    def next_page_if_needed(min_space: float = 80.0) -> None:
        nonlocal y, page
        if y + min_space > 842 - margin:
            page = doc.new_page(width=595, height=842)
            y = margin

    y = draw_text(ctx.title, size=18, bold=True, color=(0.05, 0.23, 0.4))
    y = draw_text(ctx.subtitle, size=11, color=(0.4, 0.4, 0.4))
    y = draw_text(
        f"Equipo {ctx.team_code} · Categoria {ctx.categoria} · "
        f"Generado {ctx.generated_at.strftime('%Y-%m-%d %H:%M')}",
        size=9,
        color=(0.5, 0.5, 0.5),
    )
    y += 6
    hr()

    y = draw_text("Indicadores clave", size=13, bold=True, color=(0.05, 0.23, 0.4))
    for label, value in ctx.kpis:
        next_page_if_needed()
        y = draw_text(f"   {label}:  {value}", size=10)
    y += 6
    hr()

    if ctx.insights:
        y = draw_text("Hallazgos", size=13, bold=True, color=(0.05, 0.23, 0.4))
        for bullet in ctx.insights:
            next_page_if_needed()
            y = draw_text(f"   • {bullet}", size=10)
        y += 6
        hr()

    def table_block(title: str, df: pl.DataFrame, max_rows: int = 12) -> None:
        nonlocal y
        next_page_if_needed(120)
        y = draw_text(title, size=13, bold=True, color=(0.05, 0.23, 0.4))
        if df.is_empty():
            y = draw_text("   (sin datos)", size=10, color=(0.5, 0.5, 0.5))
            return
        pdf = df.head(max_rows).to_pandas().astype(str)
        col_widths = [
            max(pdf[c].str.len().max() if len(pdf) else 0, len(c)) for c in pdf.columns
        ]
        header_line = "   " + "  ".join(
            c.ljust(col_widths[i])[: col_widths[i]] for i, c in enumerate(pdf.columns)
        )
        y = draw_text(header_line, size=8, bold=True)
        for _, row in pdf.iterrows():
            next_page_if_needed(20)
            line = "   " + "  ".join(
                str(row.iloc[i])[: col_widths[i]].ljust(col_widths[i])
                for i in range(len(pdf.columns))
            )
            y = draw_text(line, size=8)

    table_block("Alertas activas (top 12)", ctx.alerts_table)
    y += 6
    hr()
    table_block("Forecasting — comparativa de modelos", ctx.forecast_table)
    y += 6
    hr()
    table_block("Top estaciones cronicas (cloro)", ctx.chronic_stations)

    data = doc.tobytes()
    doc.close()
    logger.info("build_report_pdf: %d bytes", len(data))
    return data
