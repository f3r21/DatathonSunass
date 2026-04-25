"""Smoke test para el pipeline de feature engineering SENAMHI.

Crea un dataset dummy con dos estaciones y huecos temporales deliberados,
luego verifica shapes, ausencia de gaps, valores de lags y ausencia de leakage.

Ejecutar desde la raíz del proyecto:
    python scripts/smoke_senamhi_features.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Asegurar que src/ sea importable desde cualquier directorio
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.modeling.features import (
    add_climate_interactions,
    add_climate_lags,
    add_climate_rollings,
    build_senamhi_features,
    clean_senamhi_missing,
    ensure_daily_frequency,
)

def make_dummy() -> pl.DataFrame:
    """Crea un DataFrame SENAMHI de prueba con huecos intencionales."""
    rows_a = [
        # (fecha,        estacion,   precip, tmax,  tmin)
        (date(2024, 1,  1), "AREQUIPA",  0.0,  25.0, 10.0),
        (date(2024, 1,  2), "AREQUIPA",  5.0,  24.0,  9.0),
        # gap: 2024-01-03 falta
        (date(2024, 1,  4), "AREQUIPA",  0.0,  26.0, 11.0),
        (date(2024, 1,  5), "AREQUIPA",  0.0,  27.0, 12.0),
        (date(2024, 1,  6), "AREQUIPA",  3.0,  23.0,  8.0),
        (date(2024, 1,  7), "AREQUIPA",  0.0,  25.0, 10.0),
        (date(2024, 1,  8), "AREQUIPA",  0.0,  26.0, 11.0),
        # gap: 2024-01-09, 2024-01-10 faltan
        (date(2024, 1, 11), "AREQUIPA",  0.0,  24.0,  9.0),
        (date(2024, 1, 12), "AREQUIPA",  8.0,  22.0,  7.0),
    ]
    rows_b = [
        (date(2024, 1,  1), "CUSCO",     2.0,  20.0,  5.0),
        (date(2024, 1,  2), "CUSCO",     0.0,  21.0,  6.0),
        (date(2024, 1,  3), "CUSCO",     0.0,  22.0,  7.0),
        # gap: 2024-01-04 falta
        (date(2024, 1,  5), "CUSCO",     0.0,  23.0,  8.0),
        (date(2024, 1,  6), "CUSCO",     4.0,  19.0,  4.0),
    ]
    schema = {
        "fecha":      pl.Date,
        "estacion":   pl.Utf8,
        "precip_acum": pl.Float64,
        "tmax":       pl.Float64,
        "tmin":       pl.Float64,
    }
    return pl.from_records(
        rows_a + rows_b,
        schema=schema,
    )


def assert_no_date_gaps(df: pl.DataFrame, date_col: str = "fecha", group_col: str = "estacion") -> None:
    """Verifica que no haya huecos en fechas por estación."""
    for est, part in df.partition_by(group_col, maintain_order=True, as_dict=True).items():
        dates = part.sort(date_col).get_column(date_col).to_list()
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            assert delta == 1, (
                f"Hueco en {est}: entre {dates[i-1]} y {dates[i]} hay {delta} días"
            )


def assert_lag_correct(
    df: pl.DataFrame,
    col: str,
    lag: int,
    date_col: str = "fecha",
    group_col: str = "estacion",
) -> None:
    """Verifica que {col}_lag{lag} corresponde exactamente al valor N días atrás."""
    lag_col = f"{col}_lag{lag}"
    assert lag_col in df.columns, f"Columna {lag_col} ausente"
    for _, part in df.partition_by(group_col, maintain_order=True, as_dict=True).items():
        part = part.sort(date_col)
        dates = part.get_column(date_col).to_list()
        vals  = part.get_column(col).to_list()
        lags  = part.get_column(lag_col).to_list()
        for i, (d, lag_v) in enumerate(zip(dates, lags)):
            target_date = d - __import__("datetime").timedelta(days=lag)
            if target_date in dates:
                expected = vals[dates.index(target_date)]
                assert lag_v == expected, (
                    f"{lag_col} en {d}: esperado {expected}, obtenido {lag_v}"
                )


def assert_rolling_no_leakage(
    df: pl.DataFrame,
    col: str,
    window: int,
    date_col: str = "fecha",
    group_col: str = "estacion",
) -> None:
    """Verifica que {col}_roll{w}_mean no incluye el valor del día actual."""
    roll_col = f"{col}_roll{window}_mean"
    assert roll_col in df.columns, f"Columna {roll_col} ausente"
    for _, part in df.partition_by(group_col, maintain_order=True, as_dict=True).items():
        part = part.sort(date_col)
        first = part.head(1)
        assert first[roll_col][0] is None, f"{roll_col}: primera fila debe ser null (sin historial)"


def test_clean_senamhi_missing() -> None:
    df = pl.DataFrame({
        "precip_acum": [-99.9, 5.0, -99.9, 0.0],
        "tmax":        [25.0, -99.9, 22.0, 23.0],
    })
    result = clean_senamhi_missing(df)
    assert result.height == 4, "height no debe cambiar"
    assert result["precip_acum"].null_count() == 2
    assert result["tmax"].null_count() == 1
    print("  [OK] clean_senamhi_missing")


def test_ensure_daily_frequency() -> None:
    raw = make_dummy()
    original_height = raw.height

    result = ensure_daily_frequency(raw)
    assert result.height > original_height, "Deben insertarse fechas faltantes"
    # AREQUIPA: 01..12 = 12 días; original tenía 9 → 3 insertadas
    # CUSCO:    01..06 = 6 días; original tenía 5 → 1 insertada
    assert result.height == original_height + 4, (
        f"Se esperaban 4 filas insertadas, resultado: {result.height - original_height}"
    )

    result_sorted = result.sort(["estacion", "fecha"])
    assert_no_date_gaps(result_sorted)

    orig_sorted = raw.sort(["estacion", "fecha"])
    joined = orig_sorted.join(
        result_sorted,
        on=["fecha", "estacion"],
        how="left",
        suffix="_new",
    )
    for col in ("precip_acum", "tmax", "tmin"):
        assert (joined[col] == joined[f"{col}_new"]).fill_null(True).all(), (
            f"Valores de {col} modificados tras ensure_daily_frequency"
        )

    print(f"  [OK] ensure_daily_frequency: {original_height} → {result.height} filas")


def test_add_climate_lags() -> None:
    raw = make_dummy()
    df = (
        ensure_daily_frequency(raw)
        .sort(["estacion", "fecha"])
    )
    original_height = df.height

    result = add_climate_lags(df, lags=(1, 3, 7))
    assert result.height == original_height, "add_climate_lags no debe cambiar height"

    expected_lag_cols = [
        f"{col}_lag{lag}"
        for col in ("precip_acum", "tmax", "tmin")
        for lag in (1, 3, 7)
    ]
    for c in expected_lag_cols:
        assert c in result.columns, f"Columna {c} ausente"

    assert_lag_correct(result, "tmax", lag=1)
    assert_lag_correct(result, "precip_acum", lag=1)

    arequipa = result.filter(pl.col("estacion") == "AREQUIPA").sort("fecha")
    assert arequipa["tmax_lag1"][0] is None, "lag1 en primera fila debe ser null"
    assert arequipa["tmax_lag3"][0] is None, "lag3 en primera fila debe ser null"

    print(f"  [OK] add_climate_lags: {result.width - df.width} columnas nuevas")


def test_add_climate_rollings() -> None:
    raw = make_dummy()
    df = (
        ensure_daily_frequency(raw)
        .sort(["estacion", "fecha"])
    )
    original_height = df.height

    result = add_climate_rollings(df, windows=(3, 7))
    assert result.height == original_height, "add_climate_rollings no debe cambiar height"

    expected_roll_cols = [
        f"{col}_roll{w}_{stat}"
        for col in ("precip_acum", "tmax", "tmin")
        for w in (3, 7)
        for stat in ("mean", "std", "max", "min")
    ]
    for c in expected_roll_cols:
        assert c in result.columns, f"Columna {c} ausente"

    assert_rolling_no_leakage(result, "precip_acum", window=3)

    # AREQUIPA, 2024-01-06, roll3_mean tmax: shift(1) → ventana [null, 26.0, 27.0], no incluye 23.0
    arequipa = result.filter(pl.col("estacion") == "AREQUIPA").sort("fecha")
    row_06 = arequipa.filter(pl.col("fecha") == date(2024, 1, 6))
    roll_mean = row_06["tmax_roll3_mean"][0]
    if roll_mean is not None:
        assert roll_mean != 23.0, "Rolling incluye el valor del día actual (leakage detectado)"

    print(f"  [OK] add_climate_rollings: {result.width - df.width} columnas nuevas")


def test_add_climate_interactions() -> None:
    raw = make_dummy()
    df = (
        ensure_daily_frequency(raw)
        .sort(["estacion", "fecha"])
    )
    original_height = df.height

    result = add_climate_interactions(df)
    assert result.height == original_height, "add_climate_interactions no debe cambiar height"
    assert "rango_termico" in result.columns
    assert "dias_consecutivos_sin_lluvia" in result.columns

    check = result.filter(
        pl.col("tmax").is_not_null() & pl.col("tmin").is_not_null()
    )
    diff = (check["tmax"] - check["tmin"] - check["rango_termico"]).abs().max()
    assert diff < 1e-9, f"rango_termico incorrecto, max diff = {diff}"

    consec = result["dias_consecutivos_sin_lluvia"]
    assert consec.min() >= 0, "dias_consecutivos_sin_lluvia con valores negativos"

    # AREQUIPA, 2024-01-01: primer día → 0 días sin lluvia previos
    arequipa = result.filter(pl.col("estacion") == "AREQUIPA").sort("fecha")
    assert arequipa["dias_consecutivos_sin_lluvia"][0] == 0

    print(f"  [OK] add_climate_interactions: {result.width - df.width} columnas nuevas")


def test_pipeline_full() -> None:
    """Test del pipeline completo con build_senamhi_features."""
    raw = make_dummy()
    original_height = raw.height

    result = build_senamhi_features(
        raw,
        lags=(1, 3, 7),
        windows=(3, 7),
    )

    assert result.height >= original_height, "Pipeline no debe perder filas"

    for col in ("precip_acum_lag1", "tmax_roll7_mean", "rango_termico",
                "dias_consecutivos_sin_lluvia"):
        assert col in result.columns, f"Columna esperada ausente: {col}"

    assert_no_date_gaps(result)

    print(
        f"  [OK] build_senamhi_features: {original_height} filas originales → "
        f"{result.height} filas, {result.width} columnas"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests con CSV real SENAMHI (datos/senahmi/estaciones_aqp.csv)
# ──────────────────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).resolve().parents[2] / "datos" / "senahmi" / "estaciones_aqp.csv"

_CSV_RENAME = {"station": "estacion", "precipitation": "precip_acum"}


def _load_real_csv() -> pl.DataFrame:
    """Carga y normaliza estaciones_aqp.csv al esquema esperado por las funciones."""
    df = pl.read_csv(CSV_PATH, infer_schema_length=5000)
    df = df.rename({k: v for k, v in _CSV_RENAME.items() if k in df.columns})
    df = df.with_columns(
        pl.date(pl.col("year"), pl.col("month"), pl.col("day")).alias("fecha")
    ).drop(["year", "month", "day"])
    return df


def test_real_csv_load() -> None:
    if not CSV_PATH.exists():
        print(f"  [SKIP] test_real_csv_load: no encontrado en {CSV_PATH}")
        return
    df = _load_real_csv()
    assert df.height > 0, "CSV vacío"
    for col in ("estacion", "fecha", "precip_acum", "tmax", "tmin"):
        assert col in df.columns, f"Columna '{col}' ausente tras renombrar"
    stations = sorted(df["estacion"].unique().to_list())
    assert len(stations) > 1, "Se esperan múltiples estaciones"
    print(
        f"  [OK] test_real_csv_load: {df.height:,} filas, "
        f"{len(stations)} estaciones: {stations}"
    )


def test_real_csv_clean_sentinel() -> None:
    if not CSV_PATH.exists():
        print("  [SKIP] test_real_csv_clean_sentinel: archivo no encontrado")
        return
    df = _load_real_csv()
    original_height = df.height

    sentinel_before = sum(
        int((df[c] == -99.9).sum())
        for c in ("precip_acum", "tmax", "tmin") if c in df.columns
    )
    assert sentinel_before > 0, "Se esperaban valores -99.9 en el CSV real"

    cleaned = clean_senamhi_missing(df)
    assert cleaned.height == original_height, "height no debe cambiar"

    sentinel_after = sum(
        int((cleaned[c] == -99.9).fill_null(False).sum())
        for c in ("precip_acum", "tmax", "tmin") if c in cleaned.columns
    )
    assert sentinel_after == 0, f"Aún quedan {sentinel_after} centinelas tras clean"

    print(
        f"  [OK] test_real_csv_clean_sentinel: {sentinel_before:,} centinelas → null "
        f"(precip={cleaned['precip_acum'].null_count():,}, "
        f"tmax={cleaned['tmax'].null_count():,}, "
        f"tmin={cleaned['tmin'].null_count():,})"
    )


def test_real_csv_ensure_daily_frequency() -> None:
    if not CSV_PATH.exists():
        print("  [SKIP] test_real_csv_ensure_daily_frequency: archivo no encontrado")
        return
    station = "andahua"
    sample = _load_real_csv().filter(pl.col("estacion") == station)
    assert sample.height > 0, f"Estación '{station}' no encontrada"
    original_height = sample.height

    result = ensure_daily_frequency(sample, group_cols=("estacion",))
    assert result.height >= original_height, "ensure_daily_frequency no debe perder filas"
    assert_no_date_gaps(result)

    print(
        f"  [OK] test_real_csv_ensure_daily_frequency ({station}): "
        f"{original_height:,} → {result.height:,} filas "
        f"({result.height - original_height} fechas insertadas)"
    )


def test_real_csv_lags_no_leakage() -> None:
    """Valores concretos: precip 1950-12-09=8.5 → lag1 del 10 y lag3 del 12 == 8.5."""
    if not CSV_PATH.exists():
        print("  [SKIP] test_real_csv_lags_no_leakage: archivo no encontrado")
        return
    station = "andahua"
    df = (
        _load_real_csv()
        .filter(pl.col("estacion") == station)
        .pipe(clean_senamhi_missing)
        .pipe(ensure_daily_frequency, group_cols=("estacion",))
        .sort(["estacion", "fecha"])
    )
    result = add_climate_lags(df, lags=(1, 3, 7), group_col="estacion")
    assert result.height == df.height, "add_climate_lags no debe cambiar height"

    first = result.sort("fecha").head(1)
    assert first["precip_acum_lag1"][0] is None, "lag1 primera fila debe ser null"
    assert first["tmax_lag7"][0] is None,         "lag7 primera fila debe ser null"

    # 1950-12-09 precipitation = 8.5  →  lag1 del 1950-12-10 debe ser 8.5
    row_10 = result.filter(pl.col("fecha") == date(1950, 12, 10))
    if row_10.height > 0:
        val = row_10["precip_acum_lag1"][0]
        assert val == 8.5, f"lag1 precip 1950-12-10: esperado 8.5, obtenido {val}"

    # lag3 del 1950-12-12 → 3 días atrás = 1950-12-09 = 8.5
    row_12 = result.filter(pl.col("fecha") == date(1950, 12, 12))
    if row_12.height > 0:
        val = row_12["precip_acum_lag3"][0]
        assert val == 8.5, f"lag3 precip 1950-12-12: esperado 8.5, obtenido {val}"

    print(
        f"  [OK] test_real_csv_lags_no_leakage ({station}): "
        f"lag1/lag3 verificados con valores concretos del CSV"
    )


def test_real_csv_rollings_shape() -> None:
    if not CSV_PATH.exists():
        print("  [SKIP] test_real_csv_rollings_shape: archivo no encontrado")
        return
    station = "chivay"  # tiene tmax/tmin reales (no todo -99.9)
    df = (
        _load_real_csv()
        .filter(pl.col("estacion") == station)
        .pipe(clean_senamhi_missing)
        .pipe(ensure_daily_frequency, group_cols=("estacion",))
        .sort(["estacion", "fecha"])
    )
    result = add_climate_rollings(df, windows=(3, 7), group_col="estacion")
    assert result.height == df.height, "add_climate_rollings no debe cambiar height"

    expected_cols = [
        f"{col}_roll{w}_{stat}"
        for col in ("precip_acum", "tmax", "tmin")
        for w in (3, 7)
        for stat in ("mean", "std", "max", "min")
    ]
    for c in expected_cols:
        assert c in result.columns, f"Columna esperada ausente: {c}"

    # Sin leakage: primera fila no puede tener rolling completo
    first = result.sort("fecha").head(1)
    assert first["tmax_roll3_mean"][0] is None, "roll3_mean primera fila debe ser null"

    print(
        f"  [OK] test_real_csv_rollings_shape ({station}): "
        f"{df.height:,} filas, {result.width} columnas totales"
    )


def test_real_csv_pipeline_full() -> None:
    if not CSV_PATH.exists():
        print("  [SKIP] test_real_csv_pipeline_full: archivo no encontrado")
        return
    station = "aplao"
    df_station = _load_real_csv().filter(pl.col("estacion") == station)
    original_height = df_station.height
    assert original_height > 0

    result = build_senamhi_features(
        df_station,
        date_col="fecha",
        group_col="estacion",
        lags=(1, 3, 7, 14),
        windows=(3, 7, 14),
    )

    assert result.height >= original_height, "Pipeline no debe perder filas"

    for col in (
        "precip_acum_lag1", "precip_acum_lag14",
        "tmax_roll7_mean", "tmin_roll14_min",
        "rango_termico", "dias_consecutivos_sin_lluvia",
    ):
        assert col in result.columns, f"Columna esperada ausente: {col}"

    assert_no_date_gaps(result)

    valid = result.filter(pl.col("tmax").is_not_null() & pl.col("tmin").is_not_null())
    if valid.height > 0:
        max_diff = (valid["tmax"] - valid["tmin"] - valid["rango_termico"]).abs().max()
        assert max_diff < 1e-6, f"rango_termico incorrecto: max diff = {max_diff}"

    assert result["dias_consecutivos_sin_lluvia"].min() >= 0

    print(
        f"  [OK] test_real_csv_pipeline_full ({station}): "
        f"{original_height:,} filas → {result.height:,} filas, {result.width} columnas"
    )


if __name__ == "__main__":
    print("=== Smoke test: SENAMHI feature engineering ===\n")
    tests = [
        # --- dummy dataset ---
        test_clean_senamhi_missing,
        test_ensure_daily_frequency,
        test_add_climate_lags,
        test_add_climate_rollings,
        test_add_climate_interactions,
        test_pipeline_full,
        # --- CSV real (estaciones_aqp.csv) ---
        test_real_csv_load,
        test_real_csv_clean_sentinel,
        test_real_csv_ensure_daily_frequency,
        test_real_csv_lags_no_leakage,
        test_real_csv_rollings_shape,
        test_real_csv_pipeline_full,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{'All tests passed.' if not failed else f'{failed} test(s) failed.'}")
    sys.exit(0 if not failed else 1)
