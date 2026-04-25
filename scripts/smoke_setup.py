"""Smoke test rapido para validar el setup del repo.

Uso:
    uv run python scripts/smoke_setup.py

Chequea en orden y aborta a la primera falla:
    1. Python >=3.12
    2. .env presente
    3. Variables de entorno requeridas en .env
    4. Archivos de datos accesibles
    5. Imports criticos del paquete
    6. Carga real de un sample de cada dataset
    7. Auth config valido
    8. Encoding bcrypt verificable

Pensado para correr en <30 segundos. Devuelve exit code 0 si todo OK,
1 si algo truena. No imprime nada superfluo: solo [OK] / [FAIL] por chequeo.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_python_version() -> bool:
    if sys.version_info[:2] != (3, 12):
        _fail(f"Se requiere Python 3.12, actual {sys.version.split()[0]}")
        return False
    _ok(f"Python 3.12.{sys.version_info[2]}")
    return True


def check_env_file() -> bool:
    env = ROOT / ".env"
    if not env.exists():
        _fail("Falta .env. Copia .env.example a .env y edita rutas.")
        return False
    _ok(".env presente")
    return True


def check_env_vars() -> bool:
    try:
        from src.io import paths_from_env

        paths = paths_from_env()
    except Exception as exc:
        _fail(f".env mal configurado: {exc}")
        return False
    _ok(f"INTERRUPCIONES_PATH = {paths.interrupciones}")
    _ok(f"MOREA_PARQUET_PATH  = {paths.morea_parquet}")
    _ok(f"MOREA_ESTACIONES_PATH = {paths.morea_estaciones}")
    return True


def check_datasets_exist() -> bool:
    from src.io import paths_from_env

    paths = paths_from_env()
    missing = []
    for label, p in (
        ("interrupciones", paths.interrupciones),
        ("morea_parquet", paths.morea_parquet),
        ("morea_estaciones", paths.morea_estaciones),
    ):
        if not p.exists():
            missing.append(f"{label} ({p})")
    if missing:
        _fail(f"Datasets ausentes: {missing}")
        return False
    _ok("Los 3 datasets oficiales existen y son accesibles")
    return True


def check_imports() -> bool:
    try:
        import polars  # noqa: F401
        import xgboost  # noqa: F401
        import lightgbm  # noqa: F401
        import streamlit  # noqa: F401
        import plotly  # noqa: F401
        import folium  # noqa: F401
        import bcrypt  # noqa: F401

        from src.modeling.anomalias import filter_imposibles  # noqa: F401
        from src.modeling.clasificacion import train_xgboost_grid  # noqa: F401
        from src.modeling.forecasting import forecast_xgb_lags  # noqa: F401
        from src.monitoring.thresholds import detect_violations  # noqa: F401
        from src.inspector import inspect_dataframe  # noqa: F401
        from src.reports.export import build_report_xlsx  # noqa: F401
    except Exception as exc:
        _fail(f"Import fallo: {exc}")
        return False
    _ok("Imports criticos resueltos (polars, xgboost, lightgbm, streamlit, plotly, folium, src.*)")
    return True


def check_data_loadable() -> bool:
    try:
        from src.io import load_interrupciones, load_morea, paths_from_env

        paths = paths_from_env()
        df_int = load_interrupciones(paths.interrupciones)
        df_morea, df_est = load_morea(paths.morea_parquet, paths.morea_estaciones)
    except Exception as exc:
        _fail(f"Carga real fallo: {exc}")
        return False
    _ok(f"Interrupciones: {df_int.height:,} filas x {df_int.width} cols")
    _ok(f"MOREA sensores: {df_morea.height:,} filas x {df_morea.width} cols")
    _ok(f"MOREA estaciones: {df_est.height:,} filas x {df_est.width} cols")
    return True


def check_auth_config() -> bool:
    cfg_path = ROOT / "config" / "users.yaml"
    if not cfg_path.exists():
        _fail(f"Falta {cfg_path}")
        return False
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text())
        users = list(cfg["credentials"]["usernames"].keys())
    except Exception as exc:
        _fail(f"users.yaml invalido: {exc}")
        return False
    _ok(f"users.yaml OK con {len(users)} usuarios: {users}")
    return True


def check_bcrypt_works() -> bool:
    try:
        import bcrypt
        import yaml

        cfg = yaml.safe_load((ROOT / "config" / "users.yaml").read_text())
        # Verificar que el hash de fer descifra contra "ssa11".
        hash_fer = cfg["credentials"]["usernames"]["fer"]["password"]
        if not bcrypt.checkpw(b"ssa11", hash_fer.encode()):
            _fail("Hash de fer no descifra contra 'ssa11'.")
            return False
    except Exception as exc:
        _fail(f"bcrypt fallo: {exc}")
        return False
    _ok("Hashes bcrypt validan correctamente (fer/ssa11)")
    return True


CHECKS = (
    ("python_version", check_python_version),
    ("env_file", check_env_file),
    ("env_vars", check_env_vars),
    ("datasets_exist", check_datasets_exist),
    ("imports", check_imports),
    ("data_loadable", check_data_loadable),
    ("auth_config", check_auth_config),
    ("bcrypt_works", check_bcrypt_works),
)


def main() -> int:
    failed: list[str] = []
    for name, fn in CHECKS:
        try:
            ok = fn()
        except Exception as exc:
            _fail(f"{name}: excepcion no esperada: {exc}")
            ok = False
        if not ok:
            failed.append(name)
    print("-" * 60)
    if failed:
        print(f"[FAIL] Smoke test fallo en: {', '.join(failed)}")
        print("       Resuelve esos checks antes de levantar la app.")
        return 1
    print("[OK] Setup correcto. Listo para correr la app.")
    print("     Siguiente: make app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
