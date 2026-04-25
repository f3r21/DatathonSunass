"""Login wall basado en streamlit-authenticator + logout en sidebar.

Cada pagina debe llamar `require_auth()` antes de cualquier render. El logout
en el sidebar se renderiza automaticamente despues del login. Si las
credenciales no estan configuradas, se permite acceso anonimo con warning
(modo demo).

Notas de implementacion:
    - El Authenticate de streamlit-authenticator 0.4.x usa internamente un
      `extra_streamlit_components.CookieManager` que llama componentes Streamlit
      en su constructor; por eso NO se puede envolver en @st.cache_resource
      (Streamlit detecta el "widget en cached function" y aborta).
    - Inicializamos manualmente las claves del session_state que la libreria
      consume en su primera lectura (`logout`, `authentication_status`,
      `name`, `username`) para evitar KeyError al primer render.
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "users.yaml"

# Claves que streamlit-authenticator espera ver en session_state desde el primer
# render. Si no estan, su CookieController truena con KeyError("logout").
_REQUIRED_KEYS: dict[str, object] = {
    "logout": False,
    "authentication_status": None,
    "name": None,
    "username": None,
    "init_login": None,
    "failed_login_attempts": {},
}


def _ensure_session_state() -> None:
    for key, default in _REQUIRED_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _load_config() -> dict | None:
    if not _CONFIG_PATH.exists():
        return None
    try:
        return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("No se pudo leer %s: %s", _CONFIG_PATH, exc)
        return None


def _get_authenticator():
    """Devuelve el Authenticate vivo. NO usar @st.cache_resource: usa widgets."""
    if "_authenticator" in st.session_state:
        return st.session_state["_authenticator"]
    try:
        import streamlit_authenticator as stauth
    except ImportError:
        logger.warning("streamlit-authenticator no esta instalado.")
        return None
    cfg = _load_config()
    if cfg is None:
        return None
    auth = stauth.Authenticate(
        cfg["credentials"],
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
    )
    st.session_state["_authenticator"] = auth
    return auth


def require_auth(page_title: str = "SUNASS") -> str | None:
    """Bloquea la pagina hasta que el usuario haga login.

    Returns:
        username si autenticado, None si modo demo (sin config).
    """
    _ensure_session_state()
    authenticator = _get_authenticator()

    if authenticator is None:
        st.sidebar.warning(
            "Modo demo: auth no configurado. Instala streamlit-authenticator "
            "y crea config/users.yaml para habilitar login."
        )
        return None

    # API 0.4.x: login() acepta location y fields. Sin keyword si la firma es vieja.
    try:
        authenticator.login(
            location="main",
            fields={"Form name": f"Acceso · {page_title}"},
        )
    except TypeError:
        try:
            authenticator.login("Acceso", "main")
        except Exception as exc:
            logger.error("login() fallo: %s", exc)
            st.error(f"Auth roto: {exc}")
            st.stop()

    auth_status = st.session_state.get("authentication_status")
    if auth_status is False:
        st.error("Usuario o contrasena incorrectos.")
        st.stop()
    if auth_status is None:
        st.info(
            "Ingresa tus credenciales. Demo: usuario `fer` o `jurado`, "
            "contrasena segun config/users.yaml."
        )
        st.stop()

    name = st.session_state.get("name", "")
    username = st.session_state.get("username", "")
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.6rem 0.4rem; border-radius: 6px;
                        background: #eaf4fc; margin-bottom: 0.6rem;">
                <div style="font-size: 0.78rem; color: #6c757d;">Sesion activa</div>
                <div style="font-weight: 600; color: #0a4d8c;">{name}</div>
                <div style="font-size: 0.75rem; color: #6c757d;">@{username}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            authenticator.logout(button_name="Cerrar sesion", location="sidebar")
        except TypeError:
            authenticator.logout("Cerrar sesion", "sidebar")
    return username
