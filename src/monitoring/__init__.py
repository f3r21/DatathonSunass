"""Monitoreo de parametros DIGESA y eventos climaticos."""
from src.monitoring.alerts import AlertEvent, Severity, build_alerts, summarize_alerts
from src.monitoring.climate_thresholds import (
    DEFAULT_CLIMATE_THRESHOLDS,
    ClimateThreshold,
    climate_to_threshold_config,
    detect_climate_events,
)
from src.monitoring.incidents import (
    IncidentRecord,
    IncidentStatus,
    IncidentStore,
    alert_to_incident_key,
)
from src.monitoring.thresholds import (
    DIGESA_CLORO,
    DIGESA_PH,
    DIGESA_TURBIEDAD,
    ThresholdConfig,
    detect_violations,
    stream_scan,
)

__all__ = [
    "AlertEvent",
    "ClimateThreshold",
    "DEFAULT_CLIMATE_THRESHOLDS",
    "DIGESA_CLORO",
    "DIGESA_PH",
    "DIGESA_TURBIEDAD",
    "IncidentRecord",
    "IncidentStatus",
    "IncidentStore",
    "Severity",
    "ThresholdConfig",
    "alert_to_incident_key",
    "build_alerts",
    "climate_to_threshold_config",
    "detect_climate_events",
    "detect_violations",
    "stream_scan",
    "summarize_alerts",
]
