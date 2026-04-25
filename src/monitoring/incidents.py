"""Workflow de estado de incidencias sobre alertas detectadas.

Cubre el resultado oficial 7 (Atencion oportuna de incidencias). Cada alerta
puede transicionar entre estados y se persiste en un JSON local para no
sumar dependencia de base de datos durante la competencia.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


class IncidentStatus(str, Enum):
    NUEVO = "NUEVO"
    EN_REVISION = "EN_REVISION"
    RESUELTO = "RESUELTO"

    def next_allowed(self) -> tuple[IncidentStatus, ...]:
        if self == IncidentStatus.NUEVO:
            return (IncidentStatus.EN_REVISION, IncidentStatus.RESUELTO)
        if self == IncidentStatus.EN_REVISION:
            return (IncidentStatus.RESUELTO,)
        return ()


@dataclass
class IncidentRecord:
    """Registro mutable de una incidencia.

    `alert_key` identifica la alerta de negocio (estacion + parametro + start_ts
    en ISO). El mismo alert_key se reconcilia entre renders — si la pagina
    recalcula las alertas, el estado persistido se respeta.
    """

    alert_key: str
    station_id: str
    parameter: str
    severity: str
    start_ts: str
    end_ts: str
    peak_value: float
    duration_min: float
    status: IncidentStatus = IncidentStatus.NUEVO
    assignee: str | None = None
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def transition_to(self, new_status: IncidentStatus, assignee: str | None = None, note: str = "") -> None:
        if new_status not in self.status.next_allowed() and new_status != self.status:
            raise ValueError(
                f"Transicion invalida: {self.status.value} -> {new_status.value}. "
                f"Permitidas: {[s.value for s in self.status.next_allowed()]}"
            )
        self.status = new_status
        self.updated_at = datetime.utcnow().isoformat(timespec="seconds")
        if assignee is not None:
            self.assignee = assignee
        if note:
            separator = "\n" if self.notes else ""
            self.notes = f"{self.notes}{separator}[{self.updated_at}] {note}"

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> IncidentRecord:
        raw_status = data.get("status", IncidentStatus.NUEVO.value)
        status = IncidentStatus(raw_status) if isinstance(raw_status, str) else IncidentStatus.NUEVO
        return cls(
            alert_key=str(data["alert_key"]),
            station_id=str(data["station_id"]),
            parameter=str(data["parameter"]),
            severity=str(data["severity"]),
            start_ts=str(data["start_ts"]),
            end_ts=str(data["end_ts"]),
            peak_value=float(data.get("peak_value", 0.0)),
            duration_min=float(data.get("duration_min", 0.0)),
            status=status,
            assignee=data.get("assignee"),  # type: ignore[arg-type]
            notes=str(data.get("notes", "")),
            created_at=str(data.get("created_at", datetime.utcnow().isoformat(timespec="seconds"))),
            updated_at=str(data.get("updated_at", datetime.utcnow().isoformat(timespec="seconds"))),
            id=str(data.get("id", uuid4().hex[:12])),
        )


@dataclass(frozen=True)
class IncidentStore:
    """Persistencia JSON ligera — append-only, reconciliada por alert_key."""

    path: Path

    def load(self) -> dict[str, IncidentRecord]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {item["alert_key"]: IncidentRecord.from_dict(item) for item in raw}

    def save(self, records: dict[str, IncidentRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [rec.to_dict() for rec in records.values()]
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def upsert(self, record: IncidentRecord) -> IncidentRecord:
        existing = self.load()
        if record.alert_key in existing:
            prev = existing[record.alert_key]
            prev.severity = record.severity
            prev.peak_value = record.peak_value
            prev.duration_min = record.duration_min
            prev.end_ts = record.end_ts
            prev.updated_at = datetime.utcnow().isoformat(timespec="seconds")
            existing[record.alert_key] = prev
        else:
            existing[record.alert_key] = record
        self.save(existing)
        return existing[record.alert_key]


def alert_to_incident_key(station_id: str, parameter: str, start_ts: datetime | str) -> str:
    """Llave estable por alerta de negocio: station|param|startISO."""
    ts_iso = start_ts.isoformat() if isinstance(start_ts, datetime) else str(start_ts)
    return f"{station_id}|{parameter}|{ts_iso}"
