from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass(slots=True)
class RunContext:
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_name: str = ""
    user_target_column: str | None = None
    notes: dict[str, object] = field(default_factory=dict)