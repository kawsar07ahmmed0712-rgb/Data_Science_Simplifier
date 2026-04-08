from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


def _normalize_single_header(header: object, fallback_index: int) -> str:
    value = "" if header is None else str(header).strip().lower()
    value = re.sub(r"[^\w]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    if not value:
        return f"column_{fallback_index}"

    if value[0].isdigit():
        value = f"col_{value}"

    return value


def normalize_headers(headers: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    counts: Counter[str] = Counter()

    for idx, header in enumerate(headers, start=1):
        base = _normalize_single_header(header=header, fallback_index=idx)
        counts[base] += 1

        if counts[base] == 1:
            normalized.append(base)
        else:
            normalized.append(f"{base}_{counts[base]}")

    return normalized