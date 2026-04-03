# CLAUDE.md — Project Guidelines

## Project Overview

We are building a **FiftyOne custom plugin** for **multimodal anomaly detection**.

---

## Language & Runtime

- **Python 3.10+ only.** Use modern Python features (match statements, `X | Y` union types, `from __future__ import annotations` where needed).
- Do not write code compatible with Python < 3.10.

---

## Dependencies

| Purpose | Library |
|---|---|
| Vector mathematics | `numpy` |
| Video/multimodal embeddings | `twelvelabs` (official Python SDK) |
| Dataset management & visualization | `fiftyone` |
| Unit testing | `pytest` |

Do not introduce alternative libraries for these roles (e.g., no `scipy` for vector math, no other embedding SDKs).

---

## Code Style

- **Strict type hinting is required** on all function signatures — parameters and return types.
- **Modular function design**: each function should do one thing. Avoid large monolithic functions.
- Use `dataclasses` or `TypedDict` for structured data instead of raw dicts where practical.
- Follow PEP 8. Max line length: 100 characters.

Example of required style:

```python
import numpy as np
from numpy.typing import NDArray

def compute_cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
```

---

## Testing

- Use `pytest` for all unit tests.
- Test files live in `tests/` and follow the naming convention `test_<module>.py`.
- Each public function must have at least one test.
- Use `pytest.mark.parametrize` for data-driven tests.

---

## Sample Data Scripts

- When writing scripts to download or generate sample data, **strictly limit to 5–10 short video clips**.
- Short means under 30 seconds per clip where possible.
- This constraint exists to avoid long download times during development and CI.
- Never write a script that downloads an unbounded or large dataset without an explicit cap enforced in code.

Example:

```python
MAX_SAMPLE_CLIPS = 10  # Hard cap — do not remove

def download_samples(urls: list[str]) -> None:
    for url in urls[:MAX_SAMPLE_CLIPS]:
        ...
```

---

## FiftyOne Plugin Conventions

- Plugin entry point lives in `__init__.py` at the plugin root.
- Operators must be registered via `fiftyone.operators`.
- Keep plugin logic (operators) separate from core logic (embedding, anomaly detection).
- Plugin code should never perform heavy computation directly — delegate to standalone modules.
