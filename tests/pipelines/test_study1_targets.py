from __future__ import annotations

import pandas as pd

from studies.pain_study.study1.targets import _required_run


def test_required_run_uses_clean_events_run_id() -> None:
    events = pd.DataFrame({"run_id": [1, 2]})

    run = _required_run(events)

    assert run.tolist() == [1, 2]
