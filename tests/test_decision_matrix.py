"""
The decision matrix (thesis §4.4.3, Table 1), checked cell by cell.

                RUL urgency
              immediate  urgent    soon      planned
   Anomaly    STOP       STOP      STOP      STOP
   Warning    URGENT     URGENT    MONITOR   MONITOR
   Healthy    PLAN       PLAN      CONTINUE  CONTINUE
"""

import pytest

from fusion import MaintenanceFusion

STOP = ("stop_and_inspect", "critical")
URGENT = ("schedule_urgent_maintenance", "high")
MONITOR = ("monitor_closely", "low")
PLAN = ("schedule_maintenance_soon", "medium")
CONTINUE = ("continue_operation", "normal")

URGENCIES = ("immediate", "urgent", "soon", "planned")
MATRIX = {
    "anomaly": (STOP, STOP, STOP, STOP),
    "warning": (URGENT, URGENT, MONITOR, MONITOR),
    "healthy": (PLAN, PLAN, CONTINUE, CONTINUE),
}

CELLS = [
    (status, urgency, expected)
    for status, row in MATRIX.items()
    for urgency, expected in zip(URGENCIES, row)
]


@pytest.mark.parametrize("status,urgency,expected", CELLS,
                         ids=[f"{s}-{u}" for s, u, _ in CELLS])
def test_matrix_cell(status, urgency, expected):
    decision = MaintenanceFusion().make_decision(
        {"status": status, "confidence": 0.9, "details": "test"},
        {"rul_cycles": 50, "urgency": urgency, "maintenance_window": "test"},
    )
    assert (decision["action"], decision["priority"]) == expected
