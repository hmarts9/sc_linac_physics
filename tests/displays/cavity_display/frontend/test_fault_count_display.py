from datetime import datetime, timedelta
from random import randint
from unittest.mock import MagicMock

from PyQt5.QtCore import QDateTime
from pytestqt.qtbot import QtBot

from displays.cavity_display.backend.backend_cavity import BackendCavity
from displays.cavity_display.backend.fault import FaultCounter
from displays.cavity_display.frontend.fault_count_display import FaultCountDisplay
from utils.sc_linac.linac import Machine

non_hl_iterator = Machine(cavity_class=BackendCavity).non_hl_iterator


def test_get_data_no_data(qtbot: QtBot):
    window = FaultCountDisplay()
    qtbot.addWidget(window)

    start = datetime.now().replace(microsecond=0) - timedelta(days=1)
    start_qt = QDateTime.fromSecsSinceEpoch(int(start.timestamp()))

    end = datetime.now().replace(microsecond=0)
    end_qt = QDateTime.fromSecsSinceEpoch(int(end.timestamp()))

    window.start_selector.setDateTime(start_qt)
    window.end_selector.setDateTime(end_qt)

    window.cavity = next(non_hl_iterator)
    window.cavity.get_fault_counts = MagicMock()
    window.get_data()
    window.cavity.get_fault_counts.assert_called_with(start, end)
    assert window.y_data == []
    assert window.num_of_faults == []
    assert window.num_of_invalids == []


def test_get_data_with_pot(qtbot: QtBot):
    window = FaultCountDisplay()
    qtbot.addWidget(window)

    q_dt = QDateTime.fromSecsSinceEpoch(int(datetime.now().timestamp()))
    window.start_selector.setDateTime(q_dt)
    window.end_selector.setDateTime(q_dt)

    window.hide_pot_checkbox.isChecked = MagicMock(return_value=False)

    window.cavity = next(non_hl_iterator)
    faults = randint(0, 100)
    invalids = randint(0, 100)
    result = {"POT": FaultCounter(fault_count=faults, invalid_count=invalids)}
    window.cavity.get_fault_counts = MagicMock(return_value=result)
    window.get_data()
    window.cavity.get_fault_counts.assert_called()
    window.hide_pot_checkbox.isChecked.assert_called()
    assert window.y_data == ["POT"]
    assert window.num_of_faults == [faults]
    assert window.num_of_invalids == [invalids]


def test_get_data_without_pot(qtbot: QtBot):
    window = FaultCountDisplay()
    qtbot.addWidget(window)

    q_dt = QDateTime.fromSecsSinceEpoch(int(datetime.now().timestamp()))
    window.start_selector.setDateTime(q_dt)
    window.end_selector.setDateTime(q_dt)

    window.hide_pot_checkbox.isChecked = MagicMock(return_value=True)

    window.cavity = next(non_hl_iterator)
    faults = randint(0, 100)
    invalids = randint(0, 100)
    result = {"POT": FaultCounter(fault_count=faults, invalid_count=invalids)}
    window.cavity.get_fault_counts = MagicMock(return_value=result)
    window.get_data()
    window.cavity.get_fault_counts.assert_called()
    window.hide_pot_checkbox.isChecked.assert_called()
    assert window.y_data == []
    assert window.num_of_faults == []
    assert window.num_of_invalids == []


def test_update_plot(qtbot: QtBot):
    # TODO test the actual plot contents

    window = FaultCountDisplay()
    qtbot.addWidget(window)

    window.cavity = next(non_hl_iterator)
    window.plot_window.clear = MagicMock()

    faults = randint(0, 100)
    invalids = randint(0, 100)
    result = {"POT": FaultCounter(fault_count=faults, invalid_count=invalids)}
    window.cavity.get_fault_counts = MagicMock(return_value=result)

    window.update_plot()
    window.plot_window.clear.assert_called()
    window.cavity.get_fault_counts.assert_called()
