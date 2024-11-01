from typing import Dict, Optional

import pyqtgraph as pg
from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QDateTimeEdit,
    QLabel,
    QCheckBox,
)
from pydm import Display

from displays.cavity_display.backend.backend_cavity import BackendCavity
from displays.cavity_display.backend.fault import FaultCounter
from displays.cavity_display.frontend.cavity_widget import (
    DARK_GRAY_COLOR,
    RED_FILL_COLOR,
    PURPLE_FILL_COLOR,
)
from utils.sc_linac.linac import Machine
from utils.sc_linac.linac_utils import ALL_CRYOMODULES

DISPLAY_MACHINE = Machine(cavity_class=BackendCavity)


class FaultCountDisplay(Display):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fault Count Display")

        main_v_layout = QVBoxLayout()
        input_h_layout = QHBoxLayout()

        self.plot_window = pg.plot()
        self.plot_window.setBackground(DARK_GRAY_COLOR)

        main_v_layout.addLayout(input_h_layout)
        main_v_layout.addWidget(self.plot_window)
        self.setLayout(main_v_layout)

        self.cm_combo_box = QComboBox()
        self.cav_combo_box = QComboBox()

        end_date_time = QDateTime.currentDateTime()
        intermediate_time = QDateTime.addSecs(end_date_time, -30 * 60)  # 30 min
        min_date_time = QDateTime.addYears(end_date_time, -3)  # 3 years

        self.start_selector = QDateTimeEdit()
        self.start_selector.setCalendarPopup(True)

        self.end_selector = QDateTimeEdit()
        self.end_selector.setCalendarPopup(True)

        self.start_selector.setMinimumDateTime(min_date_time)
        self.start_selector.setDateTime(intermediate_time)
        self.end_selector.setDateTime(end_date_time)
        self.start_selector.dateChanged.connect(self.update_plot)
        self.start_selector.editingFinished.connect(self.update_plot)
        self.end_selector.dateChanged.connect(self.update_plot)
        self.end_selector.editingFinished.connect(self.update_plot)

        self.hide_pot_checkbox = QCheckBox(text="Hide POT faults")
        self.hide_pot_checkbox.stateChanged.connect(self.update_plot)

        input_h_layout.addWidget(QLabel("Cryomodule:"))
        input_h_layout.addWidget(self.cm_combo_box)
        input_h_layout.addWidget(QLabel("Cavity:"))
        input_h_layout.addWidget(self.cav_combo_box)
        input_h_layout.addStretch()
        input_h_layout.addWidget(QLabel("Start:"))
        input_h_layout.addWidget(self.start_selector)
        input_h_layout.addWidget(QLabel("End:"))
        input_h_layout.addWidget(self.end_selector)
        main_v_layout.addWidget(self.hide_pot_checkbox)

        self.cm_combo_box.addItems([""] + ALL_CRYOMODULES)
        self.cav_combo_box.addItems([""] + [str(i) for i in range(1, 9)])

        self.num_of_faults = []
        self.num_of_invalids = []
        self.y_data = None
        self.data: Dict[str, FaultCounter] = None

        self.cavity: Optional[BackendCavity] = None
        self.cm_combo_box.currentIndexChanged.connect(self.update_cavity)
        self.cav_combo_box.currentIndexChanged.connect(self.update_cavity)

    def update_cavity(self):
        cm_name = self.cm_combo_box.currentText()
        cav_num = self.cav_combo_box.currentText()

        if not cm_name or not cav_num:
            return

        self.cavity: BackendCavity = DISPLAY_MACHINE.cryomodules[cm_name].cavities[
            int(cav_num)
        ]
        self.update_plot()

    def get_data(self):
        self.num_of_faults = []
        self.num_of_invalids = []
        self.y_data = []

        start = self.start_selector.dateTime().toPyDateTime()
        end = self.end_selector.dateTime().toPyDateTime()

        """
        result is a dictionary with:
            key = fault TLC string i.e. "BCS"
            value = FaultCounter(fault_count=0, ok_count=1, invalid_count=0) <-- Example
        """
        data: Dict[str, FaultCounter] = self.cavity.get_fault_counts(start, end)

        if self.hide_pot_checkbox.isChecked():
            data.pop("POT")

        for tlc, counter_obj in data.items():
            self.y_data.append(tlc)
            self.num_of_faults.append(counter_obj.fault_count)
            self.num_of_invalids.append(counter_obj.invalid_count)

    def update_plot(self):
        if not self.cavity:
            return
        self.plot_window.clear()
        self.get_data()

        ticks = []
        y_vals_ints = []

        for idy, y_val in enumerate(self.y_data):
            ticks.append((idy, y_val))
            y_vals_ints.append(idy)

        # Create pyqt5graph bar graph for faults, then stack invalid faults on same bars
        fault_bars = pg.BarGraphItem(
            x0=0,
            y=y_vals_ints,
            height=0.6,
            width=self.num_of_faults,
            brush=RED_FILL_COLOR,
        )

        invalid_bars = pg.BarGraphItem(
            x0=self.num_of_faults,
            y=y_vals_ints,
            height=0.6,
            width=self.num_of_invalids,
            brush=PURPLE_FILL_COLOR,
        )

        tlc_axis = self.plot_window.getAxis("left")
        tlc_axis.setTicks([ticks])
        self.plot_window.showGrid(x=True, y=False, alpha=0.6)
        self.plot_window.addItem(fault_bars)
        self.plot_window.addItem(invalid_bars)
