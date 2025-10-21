from typing import Optional

import numpy as np

from applications.microphonics.gui.async_data_manager import (
    BASE_HARDWARE_SAMPLE_RATE,
)
from applications.microphonics.plots.base_plot import BasePlot


class TimeSeriesPlot(BasePlot):
    """Time series plots"""

    def __init__(self, parent=None):
        config = {
            "title": "Time Series",
            "x_label": ("Time", "s"),
            "y_label": ("Detuning", "Hz"),
            "x_range": (0, 1),
            "grid": True,
        }

        super().__init__(parent, plot_type="time_series", config=config)

        self._original_data = {}
        self._is_batch_updating = False
        self._initial_view_set = False

        self.plot_widget.setDownsampling(ds=True, auto=True, mode="subsample")
        self.plot_widget.setClipToView(True)

        vb = self.plot_widget.getViewBox()
        vb.setLimits(xMin=0)

    def _format_tooltip(self, plot_type, x, y):
        """Override base tooltip formatting"""
        return f"Time: {x:.3f} s\nDetuning: {y:.2f} Hz"

    def _smart_decimate(self, times, values, target_points=5000):
        if len(times) <= target_points:
            return times, values

        stride = len(times) // target_points
        indices = np.arange(0, len(times), stride)

        if indices[-1] != len(times) - 1:
            indices = np.append(indices, len(times) - 1)

        min_idx = np.argmin(values)
        max_idx = np.argmax(values)
        indices = np.unique(np.append(indices, [min_idx, max_idx]))
        indices.sort()

        return times[indices], values[indices]

    def _calculate_time_axis(
        self, num_points: int, decimation: int
    ) -> Optional[np.ndarray]:
        if not isinstance(decimation, (int, float)) or decimation <= 0:
            print(
                f"WARN (TimeSeriesPlot): Invalid decimation '{decimation}'. Using 1."
            )
            decimation = 1
        try:
            effective_sample_rate = BASE_HARDWARE_SAMPLE_RATE / decimation
            if effective_sample_rate <= 0:
                raise ValueError(
                    f"Non-positive effective sample rate: {effective_sample_rate}"
                )
            return np.linspace(
                0, (num_points - 1) / effective_sample_rate, num_points
            )
        except (ValueError, ZeroDivisionError) as e:
            print(f"ERROR (TimeSeriesPlot): Could not calculate time axis: {e}")
            return None

    def _create_or_update_curve(
        self, cavity_num: int, times: np.ndarray, values: np.ndarray
    ):
        pen = self._get_cavity_pen(cavity_num)

        display_times, display_values = self._smart_decimate(
            times, values, target_points=20000
        )

        if cavity_num not in self.plot_curves:
            curve = self.plot_widget.plot(
                display_times,
                display_values,
                pen=pen,
                name=f"Cavity {cavity_num}",
                clipToView=True,
                skipFiniteCheck=True,
                antialias=False,
            )
            self.plot_curves[cavity_num] = curve
        else:
            self.plot_curves[cavity_num].setData(
                display_times, display_values, skipFiniteCheck=True
            )

    def _adjust_view(self):
        if not self._original_data:
            self.plot_widget.setXRange(0, 1, padding=0)
            return

        self.plot_widget.getViewBox().autoRange()

    def update_plot(self, cavity_num, cavity_channel_data):
        df_data, is_valid = self._preprocess_data(
            cavity_channel_data, channel_type="DF"
        )
        if not is_valid or df_data.size == 0:
            if cavity_num in self.plot_curves:
                self.plot_curves[cavity_num].setData([], [])
            return

        decimation = cavity_channel_data.get("decimation", 1)
        times = self._calculate_time_axis(len(df_data), decimation)

        if times is None:
            if cavity_num in self.plot_curves:
                self.plot_curves[cavity_num].setData([], [])
            return

        self._original_data[cavity_num] = (times, df_data)
        self._create_or_update_curve(cavity_num, times, df_data)

    def begin_batch_update(self):
        self._is_batch_updating = True

    def end_batch_update(self):
        self._is_batch_updating = False
        if not self._initial_view_set and self._original_data:
            self._adjust_view()
            self._initial_view_set = True

    def clear_plot(self):
        """Override clear_plot to also clear time series specific data structures"""
        super().clear_plot()
        self._original_data = {}
        self._initial_view_set = False
