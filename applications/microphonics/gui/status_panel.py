"""Status panel for displaying cavity states and progress"""

from typing import Dict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel,
    QProgressBar, QGridLayout
)


class StatusPanel(QWidget):
    """Panel for displaying cavity status information"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.status_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Create status group
        group = QGroupBox("Cavity Status")
        grid_layout = QGridLayout()

        # Add headers
        headers = ["Cavity", "Status", "Progress", "Message"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold")
            grid_layout.addWidget(label, 0, col)

        # Create status widgets for each cavity
        for row, cavity_num in enumerate(range(1, 9), 1):
            # Cavity number
            grid_layout.addWidget(QLabel(f"Cavity {cavity_num}"), row, 0)

            # Status label
            status_label = QLabel("Not configured")
            grid_layout.addWidget(status_label, row, 1)

            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setMinimum(0)
            progress_bar.setMaximum(100)
            grid_layout.addWidget(progress_bar, row, 2)

            # Message label
            msg_label = QLabel("")
            grid_layout.addWidget(msg_label, row, 3)

            # Store references to widgets
            self.status_widgets[cavity_num] = {
                'status': status_label,
                'progress': progress_bar,
                'message': msg_label
            }

        group.setLayout(grid_layout)
        layout.addWidget(group)

    def update_cavity_status(self, cavity_num: int, status: str, progress: int, message: str):
        """Update status for a single cavity

        Args:
            cavity_num: Cavity number (1-8)
            status: Status string to display
            progress: Progress value (0-100)
            message: Message to display
        """
        if cavity_num in self.status_widgets:
            widgets = self.status_widgets[cavity_num]
            widgets['status'].setText(status)
            widgets['progress'].setValue(progress)
            widgets['message'].setText(message)

    def update_all_status(self, status_dict: Dict):
        """Update status for all cavities

        Args:
            status_dict: Dictionary mapping cavity numbers to status info
                Format: {
                    cavity_num: {
                        'status': str,
                        'progress': int,
                        'message': str
                    }
                }
        """
        for cavity_num, info in status_dict.items():
            self.update_cavity_status(
                cavity_num,
                info.get('status', ''),
                info.get('progress', 0),
                info.get('message', '')
            )

    def reset_all(self):
        """Reset all cavities to initial state"""
        for cavity_num in range(1, 9):
            self.update_cavity_status(
                cavity_num,
                "Not configured",
                0,
                ""
            )

    def set_cavity_error(self, cavity_num: int, error_msg: str):
        """Set error state for a cavity

        Args:
            cavity_num: Cavity number (1-8)
            error_msg: Error message to display
        """
        self.update_cavity_status(
            cavity_num,
            "Error",
            0,
            error_msg
        )
        # Optionally highlight the error state
        if cavity_num in self.status_widgets:
            self.status_widgets[cavity_num]['status'].setStyleSheet("color: red")

    def clear_cavity_error(self, cavity_num: int):
        """Clear error state for a cavity

        Args:
            cavity_num: Cavity number (1-8)
        """
        if cavity_num in self.status_widgets:
            self.status_widgets[cavity_num]['status'].setStyleSheet("")

    def get_cavity_status(self, cavity_num: int) -> Dict:
        """Get current status info for a cavity

        Args:
            cavity_num: Cavity number (1-8)

        Returns:
            Dictionary with status information or empty dict if cavity not found
        """
        if cavity_num in self.status_widgets:
            widgets = self.status_widgets[cavity_num]
            return {
                'status': widgets['status'].text(),
                'progress': widgets['progress'].value(),
                'message': widgets['message'].text()
            }
        return {}
