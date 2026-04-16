try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QFileDialog,
        QTextEdit,
        QGroupBox,
        QMessageBox,
        QCheckBox,
        QComboBox,
        QDialog,
        QSpinBox,
        QProgressBar,
        QMenu,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
    from PyQt5.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent, QPainter, QColor, QPolygon, QIcon

    QT_BACKEND = "PyQt5"
except Exception:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QFileDialog,
        QTextEdit,
        QGroupBox,
        QMessageBox,
        QCheckBox,
        QComboBox,
        QDialog,
        QSpinBox,
        QProgressBar,
        QMenu,
    )
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QPoint
    from PySide6.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent, QPainter, QColor, QPolygon, QIcon

    QT_BACKEND = "PySide6"

