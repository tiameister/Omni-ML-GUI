"""Application-wide error reporting for the desktop GUI.

In a packaged ``.exe`` build there is no console attached, so an
unhandled exception that escapes our normal try/except guards would
otherwise vanish silently. This module installs hooks that route every
unhandled error — whether raised on the main GUI thread, a Python
worker thread, or a Qt slot — into a clean modal dialog. The dialog
shows a calm headline with the technical traceback collapsed under a
"Show Details" toggle so support staff can still grab the stack.

Public surface
--------------
* :class:`ErrorDialog` — the modal itself.
* :func:`install_global_exception_handlers` — call once during app
  startup, before any user interaction.
* :func:`show_error` — programmatically trigger the same dialog from
  anywhere (e.g. a worker thread that wants to surface a non-fatal
  problem).
"""
from __future__ import annotations

import sys
import threading
import traceback
from typing import Any

from PySide6.QtCore import QMetaObject, QObject, Qt, Q_ARG, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from utils.logger import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dialog widget
# ---------------------------------------------------------------------------
class ErrorDialog(QDialog):
    """Modal error dialog with a collapsible stack-trace panel.

    The dialog is intentionally framework-agnostic so it can also be
    used outside of unhandled-exception flows (e.g. by validation
    failures) via :func:`show_error`.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str = "Unexpected Error",
        message: str = "An unexpected error occurred.",
        details: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(540)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(14)
        icon_label = QLabel(self)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
        icon_label.setPixmap(icon.pixmap(48, 48))
        icon_label.setFixedSize(48, 48)
        header.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)

        message_label = QLabel(message, self)
        message_label.setWordWrap(True)
        message_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg_font = message_label.font()
        msg_font.setPointSize(msg_font.pointSize() + 1)
        message_label.setFont(msg_font)
        header.addWidget(message_label, 1)
        outer.addLayout(header)

        secondary = QLabel(
            "Your session is intact — you can keep working. If this happens "
            "repeatedly, copy the technical details below when reporting it.",
            self,
        )
        secondary.setWordWrap(True)
        secondary.setStyleSheet("color: palette(mid);")
        outer.addWidget(secondary)

        toggle_row = QHBoxLayout()
        self._details_btn = QPushButton("Show Details", self)
        self._details_btn.setCheckable(True)
        self._details_btn.setChecked(False)
        self._details_btn.toggled.connect(self._toggle_details)
        toggle_row.addWidget(self._details_btn, 0, Qt.AlignmentFlag.AlignLeft)
        toggle_row.addStretch(1)
        outer.addLayout(toggle_row)

        self._details_view = QPlainTextEdit(self)
        self._details_view.setReadOnly(True)
        self._details_view.setPlainText(details or "(no traceback available)")
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(self.font().pointSize())
        self._details_view.setFont(mono)
        self._details_view.setMinimumHeight(220)
        self._details_view.setVisible(False)
        outer.addWidget(self._details_view)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        copy_btn = buttons.addButton("Copy Details", QDialogButtonBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(self._copy_details)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

        self.adjustSize()

    @Slot(bool)
    def _toggle_details(self, expanded: bool) -> None:
        self._details_view.setVisible(expanded)
        self._details_btn.setText("Hide Details" if expanded else "Show Details")
        self.adjustSize()

    def _copy_details(self) -> None:
        clip = QApplication.clipboard()
        if clip is not None:
            clip.setText(self._details_view.toPlainText())


# ---------------------------------------------------------------------------
# Cross-thread router
# ---------------------------------------------------------------------------
class _ExceptionRouter(QObject):
    """QObject singleton that marshals error reports onto the GUI thread.

    Any thread can call :meth:`report`; the actual dialog is shown via
    a queued connection so it never blocks the originating thread and
    cannot crash the Qt event loop with cross-thread widget access.
    """

    _instance: "_ExceptionRouter | None" = None

    def __init__(self) -> None:
        super().__init__()
        # Recursion guard: if the dialog itself somehow raises while
        # showing, we must not feed that exception back into the hook
        # and infinite-loop the app.
        self._in_dialog = False

    @classmethod
    def instance(cls) -> "_ExceptionRouter":
        if cls._instance is None:
            cls._instance = _ExceptionRouter()
        return cls._instance

    @Slot(str, str, str)
    def _show_error(self, title: str, message: str, details: str) -> None:
        if self._in_dialog:
            return
        self._in_dialog = True
        try:
            app = QApplication.instance()
            parent = app.activeWindow() if app is not None else None
            dlg = ErrorDialog(parent=parent, title=title, message=message, details=details)
            dlg.exec()
        except Exception:
            LOGGER.exception("ErrorDialog failed to display; falling back to log only.")
        finally:
            self._in_dialog = False

    def report(self, title: str, message: str, details: str) -> None:
        QMetaObject.invokeMethod(
            self,
            "_show_error",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, title),
            Q_ARG(str, message),
            Q_ARG(str, details),
        )


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------
def _format_exception(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> str:
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


def _user_message_for(exc_type: type[BaseException]) -> str:
    name = getattr(exc_type, "__name__", str(exc_type))
    return f"An unexpected error occurred ({name}). The application is still running."


def install_global_exception_handlers() -> None:
    """Install process-wide exception hooks.

    The hooks:

    * Replace ``sys.excepthook`` so any unhandled exception on the main
      thread surfaces as an :class:`ErrorDialog`.
    * Replace ``threading.excepthook`` so exceptions raised on
      non-main threads (Python's ``threading`` module — the joblib
      ``loky`` workers run in subprocesses and are unaffected) are also
      surfaced via the dialog.
    * Always log the full traceback before showing the dialog, so the
      ``ml_trainer.gui.log`` file remains the canonical record.

    Idempotent: calling this function more than once just re-installs
    the same hooks.
    """
    crash_logger = get_logger("ml_trainer.crash")
    router = _ExceptionRouter.instance()

    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        details = _format_exception(exc_type, exc_value, exc_tb)
        crash_logger.error("Unhandled main-thread exception:\n%s", details)
        router.report("Unexpected Error", _user_message_for(exc_type), details)

    def _thread_hook(args: "threading.ExceptHookArgs") -> None:
        exc_type = args.exc_type
        if exc_type is None or issubclass(exc_type, SystemExit):
            return
        details = _format_exception(exc_type, args.exc_value, args.exc_traceback)
        crash_logger.error(
            "Unhandled exception in thread %r:\n%s",
            getattr(args.thread, "name", "<unknown>"),
            details,
        )
        router.report("Unexpected Error", _user_message_for(exc_type), details)

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook
    LOGGER.info("Global exception handlers installed.")


def show_error(message: str, details: str = "", *, title: str = "Error") -> None:
    """Programmatic entry point for surfacing an error from any thread."""
    _ExceptionRouter.instance().report(title, message, details)


__all__ = [
    "ErrorDialog",
    "install_global_exception_handlers",
    "show_error",
]
