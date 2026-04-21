from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from utils.logger import get_logger
from utils.paths import get_project_root

LOGGER = get_logger(__name__)

def create_header(title, subtitle=None):
    container = QWidget()
    container.setObjectName("headerWidget")
    container.setMinimumHeight(64)
    container.setMaximumHeight(76)
    v_main = QHBoxLayout(container)
    v_main.setContentsMargins(24, 12, 24, 12)
    v_main.setSpacing(16)

    _img_root = get_project_root() / "images"

    left_logo = QLabel(container)
    left_logo.setObjectName("leftLogo")
    try:
        usak_path = _img_root / "usak.png"
        if usak_path.exists():
            pm = QPixmap(str(usak_path))
            if not pm.isNull():
                left_logo.setPixmap(pm)
                left_logo.setScaledContents(True)
                left_logo.setFixedHeight(34)
                left_logo.setFixedWidth(int(34 * (pm.width() / max(1, pm.height()))))
    except Exception:
        LOGGER.exception("Header logo load failed (left)")

    right_logo = QLabel(container)
    try:
        fau_path = _img_root / "fau.png"
        if fau_path.exists():
            pm2 = QPixmap(str(fau_path))
            if not pm2.isNull():
                right_logo.setPixmap(pm2)
                right_logo.setScaledContents(True)
                right_logo.setFixedHeight(34)
                right_logo.setFixedWidth(int(34 * (pm2.width() / max(1, pm2.height()))))
    except Exception:
        LOGGER.exception("Header logo load failed (right)")

    v_main.addWidget(left_logo, 0, Qt.AlignmentFlag.AlignVCenter)

    title_stack = QVBoxLayout()
    title_stack.setContentsMargins(0, 0, 0, 0)
    title_stack.setSpacing(1)

    lbl = QLabel(title)
    lbl.setObjectName("titleLabel")
    lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    subtitle_lbl = QLabel(subtitle or "")
    subtitle_lbl.setObjectName("subtitleLabel")
    subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    subtitle_lbl.setWordWrap(False)

    title_stack.addWidget(lbl)
    title_stack.addWidget(subtitle_lbl)
    v_main.addLayout(title_stack, 1)

    actions_layout = QHBoxLayout()
    actions_layout.setContentsMargins(0, 0, 0, 0)
    actions_layout.setSpacing(6)

    info_btn = QPushButton("i")
    info_btn.setObjectName("globalInfoButton")

    actions_layout.addWidget(info_btn)

    v_main.addLayout(actions_layout)
    v_main.addWidget(right_logo, 0, Qt.AlignmentFlag.AlignVCenter)

    container.globalInfoButton = info_btn

    return container
