from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import os

def create_header(title, subtitle=None):
    container = QWidget()
    container.setObjectName("headerWidget")
    container.setMinimumHeight(64)
    container.setMaximumHeight(76)
    v_main = QHBoxLayout(container)
    v_main.setContentsMargins(24, 12, 24, 12)
    v_main.setSpacing(16)

    left_logo = QLabel(container)
    left_logo.setObjectName("leftLogo")
    try:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        usak_path = os.path.join(proj_root, 'images', 'usak.png')
        if os.path.exists(usak_path):
            pm = QPixmap(usak_path)
            if not pm.isNull():
                pass # left_logo.setPixmap(pm)
                left_logo.setScaledContents(True)
                left_logo.setFixedHeight(34)
                left_logo.setFixedWidth(int(34 * (pm.width() / max(1, pm.height()))))
    except Exception:
        pass

    right_logo = QLabel(container)
    try:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        fau_path = os.path.join(proj_root, 'images', 'fau.png')
        if os.path.exists(fau_path):
            pm2 = QPixmap(fau_path)
            if not pm2.isNull():
                pass # right_logo.setPixmap(pm2)
                right_logo.setScaledContents(True)
                right_logo.setFixedHeight(34)
                right_logo.setFixedWidth(int(34 * (pm2.width() / max(1, pm2.height()))))
    except Exception:
        pass

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
    v_main.addStretch(1)
    v_main.addWidget(right_logo, 0, Qt.AlignmentFlag.AlignVCenter)

    container.globalInfoButton = info_btn

    return container
