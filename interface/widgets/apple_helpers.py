from PyQt6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

def create_apple_settings_row(right_widget, title_text="", subtitle_text="", show_bottom_line=True):
    row = QFrame()
    row.setObjectName("appleRow" if show_bottom_line else "appleRowLast")
    
    lay = QHBoxLayout(row)
    lay.setContentsMargins(16, 12, 16, 12)
    lay.setSpacing(16)
    
    text_lay = QVBoxLayout()
    text_lay.setContentsMargins(0, 0, 0, 0)
    text_lay.setSpacing(2)
    
    title = QLabel(title_text)
    title.setObjectName("appleRowTitle")
    title.setWordWrap(True)
    text_lay.addWidget(title)
    
    subtitle = QLabel(subtitle_text)
    subtitle.setObjectName("appleRowSubtitle")
    subtitle.setWordWrap(True)
    text_lay.addWidget(subtitle)
    
    text_lay.addStretch(1)
    lay.addLayout(text_lay)
    
    if right_widget:
        lay.addWidget(right_widget, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
    return row, title, subtitle
