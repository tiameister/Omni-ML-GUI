from PyQt6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

def create_apple_settings_row(right_widget, title_text="", subtitle_text="", show_bottom_line=True):
    container = QFrame()
    container.setObjectName("appleRowContainer")
    container_lay = QVBoxLayout(container)
    container_lay.setContentsMargins(0, 0, 0, 0)
    container_lay.setSpacing(0)
    
    row = QFrame()
    row.setObjectName("appleRowContent")
    
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
    lay.addLayout(text_lay, 1)
    
    if right_widget:
        # Give right widget stretch of 0, aligned right
        lay.addWidget(right_widget, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
    container_lay.addWidget(row)
    
    if show_bottom_line:
        line_container = QFrame()
        line_container.setObjectName("appleRowDividerContainer")
        line_lay = QHBoxLayout(line_container)
        line_lay.setContentsMargins(16, 0, 0, 0) # indented from left
        line_lay.setSpacing(0)
        
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #E5E5EA; border: none;")
        line_lay.addWidget(line)
        container_lay.addWidget(line_container)
        
    return container, title, subtitle
