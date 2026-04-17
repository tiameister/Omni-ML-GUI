import re

with open("interface/style/style.qss", "r") as f:
    text = f.read()

# Replace hardcoded light mode colors with Dark Mode macOS colors
replacements = {
    r"#FFFFFF": "#1E1E1E", # White to Dark surface
    r"#EAF0F6": "#121212", # appCanvas background
    r"rgba\(255, 255, 255, 0.94\)": "rgba(30, 30, 30, 0.94)",
    r"#CFDCEC": "#38383A", # borders
    r"#F8FBFF": "#2C2C2E", # header top
    r"#ECF3FA": "#242426", # header bottom
    r"#102333": "#FFFFFF", # sectionTitle text
    r"#5B6C7B": "#98989D", # hintLabel
    r"#E7F3FF": "#1C1C1E", # workflowBanner
    r"#EAF8F6": "#2C2C2E",
    r"#103756": "#FFFFFF",
    r"#BFD9ED": "#3A3A3C",
    r"#D2DEEC": "#3A3A3C",
    r"#5A6E81": "#98989D",
    r"#0D2B43": "#FFFFFF",
    r"#F8FCFF": "#1C1C1E",
    r"#C7DDED": "#3A3A3C",
    r"#163B59": "#FFFFFF",
    r"#EBEBF0": "#3A3A3C",
    r"#EBEBEE": "#2C2C2E",
    r"#F9F9FB": "#1C1C1E",
    r"#D1E5F9": "#3A3A3C",
    r"#F0F7FF": "#2C2C2E",
    r"#0051A3": "#0A84FF",
    r"#0F3A5C": "#FFFFFF",
    r"#0E304A": "#E0E0E0",
    r"#C9D9EA": "#3A3A3C",
    r"#ECF2F8": "#2C2C2E",
    r"#EAF4FF": "#2A2A2C",
    r"#0D3E62": "#FFFFFF",
    r"#E6EEF7": "#3A3A3C",
    r"#0E3E62": "#FFFFFF",
    r"#F4F9FF": "#2A2A2C",
    r"#112A3D": "#FFFFFF",
    r"#DCDCE0": "#48484A",
    r"#519DD9": "#0A84FF",
    r"#F4FAFF": "#3A3A3C",
    r"#E9F2FB": "#2C2C2E",
    r"#9AA8B7": "#636366",
    r"#E1E8F1": "#3A3A3C",
    r"#F7F9FB": "#1C1C1E",
    r"#007AFF": "#0A84FF",
    r"#0062CC": "#005EBD",
    r"#0052A3": "#004085",
    r"#F4F8FC": "#2C2C2E",
    r"#21405A": "#FFFFFF",
    r"#EBF3FC": "#3A3A3C",
    r"#90B8DD": "#0A84FF",
    r"#B0B0B8": "#636366",
    r"#6C7A89": "#98989D",
    r"rgba\(0, 94, 168, 0.16\)": "rgba(10, 132, 255, 0.25)",
    r"#F2F2F7": "#2C2C2E",
    r"#E5E5EA": "#3A3A3C",
    r"#E9F5FF": "#2A2A2C",
    r"#0F3E63": "#FFFFFF",
    r"#EFF4FA": "#1C1C1E",
    r"#1E3347": "#FFFFFF",
    r"#D4DFEC": "#3A3A3C",
    r"#E2EAF3": "#3A3A3C",
    r"#F7FBFF": "#1E1E1E",
    r"#CFE0F0": "#3A3A3C",
    r"#184364": "#FFFFFF",
    r"#144265": "#FFFFFF",
    r"#BFD8EC": "#3A3A3C",
    r"#D7E2EF": "#3A3A3C",
    r"#153A58": "#FFFFFF",
    r"#C8D8EA": "#3A3A3C",
    r"#3D8ED1": "#5E5CE6",
    r"#1A2530": "#FFFFFF",
    r"color: #102333;": "color: #FFFFFF;"
}

# Also need to manually switch scrollbars to look dark or transparent.

for old, new in replacements.items():
    text = re.sub(old, new, text)

with open("interface/style/dark_style.qss", "w") as f:
    f.write(text)
print("done")
