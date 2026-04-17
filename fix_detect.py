import re

with open("data/loader.py", "r", encoding="utf-8") as f:
    text = f.read()

# Combining the drop_cols detection loops into one loop.
old_detect = """    drop_cols = [c for c, n in nm.items() if n in drop_norms]
    drop_cols += [c for c, n in nm.items() if re.match(r"faal", n)]
    drop_cols += [c for c, n in nm.items() if n == "id"]
    drop_cols += [c for c, n in nm.items() if re.match(r"(irk|etnik|etnisite|ethnicity|race)(_|$)", n)]"""

new_detect = """    # Compile regexes once
    re_faal = re.compile(r"faal")
    re_sensitive = re.compile(r"(irk|etnik|etnisite|ethnicity|race)(_|$)")
    
    # Collect drop columns efficiently with O(N) single-pass lookup
    drop_cols = []
    for c, n in nm.items():
        if n in drop_norms or n == "id" or re_faal.match(n) or re_sensitive.match(n):
            drop_cols.append(c)"""

text = text.replace(old_detect, new_detect)

with open("data/loader.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Optimized column scanning logic!")
