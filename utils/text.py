import re


def normalize_text(s: str) -> str:
    txt = str(s).strip()
    mapping = str.maketrans({
        "I": "I", "i": "i", "İ": "I", "ı": "i",
        "Ş": "S", "ş": "s",
        "Ğ": "G", "ğ": "g",
        "Ü": "U", "ü": "u",
        "Ö": "O", "ö": "o",
        "Ç": "C", "ç": "c",
    })
    txt = txt.translate(mapping).lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt



_QUOTE_TRANS = str.maketrans({
    "’": "'", "‘": "'", "‛": "'", "ʼ": "'", "ʹ": "'", "ʽ": "'", "ˈ": "'", "′": "'", "‵": "'", "＇": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"', "＂": '"',
    "\uFFFD": "'",
    "\u201A": "'",
    "\u00A0": " ",
    "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"',
})

def normalize_quotes_ascii(s: str) -> str:
    return str(s).translate(_QUOTE_TRANS)
