import re
import unicodedata



TR_UPPER = str.maketrans({"I": "ı", "İ": "i"})
APOSTROPHES = re.compile(r"['’`]")
PUNCT = re.compile(r"[^\w\s]")
SPACES = re.compile(r"\s+")

def normalize_tr(text):
    text = unicodedata.normalize("NFC", text)
    text = text.translate(TR_UPPER).lower()
    text = APOSTROPHES.sub("", text) # Ali'nin == Alinin
    text = PUNCT.sub(" ", text)
    text = SPACES.sub(" ", text).strip()
    return text
