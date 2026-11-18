"""Short forms used in the manuscript figures (one source for s13, s16 and the dataset table of the Letter).
Each entry: id -> (code, full name)."""

DATASETS = {
    "kopin": ("NG7", "NailGroup-7 development set (this work)"),
    "han": ("HAN", "Han et al. hospital sets A1 + A2 + B-E"),
    "han_A1": ("HA1", "Han et al. A1 (hospital thumbnails)"),
    "han_A2": ("HA2", "Han et al. A2 (hospital)"),
    "han_BCDE": ("HBE", "Han et al. B-E (hospital, onychomycosis)"),
    "han_virtualE": ("HVE", "Han et al. Virtual E (web)"),
    "thai": ("THA", "Thai Kaggle sets (ART1 + ART2 + NUT)"),
    "arthitaya": ("ART1", "Kaggle arthitaya"),
    "arthitaya2": ("ART2", "Kaggle arthitaya-2"),
    "nuttida": ("NUT", "Kaggle nuttidalapthanachai"),
    "josephrasanjana": ("JOS", "Kaggle josephrasanjana"),
    "nikhilgurav21": ("NIK", "Kaggle nikhilgurav21 (KNM)"),
    "kaggle17": ("K17", "Kaggle 17-class"),
    "farisfaikar": ("MEL", "Kaggle nail-melanoma-350"),
    "isratjahanmitu": ("ISR", "Kaggle isratjahanmitu"),
    "legacy2272": ("CMB", "combined Roboflow mohit + Kaggle KNM set"),
    "own_test": ("own", "held-out part of the training dataset"),
}

CLASSES = {
    "beau_s": ("BEA", "Beau's lines"),
    "black_line": ("BLK", "Black line"),
    "clubbing": ("CLB", "Clubbing"),
    "healthy_nail": ("HLT", "Healthy nail"),
    "onychomycosis": ("ONM", "Onychomycosis"),
    "onycholysis": ("ONL", "Onycholysis"),
    "white_spot": ("WSP", "White spot"),
}


def code(key: str) -> str:
    return (DATASETS.get(key) or CLASSES.get(key) or (key, key))[0]


def legend_items(keys, table=DATASETS):
    return [(table[k][0], table[k][1]) for k in keys if k in table]
