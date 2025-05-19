"""
Language-specific keyboard layouts and letter frequencies for AAC data augmentation.
"""

import numpy as np

# Dictionary of language codes to their full names
LANGUAGE_NAMES = {
    "en": "English",
    "en-GB": "English (UK)",
    "en-US": "English (US)",
    "fr": "French",
    "fr-FR": "French (France)",
    "es": "Spanish",
    "es-ES": "Spanish (Spain)",
    "de": "German",
    "de-DE": "German (Germany)",
    "it": "Italian",
    "it-IT": "Italian (Italy)",
    "nl": "Dutch",
    "nl-NL": "Dutch (Netherlands)",
    "el": "Greek",
    "el-GR": "Greek (Greece)",
    "ru": "Russian",
    "ru-RU": "Russian (Russia)",
    "ar": "Arabic",
    "ar-SA": "Arabic (Saudi Arabia)",
    "he": "Hebrew",
    "he-IL": "Hebrew (Israel)",
    "zh": "Chinese",
    "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese",
    "ja-JP": "Japanese (Japan)",
    "pt": "Portuguese",
    "pt-BR": "Portuguese (Brazil)",
    "cy": "Welsh",
    "cy-GB": "Welsh (UK)",
}

# ===== KEYBOARD LAYOUTS =====

# English QWERTY layout
ENGLISH_QWERTY = list("QWERTYUIOPASDFGHJKLZXCVBNM_")

# French AZERTY layout
FRENCH_AZERTY = list("AZERTYUIOPQSDFGHJKLMWXCVBN_")

# German QWERTZ layout
GERMAN_QWERTZ = list("QWERTZUIOPASDFGHJKLYXCVBNM_")

# Spanish QWERTY layout (with Ñ)
SPANISH_QWERTY = list("QWERTYUIOPASDFGHJKLÑZXCVBNM_")

# Italian QWERTY layout
ITALIAN_QWERTY = list("QWERTYUIOPASDFGHJKLZXCVBNM_")

# Dutch QWERTY layout
DUTCH_QWERTY = list("QWERTYUIOPASDFGHJKLZXCVBNM_")

# Greek layout
GREEK_LAYOUT = list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ_")

# Russian layout
RUSSIAN_LAYOUT = list("ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ_")

# Hebrew layout
HEBREW_LAYOUT = list("קראטוןםפשדגכעיחלךףזסבהנמצתץ_")

# Arabic layout
ARABIC_LAYOUT = list("ضصثقفغعهخحجدشسيبلاتنمكطئءؤرلاىةوزظ_")

# Portuguese QWERTY layout
PORTUGUESE_QWERTY = list("QWERTYUIOPASDFGHJKLÇZXCVBNM_")

# Welsh QWERTY layout
WELSH_QWERTY = list("QWERTYUIOPASDFGHJKLZXCVBNM_")

# ===== LETTER FREQUENCIES =====

# English letter frequencies
ENGLISH_LETTER_FREQUENCIES = {
    "e": 0.1202,
    "t": 0.0910,
    "a": 0.0812,
    "o": 0.0768,
    "i": 0.0731,
    "n": 0.0695,
    "s": 0.0628,
    "r": 0.0602,
    "h": 0.0592,
    "d": 0.0432,
    "l": 0.0398,
    "u": 0.0288,
    "c": 0.0271,
    "m": 0.0261,
    "f": 0.0230,
    "y": 0.0211,
    "w": 0.0209,
    "g": 0.0203,
    "p": 0.0182,
    "b": 0.0149,
    "v": 0.0111,
    "k": 0.0069,
    "x": 0.0017,
    "q": 0.0011,
    "j": 0.0010,
    "z": 0.0007,
    " ": 0.1800,  # Space is the most frequent
}

# French letter frequencies
FRENCH_LETTER_FREQUENCIES = {
    "e": 0.1471,
    "a": 0.0812,
    "i": 0.0731,
    "s": 0.0790,
    "n": 0.0695,
    "r": 0.0602,
    "t": 0.0593,
    "o": 0.0536,
    "l": 0.0496,
    "u": 0.0636,
    "d": 0.0367,
    "c": 0.0318,
    "m": 0.0262,
    "p": 0.0301,
    "é": 0.0271,
    "v": 0.0164,
    "q": 0.0136,
    "f": 0.0114,
    "b": 0.0090,
    "g": 0.0087,
    "h": 0.0085,
    "j": 0.0054,
    "à": 0.0054,
    "x": 0.0038,
    "y": 0.0030,
    "z": 0.0015,
    "è": 0.0035,
    "ê": 0.0024,
    "ô": 0.0013,
    "ù": 0.0007,
    "â": 0.0006,
    "î": 0.0006,
    "û": 0.0006,
    "ç": 0.0019,
    "w": 0.0004,
    "k": 0.0001,
    " ": 0.1800,
}

# German letter frequencies
GERMAN_LETTER_FREQUENCIES = {
    "e": 0.1740,
    "n": 0.0978,
    "i": 0.0755,
    "s": 0.0727,
    "r": 0.0700,
    "a": 0.0651,
    "t": 0.0615,
    "d": 0.0508,
    "h": 0.0476,
    "u": 0.0435,
    "l": 0.0344,
    "c": 0.0306,
    "g": 0.0301,
    "m": 0.0253,
    "o": 0.0251,
    "b": 0.0189,
    "w": 0.0189,
    "f": 0.0166,
    "k": 0.0121,
    "z": 0.0113,
    "p": 0.0079,
    "v": 0.0067,
    "ü": 0.0065,
    "ä": 0.0054,
    "ö": 0.0030,
    "j": 0.0024,
    "y": 0.0004,
    "x": 0.0003,
    "q": 0.0002,
    "ß": 0.0031,
    " ": 0.1800,
}

# Spanish letter frequencies
SPANISH_LETTER_FREQUENCIES = {
    "e": 0.1368,
    "a": 0.1253,
    "o": 0.0868,
    "s": 0.0798,
    "r": 0.0687,
    "n": 0.0671,
    "i": 0.0625,
    "d": 0.0586,
    "l": 0.0497,
    "c": 0.0468,
    "t": 0.0463,
    "u": 0.0393,
    "m": 0.0315,
    "p": 0.0251,
    "b": 0.0142,
    "g": 0.0101,
    "v": 0.0090,
    "y": 0.0090,
    "q": 0.0088,
    "h": 0.0070,
    "f": 0.0069,
    "z": 0.0052,
    "j": 0.0044,
    "ñ": 0.0031,
    "x": 0.0022,
    "k": 0.0001,
    "w": 0.0001,
    " ": 0.1800,
}

# Italian letter frequencies
ITALIAN_LETTER_FREQUENCIES = {
    "e": 0.1189,
    "a": 0.1085,
    "i": 0.1026,
    "o": 0.0983,
    "n": 0.0708,
    "t": 0.0635,
    "r": 0.0633,
    "l": 0.0594,
    "s": 0.0498,
    "c": 0.0452,
    "d": 0.0375,
    "p": 0.0302,
    "u": 0.0301,
    "m": 0.0262,
    "v": 0.0211,
    "g": 0.0195,
    "f": 0.0119,
    "b": 0.0104,
    "z": 0.0095,
    "h": 0.0084,
    "q": 0.0051,
    "à": 0.0040,
    "è": 0.0040,
    "ù": 0.0010,
    "ò": 0.0010,
    "ì": 0.0010,
    "j": 0.0001,
    "k": 0.0001,
    "w": 0.0001,
    "x": 0.0001,
    "y": 0.0001,
    " ": 0.1800,
}

# Greek letter frequencies
GREEK_LETTER_FREQUENCIES = {
    "α": 0.1200,
    "ε": 0.0800,
    "ι": 0.0800,
    "τ": 0.0800,
    "ο": 0.0780,
    "ν": 0.0750,
    "σ": 0.0700,
    "η": 0.0650,
    "ρ": 0.0520,
    "κ": 0.0500,
    "π": 0.0490,
    "μ": 0.0400,
    "υ": 0.0370,
    "λ": 0.0340,
    "δ": 0.0330,
    "γ": 0.0200,
    "ω": 0.0190,
    "θ": 0.0140,
    "χ": 0.0140,
    "φ": 0.0110,
    "β": 0.0080,
    "ξ": 0.0060,
    "ζ": 0.0050,
    "ψ": 0.0040,
    " ": 0.1800,
}

# Russian letter frequencies
RUSSIAN_LETTER_FREQUENCIES = {
    "о": 0.1097,
    "е": 0.0845,
    "а": 0.0801,
    "и": 0.0735,
    "н": 0.0670,
    "т": 0.0626,
    "с": 0.0547,
    "р": 0.0473,
    "в": 0.0454,
    "л": 0.0440,
    "к": 0.0349,
    "м": 0.0321,
    "д": 0.0298,
    "п": 0.0281,
    "у": 0.0262,
    "я": 0.0201,
    "ы": 0.0190,
    "ь": 0.0174,
    "г": 0.0170,
    "з": 0.0165,
    "б": 0.0159,
    "ч": 0.0144,
    "й": 0.0121,
    "х": 0.0097,
    "ж": 0.0094,
    "ш": 0.0073,
    "ю": 0.0064,
    "ц": 0.0048,
    "щ": 0.0036,
    "э": 0.0032,
    "ф": 0.0026,
    "ъ": 0.0004,
    " ": 0.1800,
}

# Hebrew letter frequencies
HEBREW_LETTER_FREQUENCIES = {
    "י": 0.1090,
    "ו": 0.0920,
    "א": 0.0840,
    "ה": 0.0770,
    "מ": 0.0660,
    "ל": 0.0580,
    "ר": 0.0550,
    "ת": 0.0440,
    "ב": 0.0380,
    "נ": 0.0370,
    "ש": 0.0350,
    "ד": 0.0230,
    "כ": 0.0230,
    "ח": 0.0230,
    "ע": 0.0210,
    "ק": 0.0200,
    "פ": 0.0170,
    "ס": 0.0160,
    "ג": 0.0150,
    "ז": 0.0110,
    "צ": 0.0090,
    "ט": 0.0080,
    "ם": 0.0060,
    "ן": 0.0050,
    "ך": 0.0040,
    "ף": 0.0030,
    "ץ": 0.0010,
    " ": 0.1800,
}

# Arabic letter frequencies
ARABIC_LETTER_FREQUENCIES = {
    "ا": 0.1250,
    "ل": 0.0380,
    "م": 0.0290,
    "و": 0.0260,
    "ن": 0.0250,
    "ي": 0.0230,
    "ر": 0.0220,
    "ب": 0.0190,
    "ت": 0.0160,
    "ه": 0.0150,
    "د": 0.0130,
    "ع": 0.0130,
    "س": 0.0120,
    "ق": 0.0110,
    "ف": 0.0100,
    "ك": 0.0090,
    "ج": 0.0080,
    "ح": 0.0080,
    "ة": 0.0070,
    "ش": 0.0070,
    "ص": 0.0060,
    "خ": 0.0060,
    "ط": 0.0050,
    "غ": 0.0040,
    "ض": 0.0040,
    "ذ": 0.0030,
    "ث": 0.0030,
    "ز": 0.0030,
    "ظ": 0.0020,
    "أ": 0.0020,
    "إ": 0.0010,
    "ء": 0.0010,
    "ؤ": 0.0010,
    "ئ": 0.0010,
    "آ": 0.0010,
    " ": 0.1800,
}

# Portuguese letter frequencies
PORTUGUESE_LETTER_FREQUENCIES = {
    "a": 0.1463,
    "e": 0.1257,
    "o": 0.1073,
    "s": 0.0781,
    "r": 0.0653,
    "i": 0.0618,
    "n": 0.0505,
    "d": 0.0499,
    "m": 0.0474,
    "u": 0.0463,
    "t": 0.0434,
    "c": 0.0388,
    "l": 0.0278,
    "p": 0.0252,
    "v": 0.0167,
    "g": 0.0130,
    "q": 0.0120,
    "b": 0.0104,
    "f": 0.0102,
    "h": 0.0073,
    "ç": 0.0053,
    "z": 0.0047,
    "j": 0.0040,
    "á": 0.0072,
    "é": 0.0045,
    "ê": 0.0039,
    "ó": 0.0036,
    "ã": 0.0033,
    "â": 0.0015,
    "ô": 0.0015,
    "x": 0.0021,
    "k": 0.0001,
    "w": 0.0001,
    "y": 0.0001,
    " ": 0.1800,
}

# Welsh letter frequencies
WELSH_LETTER_FREQUENCIES = {
    "a": 0.0980,
    "e": 0.0800,
    "i": 0.0760,
    "n": 0.0750,
    "r": 0.0680,
    "d": 0.0650,
    "y": 0.0620,
    "o": 0.0590,
    "w": 0.0580,
    "l": 0.0470,
    "g": 0.0380,
    "s": 0.0360,
    "f": 0.0290,
    "c": 0.0280,
    "t": 0.0280,
    "h": 0.0270,
    "u": 0.0260,
    "m": 0.0240,
    "b": 0.0160,
    "p": 0.0160,
    "dd": 0.0150,
    "th": 0.0120,
    "ch": 0.0100,
    "ff": 0.0080,
    "ll": 0.0070,
    "j": 0.0001,
    "k": 0.0001,
    "q": 0.0001,
    "v": 0.0001,
    "x": 0.0001,
    "z": 0.0001,
    " ": 0.1800,
}

# Dutch letter frequencies
DUTCH_LETTER_FREQUENCIES = {
    "e": 0.1891,
    "n": 0.1003,
    "a": 0.0749,
    "t": 0.0679,
    "i": 0.0650,
    "r": 0.0641,
    "o": 0.0606,
    "d": 0.0593,
    "s": 0.0373,
    "l": 0.0357,
    "g": 0.0340,
    "v": 0.0285,
    "h": 0.0238,
    "k": 0.0225,
    "m": 0.0221,
    "u": 0.0199,
    "b": 0.0158,
    "p": 0.0157,
    "w": 0.0152,
    "j": 0.0146,
    "z": 0.0139,
    "c": 0.0124,
    "f": 0.0081,
    "x": 0.0004,
    "y": 0.0035,
    "q": 0.0009,
    " ": 0.1800,
}

# Japanese letter frequencies (simplified for common kana)
JAPANESE_LETTER_FREQUENCIES = {
    "あ": 0.0800,
    "い": 0.0750,
    "う": 0.0700,
    "え": 0.0650,
    "お": 0.0600,
    "か": 0.0550,
    "き": 0.0500,
    "く": 0.0450,
    "け": 0.0400,
    "こ": 0.0350,
    "さ": 0.0300,
    "し": 0.0280,
    "す": 0.0260,
    "せ": 0.0240,
    "そ": 0.0220,
    "た": 0.0200,
    "ち": 0.0190,
    "つ": 0.0180,
    "て": 0.0170,
    "と": 0.0160,
    "な": 0.0150,
    "に": 0.0140,
    "ぬ": 0.0130,
    "ね": 0.0120,
    "の": 0.0110,
    "は": 0.0100,
    "ひ": 0.0090,
    "ふ": 0.0080,
    "へ": 0.0070,
    "ほ": 0.0060,
    "ま": 0.0050,
    "み": 0.0040,
    "む": 0.0030,
    "め": 0.0020,
    "も": 0.0010,
    "や": 0.0050,
    "ゆ": 0.0040,
    "よ": 0.0030,
    "ら": 0.0050,
    "り": 0.0040,
    "る": 0.0030,
    "れ": 0.0020,
    "ろ": 0.0010,
    "わ": 0.0050,
    "を": 0.0040,
    "ん": 0.0030,
    " ": 0.1800,
}

# Japanese keyboard layout (simplified for common kana)
JAPANESE_LAYOUT = list(
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん_"
)

# ABC layout (alphabetical order)
ABC_LAYOUT = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ_")

# Frequency-based layout (based on English letter frequencies)
FREQUENCY_LAYOUT = list("ETAOINSRHDLUCMFYWGPBVKJXQZ_")

# ===== LANGUAGE MAPPINGS =====

# Map language codes to their keyboard layouts
KEYBOARD_LAYOUTS = {
    "en": ENGLISH_QWERTY,
    "en-GB": ENGLISH_QWERTY,
    "en-US": ENGLISH_QWERTY,
    "fr": FRENCH_AZERTY,
    "fr-FR": FRENCH_AZERTY,
    "de": GERMAN_QWERTZ,
    "de-DE": GERMAN_QWERTZ,
    "es": SPANISH_QWERTY,
    "es-ES": SPANISH_QWERTY,
    "it": ITALIAN_QWERTY,
    "it-IT": ITALIAN_QWERTY,
    "nl": DUTCH_QWERTY,
    "nl-NL": DUTCH_QWERTY,
    "el": GREEK_LAYOUT,
    "el-GR": GREEK_LAYOUT,
    "ru": RUSSIAN_LAYOUT,
    "ru-RU": RUSSIAN_LAYOUT,
    "he": HEBREW_LAYOUT,
    "he-IL": HEBREW_LAYOUT,
    "ar": ARABIC_LAYOUT,
    "ar-SA": ARABIC_LAYOUT,
    "pt": PORTUGUESE_QWERTY,
    "pt-BR": PORTUGUESE_QWERTY,
    "cy": WELSH_QWERTY,
    "cy-GB": WELSH_QWERTY,
    "ja": JAPANESE_LAYOUT,
    "ja-JP": JAPANESE_LAYOUT,
    "abc": ABC_LAYOUT,
    "frequency": FREQUENCY_LAYOUT,
}

# Map language codes to their letter frequencies
LETTER_FREQUENCIES = {
    "en": ENGLISH_LETTER_FREQUENCIES,
    "en-GB": ENGLISH_LETTER_FREQUENCIES,
    "en-US": ENGLISH_LETTER_FREQUENCIES,
    "fr": FRENCH_LETTER_FREQUENCIES,
    "fr-FR": FRENCH_LETTER_FREQUENCIES,
    "de": GERMAN_LETTER_FREQUENCIES,
    "de-DE": GERMAN_LETTER_FREQUENCIES,
    "es": SPANISH_LETTER_FREQUENCIES,
    "es-ES": SPANISH_LETTER_FREQUENCIES,
    "it": ITALIAN_LETTER_FREQUENCIES,
    "it-IT": ITALIAN_LETTER_FREQUENCIES,
    "nl": DUTCH_LETTER_FREQUENCIES,
    "nl-NL": DUTCH_LETTER_FREQUENCIES,
    "el": GREEK_LETTER_FREQUENCIES,
    "el-GR": GREEK_LETTER_FREQUENCIES,
    "ru": RUSSIAN_LETTER_FREQUENCIES,
    "ru-RU": RUSSIAN_LETTER_FREQUENCIES,
    "he": HEBREW_LETTER_FREQUENCIES,
    "he-IL": HEBREW_LETTER_FREQUENCIES,
    "ar": ARABIC_LETTER_FREQUENCIES,
    "ar-SA": ARABIC_LETTER_FREQUENCIES,
    "pt": PORTUGUESE_LETTER_FREQUENCIES,
    "pt-BR": PORTUGUESE_LETTER_FREQUENCIES,
    "cy": WELSH_LETTER_FREQUENCIES,
    "cy-GB": WELSH_LETTER_FREQUENCIES,
    "ja": JAPANESE_LETTER_FREQUENCIES,
    "ja-JP": JAPANESE_LETTER_FREQUENCIES,
    # Use English frequencies for special layouts
    "abc": ENGLISH_LETTER_FREQUENCIES,
    "frequency": ENGLISH_LETTER_FREQUENCIES,
}

# ===== HELPER FUNCTIONS =====


def get_keyboard_layout(lang_code):
    """Get the keyboard layout for a specific language code."""
    # Try the exact language code first
    if lang_code in KEYBOARD_LAYOUTS:
        return KEYBOARD_LAYOUTS[lang_code]

    # Try the base language code (e.g., 'en' for 'en-GB')
    if "-" in lang_code:
        base_lang = lang_code.split("-")[0]
        if base_lang in KEYBOARD_LAYOUTS:
            return KEYBOARD_LAYOUTS[base_lang]

    # Default to English if language not found
    print(f"Warning: No keyboard layout found for '{lang_code}'.")
    print("Using English layout.")
    return ENGLISH_QWERTY


def get_letter_frequencies(lang_code):
    """Get the letter frequencies for a specific language code."""
    # Try the exact language code first
    if lang_code in LETTER_FREQUENCIES:
        return LETTER_FREQUENCIES[lang_code]

    # Try the base language code (e.g., 'en' for 'en-GB')
    if "-" in lang_code:
        base_lang = lang_code.split("-")[0]
        if base_lang in LETTER_FREQUENCIES:
            return LETTER_FREQUENCIES[base_lang]

    # Default to English if language not found
    print(f"Warning: No letter frequencies found for '{lang_code}'.")
    print("Using English frequencies.")
    return ENGLISH_LETTER_FREQUENCIES


def create_language_abc_grid(lang_code, rows, cols, fillers=None):
    """Create an ABC grid for a specific language."""
    if fillers is None:
        fillers = ["", "?", ".", ",", "!"]

    # Get the alphabet for this language
    if lang_code in KEYBOARD_LAYOUTS:
        # Use the first 26 characters (or fewer if the alphabet is shorter)
        alphabet = KEYBOARD_LAYOUTS[lang_code]
    else:
        # Default to English alphabet
        alphabet = ENGLISH_QWERTY

    # Ensure we have a space character (represented as '_')
    if "_" not in alphabet:
        alphabet.append("_")

    # Create the grid
    grid_size = rows * cols

    # Calculate how many fillers are needed
    num_fillers_needed = grid_size - len(alphabet)
    if num_fillers_needed > 0:
        # Use available fillers and pad with "" if needed
        extended_fillers = fillers[:num_fillers_needed] + [""] * (
            num_fillers_needed - len(fillers)
        )
        extended_alphabet = alphabet + extended_fillers
    else:
        # Trim alphabet if grid is smaller
        extended_alphabet = alphabet[:grid_size]

    # Create and return the grid as a NumPy array
    return np.array(extended_alphabet).reshape(rows, cols)


def create_language_qwerty_grid(lang_code, rows, cols, fillers=None):
    """Create a QWERTY-like grid for a specific language."""
    if fillers is None:
        fillers = ["", "?", ".", ",", "!"]

    # Get the keyboard layout for this language
    keyboard_layout = get_keyboard_layout(lang_code)

    # Create the grid
    grid_size = rows * cols

    # Calculate how many fillers are needed
    num_fillers_needed = grid_size - len(keyboard_layout)
    if num_fillers_needed > 0:
        # Use available fillers and pad with "" if needed
        extended_fillers = fillers[:num_fillers_needed] + [""] * (
            num_fillers_needed - len(fillers)
        )
        extended_layout = keyboard_layout + extended_fillers
    else:
        # Trim layout if grid is smaller
        extended_layout = keyboard_layout[:grid_size]

    # Create and return the grid as a NumPy array
    return np.array(extended_layout).reshape(rows, cols)


def create_language_frequency_grid(lang_code, rows, cols, fillers=None):
    """Create a frequency-based grid for a specific language."""
    if fillers is None:
        fillers = ["", "?", ".", ",", "!"]

    # Get the letter frequencies for this language
    letter_frequencies = get_letter_frequencies(lang_code)

    # Sort letters by frequency (highest first)
    sorted_letters = sorted(
        letter_frequencies.keys(), key=lambda x: -letter_frequencies[x]
    )

    # Replace space with '_'
    sorted_letters = [char.upper() if char != " " else "_" for char in sorted_letters]

    # Create the grid
    grid_size = rows * cols

    # Calculate how many fillers are needed
    num_fillers_needed = grid_size - len(sorted_letters)
    if num_fillers_needed > 0:
        # Use available fillers and pad with "" if needed
        extended_fillers = fillers[:num_fillers_needed] + [""] * (
            num_fillers_needed - len(fillers)
        )
        extended_letters = sorted_letters + extended_fillers
    else:
        # Trim letters if grid is smaller
        extended_letters = sorted_letters[:grid_size]

    # Create and return the grid as a NumPy array
    return np.array(extended_letters).reshape(rows, cols)
