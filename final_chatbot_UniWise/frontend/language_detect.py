"""
language_detect.py

Detects which language (or dialect) a user's chat message is written in, so
the chatbot can reply in that same language instead of being locked to one.
Built for broad multilingual coverage: Latin-script European/Southeast-Asian
languages, Tagalog/Taglish, and non-Latin scripts (Arabic, Hebrew, Cyrillic,
Devanagari, Greek, Thai, Chinese/Japanese/Korean, and several South Asian
scripts).

Kept as its own file (separate from bot.py / app.py / scope_filter.py) so
the detection logic is easy to find, tune, and unit-test on its own, and so
any file that needs it can just `from language_detect import ...`.

Layered approach (fastest / most-trustworthy checks first):

  1. Tagalog / Taglish marker check -- run first because a generic detector
     (langdetect below) frequently misclassifies short Tagalog/Taglish
     messages as Indonesian, Spanish, or plain English. A dedicated
     marker-word check catches these correctly and wins if it matches.

  2. Non-Latin script detection -- for scripts that (almost) always mean one
     specific language regardless of message length -- Chinese/Japanese/
     Korean, Thai, Greek, Hebrew -- the Unicode block alone is enough.
     For scripts shared by several languages -- Cyrillic (Russian,
     Ukrainian, Bulgarian, Macedonian...), Arabic script (Arabic, Persian,
     Urdu), Devanagari (Hindi, Marathi, Nepali) -- the script narrows it to
     a default language, then `langdetect` is asked to refine *within that
     script family* if the message is long enough to trust.

  3. Common short greeting lookup -- catches very short first messages
     ("Hola", "Bonjour", "Merhaba"...) in Latin-script languages that would
     otherwise be too short for `langdetect` to read reliably.

  4. English quick check -- English is this bot's base language and by far
     the most common case, so it gets a cheap, reliable marker-word check
     of its own rather than a statistical guess every time.

  5. General fallback -- `langdetect` (an unofficial Python port of
     Google's language-detection library, ~55 languages) handles everything
     else: Spanish, Malay, Italian, German, French, Vietnamese, Polish,
     Swedish, Turkish, and most other Latin-script languages. Uses
     confidence scores (not just the top guess) with a length-scaled
     threshold, since langdetect is well-known to be unreliable on short
     strings -- the shorter the message, the more confident it has to be
     before we trust it over the English default.

Falls back to English if `langdetect` isn't installed
(`pip install langdetect --break-system-packages`) or nothing above is
confident enough to call it.

Exposes:
  detect_language(text)                -> {"code": "tl", "name": "Tagalog"}
  get_reply_language_instruction(text) -> ready-to-inject sentence for the
                                           LLM system prompt telling it which
                                           language/dialect to answer in.
"""

import difflib
import re

try:
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0  # deterministic results across runs
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


def _tokenize(text):
    """Splits on anything that isn't a Unicode letter, keeping accented
    letters (e.g. Vietnamese "làm", Polish "zapisać") as whole tokens.
    Using a plain [a-zA-Z] pattern here would chop those words apart at
    every accented character, leaving short ASCII fragments ("l", "m",
    "ng"...) that can accidentally collide with unrelated marker words --
    a real bug this previously caused with the Tagalog check misfiring on
    Vietnamese text."""
    return re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)


# ── TAGALOG / TAGLISH ────────────────────────────────────────────
# Same marker list style as scope_filter.py's detect_language, kept here too
# so this file works standalone without importing scope_filter.
_TAGALOG_MARKERS = {
    "ang", "ng", "mga", "sa", "ako", "ikaw", "ka", "siya", "kami", "tayo",
    "kayo", "sila", "ito", "iyan", "iyon", "dito", "diyan", "doon",
    "paano", "bakit", "kailan", "saan", "sino", "ano", "alin", "ilan",
    "kamusta", "kumusta", "magandang", "salamat", "maraming", "po", "opo",
    "hindi", "oo", "wala", "meron", "mayroon", "gusto", "kailangan",
    "pwede", "puwede", "paki", "pakiusap", "tulungan", "tulong",
    "mag-enroll", "magparehestro", "paaralan", "eskwela", "eskuwela",
    "guro", "titser", "estudyante", "mag-aaral", "anak", "sige", "ayos",
    "tapos", "saka", "tsaka", "din", "rin", "lang", "naman", "ba",
    "yung", "yun", "nung", "pag", "kung", "para", "dahil", "kasi",
    "namin", "natin", "niya", "nila", "mo", "ko", "niyo", "ninyo",
}


_TAGALOG_HYPHENATED_MARKERS = ("mag-enroll",)


def _looks_tagalog(text):
    lowered = text.lower()
    if any(marker in lowered for marker in _TAGALOG_HYPHENATED_MARKERS):
        return True
    words = _tokenize(text)
    if not words:
        return False
    hits = sum(1 for w in words if w in _TAGALOG_MARKERS)
    if hits == 0:
        return False
    # Two or more marker words is a safe bet regardless of message length;
    # for longer messages, also accept if markers make up a decent share of
    # the words (catches a short Tagalog sentence with only one marker word).
    return hits >= 2 or (hits / len(words)) >= 0.2


# ── NON-LATIN SCRIPT DETECTION ────────────────────────────────────
# Unicode block ranges. Order matters where blocks could otherwise collide
# (e.g. Hiragana/Katakana checked before the wider CJK ideograph block).
_SCRIPT_RANGES = [
    ("hiragana_katakana", [(0x3040, 0x309F), (0x30A0, 0x30FF)]),
    ("hangul", [(0xAC00, 0xD7A3), (0x1100, 0x11FF)]),
    ("cjk", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
    ("thai", [(0x0E00, 0x0E7F)]),
    ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)]),
    ("hebrew", [(0x0590, 0x05FF)]),
    ("cyrillic", [(0x0400, 0x04FF)]),
    ("greek", [(0x0370, 0x03FF)]),
    ("devanagari", [(0x0900, 0x097F)]),
    ("bengali", [(0x0980, 0x09FF)]),
    ("gurmukhi", [(0x0A00, 0x0A7F)]),
    ("gujarati", [(0x0A80, 0x0AFF)]),
    ("tamil", [(0x0B80, 0x0BFF)]),
    ("telugu", [(0x0C00, 0x0C7F)]),
    ("kannada", [(0x0C80, 0x0CFF)]),
    ("malayalam", [(0x0D00, 0x0D7F)]),
    ("armenian", [(0x0530, 0x058F)]),
    ("georgian", [(0x10A0, 0x10FF)]),
]

# What each script defaults to. For scripts shared by several languages,
# _SCRIPT_REFINEMENT_CODES below lists which langdetect codes are accepted
# as a refinement away from this default.
_SCRIPT_DEFAULT = {
    "hiragana_katakana": {"code": "ja", "name": "Japanese"},
    "hangul": {"code": "ko", "name": "Korean"},
    "cjk": {"code": "zh", "name": "Chinese"},
    "thai": {"code": "th", "name": "Thai"},
    "hebrew": {"code": "he", "name": "Hebrew"},
    "greek": {"code": "el", "name": "Greek"},
    "bengali": {"code": "bn", "name": "Bengali"},
    "gurmukhi": {"code": "pa", "name": "Punjabi"},
    "gujarati": {"code": "gu", "name": "Gujarati"},
    "tamil": {"code": "ta", "name": "Tamil"},
    "telugu": {"code": "te", "name": "Telugu"},
    "kannada": {"code": "kn", "name": "Kannada"},
    "malayalam": {"code": "ml", "name": "Malayalam"},
    "armenian": {"code": "hy", "name": "Armenian"},
    "georgian": {"code": "ka", "name": "Georgian"},
    "arabic": {"code": "ar", "name": "Arabic"},
    "cyrillic": {"code": "ru", "name": "Russian"},
    "devanagari": {"code": "hi", "name": "Hindi"},
}

# For a script whose default could plausibly be a different specific
# language, which langdetect codes are acceptable refinements.
_SCRIPT_REFINEMENT_CODES = {
    "arabic": {"ar": ("ar", "Arabic"), "fa": ("fa", "Persian"), "ur": ("ur", "Urdu")},
    "cyrillic": {
        "ru": ("ru", "Russian"), "uk": ("uk", "Ukrainian"),
        "bg": ("bg", "Bulgarian"), "mk": ("mk", "Macedonian"),
    },
    "devanagari": {"hi": ("hi", "Hindi"), "mr": ("mr", "Marathi"), "ne": ("ne", "Nepali")},
}


def _in_range(ch, rng):
    return rng[0] <= ord(ch) <= rng[1]


def _detect_script(text):
    for script_name, ranges in _SCRIPT_RANGES:
        if any(_in_range(c, rng) for c in text for rng in ranges):
            return script_name
    return None


def _refine_within_script(script_name, text):
    """For a script shared by multiple languages, ask langdetect which one
    it most likely is -- but only accept an answer that's actually a member
    of that script family, so e.g. a short Arabic-script message can never
    get "corrected" into something in a completely different alphabet."""
    refinements = _SCRIPT_REFINEMENT_CODES.get(script_name)
    if not refinements or not _LANGDETECT_AVAILABLE or len(text) < 15:
        return None
    try:
        for guess in detect_langs(text):
            if guess.lang in refinements and guess.prob >= 0.6:
                code, name = refinements[guess.lang]
                return {"code": code, "name": name}
    except LangDetectException:
        pass
    return None


# ── COMMON SHORT GREETINGS (Latin-script) ─────────────────────────
# Greetings are often the very first, very short message in a chat --
# exactly the case where langdetect is least reliable. A small explicit
# lookup catches the common ones directly across many Latin-script
# languages.
_GREETING_WORDS = {
    "hola": {"code": "es", "name": "Spanish"}, "buenos": {"code": "es", "name": "Spanish"},
    "bonjour": {"code": "fr", "name": "French"}, "salut": {"code": "fr", "name": "French"},
    "ciao": {"code": "it", "name": "Italian"}, "salve": {"code": "it", "name": "Italian"},
    "hallo": {"code": "de", "name": "German"}, "guten": {"code": "de", "name": "German"},
    "selamat": {"code": "ms", "name": "Malay"}, "apa": {"code": "ms", "name": "Malay"},
    "olá": {"code": "pt", "name": "Portuguese"}, "ola": {"code": "pt", "name": "Portuguese"},
    "annyeong": {"code": "ko", "name": "Korean"},
    "konnichiwa": {"code": "ja", "name": "Japanese"},
    "nihao": {"code": "zh", "name": "Chinese"},
    "merhaba": {"code": "tr", "name": "Turkish"}, "selam": {"code": "tr", "name": "Turkish"},
    "cześć": {"code": "pl", "name": "Polish"}, "czesc": {"code": "pl", "name": "Polish"},
    "hej": {"code": "sv", "name": "Swedish"}, "hallå": {"code": "sv", "name": "Swedish"},
    "hoi": {"code": "nl", "name": "Dutch"},
    "xinchao": {"code": "vi", "name": "Vietnamese"}, "chào": {"code": "vi", "name": "Vietnamese"},
    "ahoj": {"code": "cs", "name": "Czech"},
    "szia": {"code": "hu", "name": "Hungarian"},
    "namaste": {"code": "hi", "name": "Hindi"},
    "jambo": {"code": "sw", "name": "Swahili"}, "habari": {"code": "sw", "name": "Swahili"},
}


def _check_greeting(text):
    words = _tokenize(text)

    # Exact match first.
    for w in words:
        if w in _GREETING_WORDS:
            return _GREETING_WORDS[w]

    # Fuzzy fallback -- catches common alternate spellings/romanizations
    # (e.g. "konichiwa" vs "konnichiwa", a very common one-letter variant
    # that would otherwise fall through to langdetect and get misread as
    # something unrelated on a short string like this).
    best_key, best_ratio = None, 0.0
    for w in words:
        if len(w) < 4:
            continue
        for key in _GREETING_WORDS:
            if abs(len(key) - len(w)) > 2:
                continue
            ratio = difflib.SequenceMatcher(None, w, key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
    if best_key and best_ratio >= 0.82:
        return _GREETING_WORDS[best_key]

    return None


# ── ENGLISH QUICK CHECK ──────────────────────────────────────────
# langdetect is trained on longer documents and is unreliable on short chat
# messages -- a plain English sentence like "How do I enroll for Grade 11?"
# can get misread as Portuguese or Welsh. English is also this school's
# default/most common language, so it's worth a cheap, reliable check of its
# own rather than leaving it to a statistical guess every time.
_ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "am", "a", "an", "how", "what",
    "when", "where", "why", "who", "which", "do", "does", "did", "can",
    "could", "would", "should", "will", "i", "you", "he", "she", "it",
    "we", "they", "my", "your", "his", "her", "our", "their", "to", "of",
    "in", "on", "at", "for", "with", "from", "and", "or", "but", "not",
    "please", "thanks", "thank", "hello", "hi", "hey", "yes", "no",
    "this", "that", "these", "those", "have", "has", "had", "need",
    "want", "get", "enroll", "enrollment", "school", "student", "grade",
}


def _looks_english(text):
    words = _tokenize(text)
    if not words:
        return False
    hits = sum(1 for w in words if w in _ENGLISH_MARKERS)
    return hits >= 1 and (hits / len(words)) >= 0.3


# ── langdetect CODE -> HUMAN-READABLE NAME ─────────────────────────
# Covers langdetect's full supported set (~55 languages) so anything it
# returns gets a proper name instead of being silently dropped.
_LANGDETECT_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "ms": "Malay",  # not natively in langdetect's set, kept for completeness
}


def _langdetect_fallback(text):
    """Confidence-scaled fallback for Latin-script languages langdetect
    actually covers well (Spanish, Italian, German, French, Vietnamese,
    Polish, Swedish, Turkish, Indonesian/Malay, and more). Shorter messages
    require higher confidence before we trust the guess over English."""
    if not _LANGDETECT_AVAILABLE or len(text) < 8:
        return None

    length = len(text)
    if length < 15:
        min_confidence = 0.97
    elif length < 30:
        min_confidence = 0.9
    elif length < 60:
        min_confidence = 0.75
    else:
        min_confidence = 0.6

    try:
        guesses = detect_langs(text)
    except LangDetectException:
        return None
    if not guesses:
        return None

    top = guesses[0]
    name = _LANGDETECT_NAMES.get(top.lang)
    if name and top.prob >= min_confidence:
        return {"code": top.lang, "name": name}
    return None


def detect_language(text):
    """Best-effort language detection for one chat message, covering Latin-
    script European/Southeast-Asian languages, Tagalog/Taglish, and a wide
    range of non-Latin scripts (Arabic, Cyrillic, Devanagari, Greek, Hebrew,
    Thai, Chinese/Japanese/Korean, and several South Asian scripts).

    Returns {"code": <iso-ish code>, "name": <human-readable name>}.
    Defaults to English for anything empty, too short to classify, or
    genuinely undetectable -- guessing wrong on a coin flip is worse than
    just falling back to the site's base language.
    """
    text = (text or "").strip()
    if not text:
        return {"code": "en", "name": "English"}

    # 1) Tagalog/Taglish -- most common non-English case for this bot, and
    #    the one a generic detector gets wrong most often.
    if _looks_tagalog(text):
        return {"code": "tl", "name": "Tagalog"}

    # 2) Non-Latin script -- the Unicode block itself is a strong,
    #    length-independent signal. For scripts shared by several languages,
    #    try to refine to the specific one; otherwise use the script default.
    script = _detect_script(text)
    if script:
        refined = _refine_within_script(script, text)
        if refined:
            return refined
        return _SCRIPT_DEFAULT[script]

    # 3) Common short greeting in another (Latin-script) language -- catches
    #    "Hola", "Bonjour", "Merhaba", etc. before anything below would
    #    otherwise default them to English.
    greeting = _check_greeting(text)
    if greeting:
        return greeting

    # 4) English quick check -- catches the majority case reliably before
    #    handing anything to langdetect, which struggles on short messages.
    if _looks_english(text):
        return {"code": "en", "name": "English"}

    # 5) Everything else -- Spanish, Malay, Italian, German, Vietnamese,
    #    Polish, Swedish, Turkish, and most other Latin-script languages.
    fallback = _langdetect_fallback(text)
    if fallback:
        return fallback

    return {"code": "en", "name": "English"}


def get_reply_language_instruction(text):
    """Ready-to-inject sentence for the LLM system prompt, telling it which
    language/dialect to answer the CURRENT message in."""
    lang = detect_language(text)

    if lang["code"] == "en":
        return "The user just wrote in English -- reply in English."

    if lang["code"] == "tl":
        return (
            "The user just wrote in Tagalog or Taglish -- reply in natural "
            "Tagalog or Taglish, matching how they wrote it."
        )

    return (
        f"The user just wrote in {lang['name']} -- reply fluently in "
        f"{lang['name']}."
    )