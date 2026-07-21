"""
scope_filter.py

Two jobs, kept separate from bot.py / app.py on purpose so they're easy to
find, tune, and unit-test on their own:

1. detect_language(text)      -- "tagalog" or "english", used so the bot's
                                   reply matches whatever language the user
                                   typed in.
2. is_probably_off_topic(text) -- a hard, deterministic backstop that blocks
                                   *obviously* off-topic requests (cooking,
                                   coding help, celebrity gossip, etc.)
                                   BEFORE the message ever reaches the LLM.

Why a separate hard filter at all, if QA_SYSTEM_PROMPT's rule 8 already
tells the model to refuse off-topic questions?
    - The prompt rule is the primary, general-purpose layer -- it catches
      almost everything, including things this keyword list doesn't know
      about.
    - But a system prompt is still just an instruction the model *chooses*
      to follow every single time. A cleverly-worded or role-play-style
      message can sometimes talk a model into ignoring it. A keyword check
      that runs in plain Python before the LLM call can't be talked out of
      anything -- if it matches, the canned refusal is returned no matter
      what the model would have said.
    - It also saves an LLM call entirely for the clearest cases (free,
      instant refusal instead of a paid generation).

This is intentionally conservative: it only blocks when the message matches
an off-topic pattern AND has no SHSWBES-related anchor word in it. Anything
ambiguous, mixed ("enrollment steps + can you also cook"), or not confidently
matched is passed through to the LLM, which still enforces rule 8 itself.
Tune the keyword lists below as real usage shows what needs adjusting.
"""

import re

# ── LANGUAGE DETECTION ──────────────────────────────────────────
# Common Tagalog/Taglish words and particles. If the message contains any of
# these (as whole words), we treat it as Tagalog; otherwise default English.
_TAGALOG_MARKERS = {
    "ang", "ng", "mga", "sa", "ako", "ikaw", "ka", "siya", "kami", "tayo",
    "kayo", "sila", "ito", "iyan", "iyon", "dito", "diyan", "doon",
    "paano", "bakit", "kailan", "saan", "sino", "ano", "alin", "ilan",
    "kamusta", "kumusta", "magandang", "salamat", "maraming", "po", "opo",
    "hindi", "oo", "wala", "meron", "mayroon", "gusto", "kailangan",
    "pwede", "puwede", "paki", "pakiusap", "tulungan", "tulong",
    "mag-enroll", "magparehistro", "paaralan", "eskwela", "eskuwela",
    "guro", "titser", "estudyante", "mag-aaral", "anak", "sige", "ayos",
    "tapos", "saka", "tsaka", "din", "rin", "lang", "naman", "ba",
    "yung", "yun", "nung", "pag", "kung", "para", "dahil", "kasi",
    "namin", "natin", "niya", "nila", "mo", "ko", "niyo", "ninyo",
}


def detect_language(text):
    """Returns 'tagalog' if the message contains recognizable Tagalog/Taglish
    words, otherwise 'english'. Simple word-overlap heuristic -- good enough
    for routing which canned string to use; not a full language classifier.
    """
    words = re.findall(r"[a-zA-Z\-']+", str(text or "").lower())
    if any(w in _TAGALOG_MARKERS for w in words):
        return "tagalog"
    return "english"


# ── SCOPE GUARD ──────────────────────────────────────────────────
# Anchor words that indicate the message IS about SHSWBES -- if any of these
# appear, we never hard-block, even if an off-topic word also appears (that
# case is left to rule 8/16 in the LLM prompt, which handles mixed messages
# properly instead of refusing the whole thing).
_IN_SCOPE_ANCHORS = {
    "enroll", "enrollment", "enrolment", "register", "registration",
    "registrar", "strand", "stem", "humss", "uniform", "id", "report card",
    "sf9", "sf10", "f137", "grade", "grading", "absence", "absences",
    "transferee", "transfer", "scholarship", "requirement", "requirements",
    "document", "documents", "certificate", "good moral", "graduation",
    "subject", "adviser", "advisor", "guidance", "canteen", "facility",
    "facilities", "school", "teacher", "faculty", "principal", "address",
    "contact", "schedule", "exam", "quarter", "semester", "dropping",
    "form", "application", "apply", "slot", "section", "deped",
    "bacoor", "shswbes", "senior high", "tuition", "fee", "class",
    "learner", "student", "classroom", "curriculum", "module", "modules",
    "e-beef", "beef form", "clearance", "org", "organization",
    "mag-enroll", "paaralan", "eskwela", "guro", "titser", "mag-aaral",
}

# Off-topic request patterns, grouped by domain just for readability. Each
# entry is matched as a whole phrase/word against the lowercased message.
_OFF_TOPIC_PATTERNS = [
    # Cooking / recipes
    r"\brecipe\b", r"\bcook(ing)?\b", r"\blutuin\b", r"\bsalad\b",
    r"\bboil(ed)? (an? )?egg\b", r"\bhow to cook\b",
    # Programming / tech support unrelated to the school
    r"\bwrite (me )?(a |some )?code\b", r"\bdebug (my|this)\b",
    r"\bpython script\b", r"\bhtml code\b", r"\bfix my code\b",
    # Entertainment / celebrities / media
    r"\bcelebrity\b", r"\bartista\b", r"\bmovie recommendation\b",
    r"\bsong lyrics\b", r"\btv series\b", r"\bshowbiz\b",
    # Sports scores / general trivia unrelated to school
    r"\bnba score\b", r"\bpba score\b", r"\bwho won the\b",
    r"\bhoroscope\b", r"\btell me a joke\b",
    # Weather / current events unrelated to school
    r"\bweather today\b", r"\bforecast\b",
    # Personal/relationship advice not about the school
    r"\bmy (boyfriend|girlfriend|jowa|crush)\b", r"\bbreak ?up\b",
    # Finance
    r"\bstock price\b", r"\bcrypto\b", r"\bbitcoin\b",
    # Generic translation/essay requests with no school anchor
    r"\btranslate this (sentence|paragraph)\b",
]
_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)

# Small talk / greetings / farewells the bot should always still answer
# warmly (per rules 1, 13, 15) -- never hard-blocked, no matter what.
_CASUAL_PATTERNS = [
    r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bkamusta\b", r"\bkumusta\b",
    r"\bmagandang (araw|umaga|hapon|gabi)\b", r"\bgood (morning|afternoon|evening)\b",
    r"\bthanks?\b", r"\bthank you\b", r"\bsalamat\b", r"\bok(ay)?\b",
    r"\bbye\b", r"\bgoodbye\b", r"\bpaalam\b", r"\bwala nang tanong\b",
    r"\bno more questions\b", r"\bthat'?s all\b", r"\bsige\b", r"\bayos\b",
]
_CASUAL_RE = re.compile("|".join(_CASUAL_PATTERNS), re.IGNORECASE)


def is_casual_or_greeting(text):
    """True for small talk / greetings / thanks / goodbyes -- these should
    never be hard-blocked even though they aren't "SHSWBES questions"."""
    return bool(_CASUAL_RE.search(str(text or "")))


def is_probably_off_topic(text):
    """Conservative hard block: True only when the message matches a known
    off-topic pattern AND contains no SHSWBES anchor word. Mixed or
    ambiguous messages are left for the LLM (rule 8/9/16) to handle."""
    lowered = str(text or "").lower()
    if not lowered.strip():
        return False
    if not _OFF_TOPIC_RE.search(lowered):
        return False
    if any(anchor in lowered for anchor in _IN_SCOPE_ANCHORS):
        return False
    return True


# ── CANNED REFUSAL (matches QA_SYSTEM_PROMPT rule 8 exactly) ────
_REFUSAL_EN = "I'm sorry but this system only answers within SHSWBES school inquiries."
_REFUSAL_TL = "Paumanhin, ang sistemang ito ay sumasagot lamang sa mga tanong tungkol sa SHSWBES."


def get_refusal_message(language):
    """language: 'tagalog' or 'english' (output of detect_language)."""
    return _REFUSAL_TL if language == "tagalog" else _REFUSAL_EN