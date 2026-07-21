"""
faq_lock.py

Guarantees that faq.txt only ever changes through a deliberate action by
whoever runs this project (the developer/admin editing the file and then
running relock_faq.py) -- never silently, and never as a side effect of
anything a chatbot user does through the website.

Why this exists on top of the fact that app.py never opens faq.txt in write
mode: "the code doesn't write to it" is true today, but it's not something
you can *see* happen or fail loudly if it's ever violated (a future route
added by mistake, a bug, direct file-system tampering, etc). This module
makes that guarantee observable instead of implicit:

  - The trusted content of faq.txt is fingerprinted (SHA-256) into a small
    faq.lock file sitting next to it.
  - Every time the app starts, it re-hashes the live faq.txt and compares it
    against that locked fingerprint.
  - First run ever (no faq.lock yet): the current faq.txt becomes the
    trusted baseline automatically.
  - Any run after that where the hash doesn't match: faq.txt was changed
    since it was last locked. The app prints a loud, impossible-to-miss
    warning identifying exactly that -- it does NOT block the server from
    starting (so a legitimate content update by you doesn't accidentally
    take the whole site down), but it does make an unexpected change
    impossible to miss in the console/logs.
  - After you intentionally edit faq.txt yourself, run `python relock_faq.py`
    once to update the trusted fingerprint to match your new content -- that
    is the ONLY sanctioned way the lock should ever move.
"""

import hashlib
import os


def _hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _lock_path_for(faq_file_path):
    return os.path.join(os.path.dirname(os.path.abspath(faq_file_path)), "faq.lock")


def verify_or_establish_lock(faq_file_path):
    """Call once at app startup, right after locating faq.txt.

    Returns a small status dict:
        {"status": "locked_first_time" | "ok" | "MISMATCH", "lock_path": ...}

    Never raises for a mismatch -- the caller decides whether to just log it
    (default) or treat it as fatal.
    """
    if not os.path.exists(faq_file_path):
        return {"status": "missing_faq_file", "lock_path": None}

    lock_path = _lock_path_for(faq_file_path)
    current_hash = _hash_file(faq_file_path)

    if not os.path.exists(lock_path):
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(current_hash)
        return {"status": "locked_first_time", "lock_path": lock_path}

    with open(lock_path, "r", encoding="utf-8") as f:
        trusted_hash = f.read().strip()

    if current_hash == trusted_hash:
        return {"status": "ok", "lock_path": lock_path}

    return {"status": "MISMATCH", "lock_path": lock_path}


def print_lock_result(result, faq_file_path):
    """Human-readable console output for the result of verify_or_establish_lock."""
    status = result.get("status")

    if status == "missing_faq_file":
        print(f"[faq_lock] WARNING: faq.txt not found at {faq_file_path} -- nothing to verify.")
        return

    if status == "locked_first_time":
        print(f"[faq_lock] No faq.lock found -- current faq.txt content has been "
              f"fingerprinted and locked as the trusted baseline ({result['lock_path']}).")
        return

    if status == "ok":
        print("[faq_lock] faq.txt integrity check passed -- content matches the locked fingerprint.")
        return

    if status == "MISMATCH":
        print("=" * 70)
        print("[faq_lock] WARNING: faq.txt does NOT match its locked fingerprint!")
        print("This means faq.txt's content changed since it was last locked.")
        print("If YOU intentionally edited faq.txt, run: python relock_faq.py")
        print("If you did NOT edit it, treat this as a possible integrity issue")
        print("and check what changed before trusting the chatbot's answers.")
        print(f"Lock file: {result['lock_path']}")
        print("=" * 70)