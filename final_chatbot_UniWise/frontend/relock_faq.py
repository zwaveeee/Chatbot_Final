"""
relock_faq.py

Run this ONE command after you intentionally edit faq.txt on purpose:

    python relock_faq.py

It updates faq.lock to fingerprint your new faq.txt content as the new
trusted baseline. Do this every time -- and only when -- you (the
developer/admin) deliberately change faq.txt's content. If app.py ever
reports a faq.txt mismatch that you did NOT cause, do NOT just run this
script to silence it -- investigate first, since that warning exists
specifically to catch unexpected/unauthorized changes.
"""

import os
from faq_lock import _hash_file, _lock_path_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FAQ_FILE = os.path.join(PROJECT_ROOT, "faq.txt")


def main():
    if not os.path.exists(FAQ_FILE):
        print(f"faq.txt not found at {FAQ_FILE} -- nothing to lock.")
        return

    lock_path = _lock_path_for(FAQ_FILE)
    new_hash = _hash_file(FAQ_FILE)

    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(new_hash)

    print(f"faq.lock updated. Current faq.txt content is now the trusted baseline.")
    print(f"Lock file: {lock_path}")


if __name__ == "__main__":
    main()