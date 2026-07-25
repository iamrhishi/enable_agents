#!/usr/bin/env python3
"""
Rotate SETTINGS_ENCRYPTION_KEY: decrypts every UserSettingModel and
ProjectSettingModel row with the OLD key and re-encrypts it with the NEW
key, in a single DB transaction (all-or-nothing - if anything fails to
decrypt, nothing is written).

This exists because core/crypto.py has no rotation story on its own:
changing SETTINGS_ENCRYPTION_KEY without doing this first makes every
previously-saved API key permanently undecryptable (users would see their
keys silently stop working, with no error pointing at why).

Usage:
    OLD_ENCRYPTION_KEY=<current key> NEW_ENCRYPTION_KEY=<new key> \\
        python scripts/rotate_encryption_key.py [--dry-run]

Generate a new key with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

After a successful (non-dry-run) run, update SETTINGS_ENCRYPTION_KEY in
the environment to NEW_ENCRYPTION_KEY and restart the backend.
"""

import argparse
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from cryptography.fernet import Fernet, InvalidToken

from core.database import db
from core.models import ProjectSettingModel, UserSettingModel


def rotate(old_key: str, new_key: str, dry_run: bool) -> None:
    old_fernet = Fernet(old_key.encode())
    new_fernet = Fernet(new_key.encode())

    rotated = 0
    failed = []

    for model in (UserSettingModel, ProjectSettingModel):
        rows = model.query.all()
        for row in rows:
            try:
                plaintext = old_fernet.decrypt(row.value_encrypted.encode()).decode()
            except InvalidToken:
                failed.append(f"{model.__tablename__}#{row.id}")
                continue
            new_ciphertext = new_fernet.encrypt(plaintext.encode()).decode()
            if not dry_run:
                row.value_encrypted = new_ciphertext
            rotated += 1

    if failed:
        print(f"ABORTING: {len(failed)} row(s) could not be decrypted with OLD_ENCRYPTION_KEY: {failed}")
        print("Nothing was written. Check that OLD_ENCRYPTION_KEY is actually the key currently in use.")
        db.session.rollback()
        sys.exit(1)

    if dry_run:
        print(f"Dry run OK: {rotated} row(s) would be rotated. Re-run without --dry-run to apply.")
        db.session.rollback()
    else:
        db.session.commit()
        print(f"Rotated {rotated} row(s). Now update SETTINGS_ENCRYPTION_KEY to NEW_ENCRYPTION_KEY and restart the backend.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Decrypt/re-encrypt in memory only, don't write anything.")
    args = parser.parse_args()

    old_key = os.environ.get("OLD_ENCRYPTION_KEY")
    new_key = os.environ.get("NEW_ENCRYPTION_KEY")
    if not old_key or not new_key:
        print("Set both OLD_ENCRYPTION_KEY and NEW_ENCRYPTION_KEY environment variables.")
        sys.exit(1)
    if old_key == new_key:
        print("OLD_ENCRYPTION_KEY and NEW_ENCRYPTION_KEY are the same - nothing to do.")
        sys.exit(1)

    from app import app
    with app.app_context():
        rotate(old_key, new_key, args.dry_run)


if __name__ == "__main__":
    main()
