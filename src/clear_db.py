# clear_chroma.py

import shutil
import os

DB_PATH = "db/chroma_db"

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("✅ Chroma database deleted.")
else:
    print("Database not found.")
