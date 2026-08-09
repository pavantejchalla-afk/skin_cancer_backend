import sqlite3
import csv
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

DB_PATH = Path("database.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Dataset metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lesion_id TEXT,
            image_id TEXT UNIQUE,
            dx TEXT,
            dx_type TEXT,
            age REAL,
            sex TEXT,
            localization TEXT
        )
    """)

    # 2. Scans history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_records (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            filename TEXT,
            predicted_label TEXT,
            predicted_class TEXT,
            confidence REAL,
            risk_level TEXT,
            scanned_at TEXT
        )
    """)

    conn.commit()

    # Populate HAM10000 dataset if table is empty
    cursor.execute("SELECT COUNT(*) FROM dataset_metadata")
    count = cursor.fetchone()[0]
    if count == 0:
        csv_path = Path("HAM10000_metadata.csv")
        if csv_path.exists():
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                to_db = [
                    (
                        row["lesion_id"],
                        row["image_id"],
                        row["dx"],
                        row["dx_type"],
                        float(row["age"]) if row["age"] else None,
                        row["sex"],
                        row["localization"],
                    )
                    for row in reader
                ]
                cursor.executemany("""
                    INSERT OR IGNORE INTO dataset_metadata (lesion_id, image_id, dx, dx_type, age, sex, localization)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, to_db)
                conn.commit()
                logger.info("Loaded %d HAM10000 dataset records into SQLite database.", len(to_db))

    conn.close()