import os
import sqlite3
import csv
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

DB_PATH = Path("database.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "")


def is_postgres():
    return DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")


def get_db_connection():
    if is_postgres():
        import psycopg2
        import psycopg2.extras
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        # Add sslmode=require if not present for Railway/cloud PostgreSQL
        if "sslmode" not in url.lower():
            url += "?sslmode=require" if "?" not in url else "&sslmode=require"
        conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.DictCursor, connect_timeout=5)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if is_postgres():
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    id SERIAL PRIMARY KEY,
                    lesion_id VARCHAR(100),
                    image_id VARCHAR(100) UNIQUE,
                    dx VARCHAR(50),
                    dx_type VARCHAR(50),
                    age REAL,
                    sex VARCHAR(20),
                    localization VARCHAR(100)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_records (
                    id VARCHAR(100) PRIMARY KEY,
                    user_id VARCHAR(100),
                    filename VARCHAR(255),
                    predicted_label VARCHAR(50),
                    predicted_class VARCHAR(100),
                    confidence REAL,
                    risk_level VARCHAR(20),
                    scanned_at VARCHAR(100)
                );
            """)
            conn.commit()

            cursor.execute("SELECT COUNT(*) FROM dataset_metadata;")
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
                        import psycopg2.extras
                        psycopg2.extras.execute_values(
                            cursor,
                            """
                            INSERT INTO dataset_metadata (lesion_id, image_id, dx, dx_type, age, sex, localization)
                            VALUES %s ON CONFLICT (image_id) DO NOTHING;
                            """,
                            to_db
                        )
                        conn.commit()
                        logger.info("Loaded %d HAM10000 dataset records into PostgreSQL database.", len(to_db))

        else:
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
    except Exception as err:
        logger.warning("Database init info: %s", err)