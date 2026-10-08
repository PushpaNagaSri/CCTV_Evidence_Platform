
import json
import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

EVIDENCE_DIR = PROJECT_DIR / "data" / "evidence"
DATABASE_DIR = PROJECT_DIR / "data" / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATABASE_DIR / "cctv_investigation.db"


def connect_db():
    return sqlite3.connect(DB_PATH)


def create_tables():
    with connect_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL UNIQUE,
                source_sha256 TEXT,
                model_name TEXT,
                total_detections INTEGER DEFAULT 0
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_path TEXT,
                timestamp_seconds REAL NOT NULL,
                frame_index INTEGER,
                object_class TEXT NOT NULL,
                confidence REAL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_detection_search
            ON detections(object_class, timestamp_seconds)
        """)

    print("Database and tables are ready.")


def import_detection_file(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        record = json.load(file)

    with connect_db() as connection:
        connection.execute("""
            INSERT INTO videos (
                video_name, source_sha256, model_name, total_detections
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(video_name) DO UPDATE SET
                source_sha256 = excluded.source_sha256,
                model_name = excluded.model_name,
                total_detections = excluded.total_detections
        """, (
            record["video_name"],
            record.get("source_sha256"),
            record.get("model"),
            record.get("detection_count", len(record["detections"]))
        ))

        video_row = connection.execute(
            "SELECT id FROM videos WHERE video_name = ?",
            (record["video_name"],)
        ).fetchone()

        video_id = video_row[0]

        connection.execute(
            "DELETE FROM detections WHERE video_id = ?",
            (video_id,)
        )

        for item in record["detections"]:
            box = item["bounding_box"]

            connection.execute("""
                INSERT INTO detections (
                    video_id, frame_path, timestamp_seconds, frame_index,
                    object_class, confidence, x1, y1, x2, y2
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                item["frame"],
                item["timestamp_seconds"],
                item["frame_index"],
                item["object_class"],
                item["confidence"],
                box["x1"],
                box["y1"],
                box["x2"],
                box["y2"]
            ))

    print("Imported detections from:", Path(json_path).name)


def import_all_detection_files():
    files = sorted(EVIDENCE_DIR.glob("*_detections.json"))

    if not files:
        print("No detection JSON files found. Run detector.py first.")
        return

    create_tables()

    for json_path in files:
        import_detection_file(json_path)

    print("All detection files imported.")


if __name__ == "__main__":
    import_all_detection_files()
