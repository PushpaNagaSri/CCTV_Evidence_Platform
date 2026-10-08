
import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "database" / "cctv_investigation.db"


def search_events(object_class=None, start_time=0, end_time=None):
    if not DB_PATH.exists():
        print("Database not found. Run database_manager.py first.")
        return

    if end_time is None:
        end_time = float("inf")

    query = """
        SELECT
            v.video_name,
            d.object_class,
            d.timestamp_seconds,
            d.confidence,
            d.frame_path,
            d.x1, d.y1, d.x2, d.y2
        FROM detections d
        JOIN videos v ON v.id = d.video_id
        WHERE d.timestamp_seconds >= ?
          AND d.timestamp_seconds <= ?
    """

    parameters = [start_time, end_time]

    if object_class:
        query += " AND LOWER(d.object_class) = LOWER(?)"
        parameters.append(object_class.strip())

    query += " ORDER BY d.timestamp_seconds ASC"

    with sqlite3.connect(DB_PATH) as connection:
        results = connection.execute(query, parameters).fetchall()

    if not results:
        print("No matching detections found.")
        return

    print(f"\nFound {len(results)} matching detections")
    print("-" * 70)

    for row in results:
        video, label, timestamp, confidence, frame, x1, y1, x2, y2 = row

        print(f"Video: {video}")
        print(f"Object: {label}")
        print(f"Timestamp: {timestamp:.2f} seconds")
        print(f"Confidence: {confidence:.2f}")
        print(f"Frame: {frame}")
        print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
        print("-" * 70)


if __name__ == "__main__":
    print("CCTV Event Search")
    print("Object classes: person, car, bus, truck, bicycle, etc.")

    selected_class = input(
        "Object class (press Enter for all objects): "
    ).strip()

    start = float(input("Start time in seconds: "))
    end = float(input("End time in seconds: "))

    search_events(
        object_class=selected_class or None,
        start_time=start,
        end_time=end
    )
