
import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "database" / "cctv_investigation.db"


def get_videos():
    """Get CCTV video names from the database."""
    if not DB_PATH.exists():
        return []

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT video_name FROM videos ORDER BY video_name"
        ).fetchall()

    return [row[0] for row in rows]


def answer_question(question, video_name=None):
    """Answer investigation questions using database records."""
    if not DB_PATH.exists():
        return f"Database not found: {DB_PATH}"

    if not question or not question.strip():
        return "Please enter a question."

    q = question.strip().lower()

    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row

            videos = connection.execute(
                "SELECT id, video_name FROM videos ORDER BY video_name"
            ).fetchall()

            if not videos:
                return "No videos found in the database."

            video_filter = ""
            params = []

            if video_name and video_name != "All videos":
                video_filter = " AND v.video_name = ?"
                params.append(video_name)

            if any(term in q for term in (
                "list the available videos",
                "list videos",
                "available videos",
                "video names",
                "how many videos",
                "which videos"
            )):
                names = [
                    row["video_name"] for row in videos
                    if not video_name
                    or video_name == "All videos"
                    or row["video_name"] == video_name
                ]

                return "Available videos:\n" + "\n".join(
                    f"- {name}" for name in names
                )

            if any(term in q for term in (
                "summary", "overview", "total detections",
                "how many detections", "detection count",
                "what is in the video"
            )):
                rows = connection.execute(
                    """
                    SELECT v.video_name,
                           COUNT(d.id) AS total,
                           COUNT(DISTINCT d.object_class) AS classes
                    FROM videos v
                    LEFT JOIN detections d ON d.video_id = v.id
                    WHERE 1=1
                    """ + video_filter + """
                    GROUP BY v.id, v.video_name
                    ORDER BY v.video_name
                    """,
                    params
                ).fetchall()

                return "Investigation summary:\n" + "\n".join(
                    f"- {row['video_name']}: {row['total']} detections "
                    f"across {row['classes']} object classes"
                    for row in rows
                )

            if any(term in q for term in (
                "what objects", "which objects", "detected objects",
                "object classes", "classes detected"
            )):
                rows = connection.execute(
                    """
                    SELECT d.object_class, COUNT(*) AS total
                    FROM detections d
                    JOIN videos v ON v.id = d.video_id
                    WHERE 1=1
                    """ + video_filter + """
                    GROUP BY d.object_class
                    ORDER BY total DESC
                    """,
                    params
                ).fetchall()

                if not rows:
                    return "No detections found for this selection."

                return "Detected objects:\n" + "\n".join(
                    f"- {row['object_class']}: {row['total']} detections"
                    for row in rows
                )

            classes = connection.execute(
                "SELECT DISTINCT object_class FROM detections"
            ).fetchall()

            object_class = next(
                (
                    row["object_class"]
                    for row in sorted(
                        classes,
                        key=lambda item: len(item["object_class"]),
                        reverse=True
                    )
                    if row["object_class"].lower() in q
                ),
                None
            )

            if object_class:
                rows = connection.execute(
                    """
                    SELECT v.video_name, d.timestamp_seconds,
                           d.frame_index, d.confidence
                    FROM detections d
                    JOIN videos v ON v.id = d.video_id
                    WHERE LOWER(d.object_class) = LOWER(?)
                    """ + video_filter + """
                    ORDER BY d.timestamp_seconds
                    LIMIT 100
                    """,
                    [object_class] + params
                ).fetchall()

                count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM detections d
                    JOIN videos v ON v.id = d.video_id
                    WHERE LOWER(d.object_class) = LOWER(?)
                    """ + video_filter,
                    [object_class] + params
                ).fetchone()[0]

                if count == 0:
                    return f"No {object_class} detections found."

                lines = [
                    f"Found {count} detections of '{object_class}'.",
                    "Showing up to 100 records:"
                ]

                for row in rows:
                    lines.append(
                        f"- Video: {row['video_name']} | "
                        f"Time: {row['timestamp_seconds']:.2f}s | "
                        f"Frame: {row['frame_index']} | "
                        f"Confidence: {row['confidence']:.2f}"
                    )

                return "\n".join(lines)

            return (
                "Try asking:\n"
                "- List the available videos\n"
                "- Give me a summary of all videos\n"
                "- What objects were detected?\n"
                "- How many persons were detected?\n"
                "- Find cars in this video"
            )

    except sqlite3.Error as error:
        return f"Database query failed: {error}"
