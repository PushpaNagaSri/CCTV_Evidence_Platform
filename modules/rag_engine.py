import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "database" / "cctv_investigation.db"
RAG_DIR = DATA_DIR / "rag"
INDEX_PATH = RAG_DIR / "event_embeddings.npy"
RECORDS_PATH = RAG_DIR / "event_records.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_videos():
    if not DB_PATH.exists():
        return []

    with closing(sqlite3.connect(DB_PATH)) as connection:
        rows = connection.execute(
            "SELECT video_name FROM videos ORDER BY video_name"
        ).fetchall()

    return [row[0] for row in rows]


def answer_question(question, video_name=None):
    if not DB_PATH.exists():
        return f"Database not found: {DB_PATH}"

    if not question or not question.strip():
        return "Please enter a question."

    query = question.strip().lower()

    with closing(sqlite3.connect(DB_PATH)) as connection:
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

        if any(term in query for term in (
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

        if any(term in query for term in (
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

        if any(term in query for term in (
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
                if row["object_class"].lower() in query
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


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME, device="cpu")


def load_events():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute("""
            SELECT
                videos.video_name,
                videos.source_sha256,
                detections.frame_path,
                detections.timestamp_seconds,
                detections.frame_index,
                detections.object_class,
                detections.confidence,
                detections.x1,
                detections.y1,
                detections.x2,
                detections.y2
            FROM detections
            JOIN videos ON videos.id = detections.video_id
            ORDER BY videos.video_name, detections.timestamp_seconds
        """).fetchall()
    finally:
        connection.close()

    events = []

    for row in rows:
        event = dict(row)

        event["description"] = (
            f"CCTV video {event['video_name']}. "
            f"Detected object: {event['object_class']}. "
            f"Timestamp: {event['timestamp_seconds']:.2f} seconds. "
            f"Detection confidence: {event['confidence']:.2f}. "
            f"Frame number: {event['frame_index']}. "
            f"Bounding box coordinates: "
            f"{event['x1']}, {event['y1']}, "
            f"{event['x2']}, {event['y2']}."
        )

        events.append(event)

    return events


def build_event_index():
    events = load_events()

    if not events:
        raise ValueError(
            "No detections found. Process a video and import "
            "its detections into the database first."
        )

    RAG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_embedding_model()

    descriptions = [
        event["description"] for event in events
    ]

    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(INDEX_PATH, embeddings)

    with open(RECORDS_PATH, "w", encoding="utf-8") as file:
        json.dump(events, file, indent=2)

    print(f"Events indexed: {len(events)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved embeddings: {INDEX_PATH}")
    print(f"Saved event records: {RECORDS_PATH}")

    return len(events)




def search_events(question, top_k=5, video_name=None):
    if not INDEX_PATH.exists() or not RECORDS_PATH.exists():
        raise FileNotFoundError(
            "RAG index not found. Run build_event_index() first."
        )

    with open(RECORDS_PATH, "r", encoding="utf-8") as file:
        all_events = json.load(file)

    all_embeddings = np.load(INDEX_PATH)

    # Filter events by selected CCTV video before retrieval.
    if video_name and video_name != "All videos":
        selected_indices = [
            i for i, event in enumerate(all_events)
            if event["video_name"] == video_name
        ]

        all_events = [all_events[i] for i in selected_indices]
        all_embeddings = all_embeddings[selected_indices]

        if not all_events:
            return []

    if len(all_events) != len(all_embeddings):
        raise ValueError(
            "The event records and embeddings do not match. "
            "Rebuild the index."
        )

    question_lower = question.lower()
    words = set(question_lower.replace("-", " ").split())

    
    aliases = {
        "person": {
            "person", "people", "persons", "human", "humans",
            "pedestrian", "pedestrians"
        },
        "backpack": {
            "backpack", "backpacks", "rucksack", "rucksacks"
        },
        "bench": {"bench", "benches"},
        "chair": {"chair", "chairs"},
        "book": {"book", "books"},
        "potted plant": {
            "plant", "plants", "potted plant", "potted plants"
        },
        "car": {"car", "cars", "automobile", "automobiles"},
        "bicycle": {"bicycle", "bicycles", "bike", "bikes"},
        "motorcycle": {
            "motorcycle", "motorcycles", "motorbike", "motorbikes"
        },
        "bus": {"bus", "buses"},
        "truck": {"truck", "trucks"},
        "dog": {"dog", "dogs"},
        "cat": {"cat", "cats"},
        "bird": {"bird", "birds"},
    }


    # Detect explicitly requested object classes.
    requested_classes = set()

    for class_name, terms in aliases.items():
        if any(
            term in question_lower
            for term in terms
        ):
            requested_classes.add(class_name)

    # Match user-friendly names to actual detector class names.
    def class_matches(actual_class, requested_class):
        actual = actual_class.strip().lower()

        if requested_class == "potted plant":
            return actual in {"potted plant", "plant"}

        return actual == requested_class

    events = all_events
    embeddings = all_embeddings

    if requested_classes:
        matching_indices = [
            i for i, event in enumerate(all_events)
            if any(
                class_matches(
                    str(event["object_class"]),
                    requested_class
                )
                for requested_class in requested_classes
            )
        ]

        if not matching_indices:
            return []

        events = [all_events[i] for i in matching_indices]
        embeddings = all_embeddings[matching_indices]

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarities = embeddings @ question_embedding

    result_count = min(max(1, top_k), len(events))
    best_indices = np.argsort(similarities)[::-1][:result_count]

    results = []

    for index in best_indices:
        event = events[int(index)].copy()
        event["similarity"] = round(
            float(similarities[index]), 4
        )
        results.append(event)

    print("\nDEBUG - Retrieved object classes:")
    for event in results:
        print(repr(event["object_class"]), event["timestamp_seconds"])

    return results


def generate_investigation_summary(question, results):
    if not results:
        return (
            "No indexed CCTV records were retrieved for this query. "
            "This does not prove that the event never occurred."
        )

    lines = [
        "INVESTIGATION SUMMARY",
        "",
        f"Question: {question}",
        f"Retrieved records: {len(results)}",
        "",
        "FINDINGS"
    ]

    for number, event in enumerate(results, start=1):
        timestamp = float(event["timestamp_seconds"])
        confidence = float(event["confidence"])

        lines.append(
            f"{number}. A {event['object_class']} detection was "
            f"recorded at {timestamp:.2f} seconds in "
            f"{event['video_name']}. "
            f"Detection confidence: {confidence:.2f}."
        )

        lines.append(
            f"   Supporting frame: {event['frame_path']}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "These findings describe the retrieved detection records. "
        "Detection confidence is the detector's confidence score, "
        "not the probability that an event occurred.",
        "",
        "LIMITATIONS",
        "Semantic retrieval may omit relevant records. The results "
        "are not necessarily an exhaustive list of every occurrence. "
        "Review the supporting frames and use structured searches "
        "when exact object classes or time ranges are required."
    ])

    return "\n".join(lines)



if __name__ == "__main__":
    print("Building CCTV event knowledge base...")
    build_event_index()

    print("\nTesting semantic event search...")

    query = input(
        "\nEnter your investigation question: "
    ).strip()

    if query:
        results = search_events(query)

        print("\nRetrieved CCTV events:")

        for number, event in enumerate(results, start=1):
            print(f"\nResult {number}")
            print(f"Video: {event['video_name']}")
            print(f"Object: {event['object_class']}")
            print(
                f"Timestamp: "
                f"{event['timestamp_seconds']:.2f} seconds"
            )
            print(f"Confidence: {event['confidence']:.2f}")
            print(f"Frame: {event['frame_path']}")
            print(f"Similarity: {event['similarity']}")
            print(f"Description: {event['description']}")