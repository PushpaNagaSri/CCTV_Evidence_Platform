import json
import sqlite3
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



def search_events(question, top_k=5):
    if not INDEX_PATH.exists() or not RECORDS_PATH.exists():
        raise FileNotFoundError(
            "RAG index not found. Run build_event_index() first."
        )

    with open(RECORDS_PATH, "r", encoding="utf-8") as file:
        events = json.load(file)

    embeddings = np.load(INDEX_PATH)

    if len(events) != len(embeddings):
        raise ValueError(
            "The event records and embeddings do not match. "
            "Rebuild the index."
        )

    # Map common words to detector class names.
    aliases = {
        "person": ["person", "people", "human", "pedestrian"],
        "car": ["car"],
        "bicycle": ["bicycle", "bike"],
        "motorcycle": ["motorcycle", "motorbike"],
        "bus": ["bus"],
        "truck": ["truck"],
        "dog": ["dog"],
        "cat": ["cat"],
        "bird": ["bird"],
    }

    question_lower = question.lower()
    detected_classes = {
        str(event["object_class"]).lower()
        for event in events
    }

    # Identify whether the question explicitly asks about
    # a known object category.
    requested_class = None

    for class_name, words in aliases.items():
        if any(
            word in question_lower.split()
            for word in words
        ):
            requested_class = class_name
            break

    # If the requested class is absent, return no matches.
    if requested_class is not None:
        matching_indices = [
            i for i, event in enumerate(events)
            if str(event["object_class"]).lower()
            == requested_class
        ]

        if not matching_indices:
            return []

        events = [events[i] for i in matching_indices]
        embeddings = embeddings[matching_indices]

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

