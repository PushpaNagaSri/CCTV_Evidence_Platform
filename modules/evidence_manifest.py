
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "database" / "cctv_investigation.db"
OUTPUT_DIR = DATA_DIR / "evidence" / "manifests"


def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_file_path(path):
    """Resolve absolute or project-relative file paths."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


def find_source_video(video_name):
    """Find a video in the project's common video locations."""
    candidates = [
        DATA_DIR / "videos" / video_name,
        PROJECT_DIR / video_name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def create_evidence_manifest(video_name=None):
    """
    Export detection records with video/frame provenance
    and SHA-256 hashes for files that are available.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Investigation database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        query = """
            SELECT
                videos.id AS video_id,
                videos.video_name,
                videos.source_sha256 AS baseline_sha256,
                detections.id AS detection_id,
                detections.frame_path,
                detections.timestamp_seconds,
                detections.frame_index,
                detections.object_class,
                detections.confidence,
                detections.x1,
                detections.y1,
                detections.x2,
                detections.y2
            FROM videos
            JOIN detections
                ON videos.id = detections.video_id
        """

        parameters = ()

        if video_name:
            query += " WHERE videos.video_name = ?"
            parameters = (video_name,)

        query += """
            ORDER BY videos.video_name,
                     detections.timestamp_seconds
        """

        rows = connection.execute(
            query, parameters
        ).fetchall()

    finally:
        connection.close()

    if not rows:
        raise ValueError(
            "No detection records found for the requested video."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    video_hash_cache = {}

    for row in rows:
        record = dict(row)

        name = record["video_name"]

        # Calculate the current source video hash once per video.
        if name not in video_hash_cache:
            source_path = find_source_video(name)

            if source_path:
                video_hash_cache[name] = {
                    "path": str(source_path),
                    "current_sha256": calculate_sha256(source_path),
                    "status": "AVAILABLE",
                }
            else:
                video_hash_cache[name] = {
                    "path": None,
                    "current_sha256": None,
                    "status": "FILE_NOT_FOUND",
                }

        source_info = video_hash_cache[name]
        baseline_hash = record.pop("baseline_sha256")

        if source_info["current_sha256"] is None:
            integrity_status = "UNVERIFIABLE"
        elif baseline_hash == source_info["current_sha256"]:
            integrity_status = "MATCH"
        else:
            integrity_status = "MISMATCH"

        frame_path = resolve_file_path(record["frame_path"])

        if frame_path.is_file():
            frame_info = {
                "path": str(frame_path),
                "sha256": calculate_sha256(frame_path),
                "status": "AVAILABLE",
            }
        else:
            frame_info = {
                "path": str(frame_path),
                "sha256": None,
                "status": "FILE_NOT_FOUND",
            }

        manifest_records.append({
            "video": {
                "id": record["video_id"],
                "filename": name,
                "path": source_info["path"],
                "baseline_sha256": baseline_hash,
                "current_sha256": source_info["current_sha256"],
                "integrity_status": integrity_status,
            },
            "detection": {
                "id": record["detection_id"],
                "object_class": record["object_class"],
                "confidence": record["confidence"],
                "timestamp_seconds": record["timestamp_seconds"],
                "frame_index": record["frame_index"],
                "bounding_box_xyxy": {
                    "x1": record["x1"],
                    "y1": record["y1"],
                    "x2": record["x2"],
                    "y2": record["y2"],
                },
            },
            "supporting_frame": frame_info,
        })

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": (
            "AI-Powered CCTV Event Investigation and "
            "Digital Evidence Verification Platform"
        ),
        "record_count": len(manifest_records),
        "records": manifest_records,
        "limitations": [
            "A matching hash establishes byte-for-byte agreement "
            "with the saved baseline, not original authenticity.",
            "A manifest records available information; it does not "
            "by itself prove that evidence is admissible.",
            "Protect the manifest and baseline hashes against "
            "unauthorized modification.",
        ],
    }

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    if video_name:
        safe_name = Path(video_name).stem
        output_path = OUTPUT_DIR / (
            f"{safe_name}_manifest_{timestamp}.json"
        )
    else:
        output_path = OUTPUT_DIR / (
            f"all_videos_manifest_{timestamp}.json"
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print("\nEvidence manifest created successfully.")
    print(f"Records exported: {len(manifest_records)}")
    print(f"Output file: {output_path}")

    for status in ("MATCH", "MISMATCH", "UNVERIFIABLE"):
        count = sum(
            record["video"]["integrity_status"] == status
            for record in manifest_records
        )
        print(f"{status}: {count} detection records")

    return output_path


if __name__ == "__main__":
    print("CCTV Evidence Manifest Generator")
    print("1. Export all videos")
    print("2. Export one video")

    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        create_evidence_manifest()

    elif choice == "2":
        name = input(
            "Enter the exact video filename: "
        ).strip()

        if name:
            create_evidence_manifest(name)
        else:
            print("No filename entered.")

    else:
        print("Invalid choice.")
