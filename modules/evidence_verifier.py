
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_DIR / "data" / "evidence"
VIDEO_DIR = PROJECT_DIR / "data" / "videos"
AUDIT_DIR = PROJECT_DIR / "data" / "audit_logs"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_video(record_path):
    with open(record_path, "r", encoding="utf-8") as file:
        record = json.load(file)

    video_path = VIDEO_DIR / record["video_name"]

    if not video_path.exists():
        result = {
            "video_name": record["video_name"],
            "status": "FILE_NOT_FOUND",
            "message": "The original video could not be found."
        }
    else:
        original_hash = record["sha256"]
        current_hash = calculate_sha256(video_path)
        matches = original_hash == current_hash

        result = {
            "video_name": record["video_name"],
            "recorded_sha256": original_hash,
            "current_sha256": current_hash,
            "status": "MATCH" if matches else "MISMATCH",
            "message": (
                "The file matches the recorded hash."
                if matches
                else "The file differs from the recorded baseline."
            )
        }

    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()

    log_path = AUDIT_DIR / "integrity_checks.jsonl"

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")

    print("\nEVIDENCE INTEGRITY CHECK")
    print("------------------------")
    print("Video:", result["video_name"])
    print("Status:", result["status"])
    print("Result:", result["message"])
    print("Audit log:", log_path)

    return result


if __name__ == "__main__":
    record_files = sorted(
        path for path in EVIDENCE_DIR.glob("*_record.json")
    )

    if not record_files:
        print("No original evidence record found.")
    else:
        verify_video(record_files[0])
