
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_DIR / "data" / "evidence"
VIDEO_DIR = PROJECT_DIR / "data" / "videos"
AUDIT_FILE = PROJECT_DIR / "data" / "audit_logs" / "integrity_checks.jsonl"


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_video(record_path, video_path=None):
    with open(record_path, "r", encoding="utf-8") as file:
        record = json.load(file)

    original_hash = record["sha256"]
    video_name = record["video_name"]

    if video_path is None:
        video_path = VIDEO_DIR / video_name
    else:
        video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = PROJECT_DIR / video_path

    if not video_path.exists():
        status = "FILE_NOT_FOUND"
        current_hash = None
    else:
        current_hash = calculate_sha256(video_path)
        status = "MATCH" if current_hash == original_hash else "MISMATCH"

    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_record": str(record_path),
        "checked_file": str(video_path),
        "original_sha256": original_hash,
        "current_sha256": current_hash,
        "status": status
    }

    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(AUDIT_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")

    print(f"File checked: {video_path.name}")
    print(f"Status: {status}")

    if status == "MATCH":
        print("The file matches the saved baseline hash.")
    elif status == "MISMATCH":
        print("The file differs from the saved baseline hash.")
    else:
        print("The file could not be found.")

    return result


if __name__ == "__main__":
    records = list(EVIDENCE_DIR.glob("*_record.json"))

    if not records:
        print("No evidence records found.")
    else:
        print("Available evidence records:")
        for index, record in enumerate(records, start=1):
            print(f"{index}. {record.name}")

        choice = int(input("Select a record number: "))
        selected_record = records[choice - 1]

        print("\n1. Verify the original video")
        print("2. Verify a test copy")
        option = input("Choose an option: ").strip()

        if option == "1":
            verify_video(selected_record)
        elif option == "2":
            copy_path = input(
                "Enter the test copy path (e.g. data/videos/test_copy.mp4): "
            ).strip()
            verify_video(selected_record, copy_path)
        else:
            print("Invalid option.")
