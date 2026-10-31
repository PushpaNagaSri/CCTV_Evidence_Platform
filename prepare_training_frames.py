
import json
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

RECORD_PATH = (
    PROJECT_DIR / "data" / "evidence" / "cctv_1_record.json"
)

OUTPUT_DIR = (
    PROJECT_DIR / "data" / "custom_dataset" / "raw_frames"
)
DATASET_DIR = PROJECT_DIR / "data" / "custom_dataset"

for folder in [
    DATASET_DIR / "raw_frames",
    DATASET_DIR / "images" / "train",
    DATASET_DIR / "images" / "val",
    DATASET_DIR / "labels" / "train",
    DATASET_DIR / "labels" / "val",
]:
    folder.mkdir(parents=True, exist_ok=True)
# Select one frame out of every three extracted frames.
FRAME_INTERVAL = 3


def main():
    if not RECORD_PATH.exists():
        print("CCTV evidence record not found:", RECORD_PATH)
        return

    with open(RECORD_PATH, "r", encoding="utf-8") as file:
        record = json.load(file)

    frames = record.get("extracted_frames", [])

    if not frames:
        print("No extracted frames found in the record.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    for index, frame_info in enumerate(frames):
        if index % FRAME_INTERVAL != 0:
            continue

        source_path = PROJECT_DIR / frame_info["frame"]

        if not source_path.exists():
            print("Missing frame:", source_path)
            missing += 1
            continue

        timestamp = frame_info["timestamp_seconds"]

        output_name = (
            f"classroom_{timestamp:.2f}s"
            .replace(".", "_") + ".jpg"
        )

        destination = OUTPUT_DIR / output_name

        shutil.copy2(source_path, destination)
        copied += 1

    print("\nFrame preparation completed.")
    print("Total extracted frames:", len(frames))
    print("Frames copied:", copied)
    print("Missing frames:", missing)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
