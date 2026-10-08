
import cv2
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_DIR = PROJECT_DIR / "data" / "videos"
FRAME_DIR = PROJECT_DIR / "data" / "frames"
EVIDENCE_DIR = PROJECT_DIR / "data" / "evidence"

for folder in [VIDEO_DIR, FRAME_DIR, EVIDENCE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


def process_video(video_path, interval_seconds=2):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open this video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("Video metadata is invalid or unavailable.")

    duration = total_frames / fps
    step = max(1, round(fps * interval_seconds))

    video_frame_dir = FRAME_DIR / video_path.stem
    video_frame_dir.mkdir(parents=True, exist_ok=True)

    extracted_frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % step == 0:
            timestamp = frame_index / fps
            frame_name = f"frame_{frame_index:08d}.jpg"
            frame_path = video_frame_dir / frame_name

            if cv2.imwrite(str(frame_path), frame):
                extracted_frames.append({
                    "frame": str(frame_path.relative_to(PROJECT_DIR)),
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp, 3)
                })

        frame_index += 1

    cap.release()

    original_hash = calculate_sha256(video_path)

    evidence_record = {
        "video_name": video_path.name,
        "video_path": str(video_path.resolve()),
        "sha256": original_hash,
        "hash_algorithm": "SHA-256",
        "duration_seconds": round(duration, 3),
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "extraction_interval_seconds": interval_seconds,
        "extracted_frame_count": len(extracted_frames),
        "extracted_frames": extracted_frames,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "processing_status": "completed"
    }

    record_path = EVIDENCE_DIR / f"{video_path.stem}_record.json"

    with open(record_path, "w", encoding="utf-8") as file:
        json.dump(evidence_record, file, indent=4)

    return evidence_record, record_path


if __name__ == "__main__":
    videos = sorted(
        path for path in VIDEO_DIR.iterdir()
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )

    if not videos:
        print("No videos found. Put a CCTV video in data/videos/")
    else:
        try:
            record, record_path = process_video(videos[0])

            print("\nVIDEO PROCESSING COMPLETED")
            print("-------------------------")
            print("Video:", record["video_name"])
            print("Duration:", record["duration_seconds"], "seconds")
            print("FPS:", record["fps"])
            print("Resolution:", record["resolution"])
            print("Frames extracted:", record["extracted_frame_count"])
            print("SHA-256:", record["sha256"])
            print("Evidence record:", record_path)

        except Exception as error:
            print("Processing failed:", error)
