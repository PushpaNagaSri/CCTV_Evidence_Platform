
import json
import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

EVIDENCE_DIR = PROJECT_DIR / "data" / "evidence"
OUTPUT_DIR = PROJECT_DIR / "data" / "detections"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25


def detect_objects(record_path):
    record_path = Path(record_path)

    with open(record_path, "r", encoding="utf-8") as file:
        video_record = json.load(file)

    video_name = video_record["video_name"]
    video_stem = Path(video_name).stem

    model = YOLO(MODEL_NAME)

    video_output_dir = OUTPUT_DIR / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []

    for frame_info in video_record["extracted_frames"]:
        frame_path = PROJECT_DIR / frame_info["frame"]

        if not frame_path.exists():
            print("Frame not found:", frame_path)
            continue

        results = model.predict(
            source=str(frame_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device="cpu"
        )

        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [
                round(value)
                for value in box.xyxy[0].tolist()
            ]

            all_detections.append({
                "video_name": video_name,
                "frame": frame_info["frame"],
                "timestamp_seconds": frame_info["timestamp_seconds"],
                "frame_index": frame_info["frame_index"],
                "object_class": model.names[class_id],
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

        annotated_frame = result.plot()
        annotated_path = video_output_dir / Path(frame_path).name
        cv2.imwrite(str(annotated_path), annotated_frame)

        print(
            f"Processed frame at "
            f"{frame_info['timestamp_seconds']} seconds"
        )

    output_record = {
        "video_name": video_name,
        "source_sha256": video_record["sha256"],
        "model": MODEL_NAME,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "frame_count": len(video_record["extracted_frames"]),
        "detection_count": len(all_detections),
        "detections": all_detections
    }

    output_path = EVIDENCE_DIR / f"{video_stem}_detections.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_record, file, indent=4)

    print("\nOBJECT DETECTION COMPLETED")
    print("--------------------------")
    print("Video:", video_name)
    print("Frames processed:", len(video_record["extracted_frames"]))
    print("Objects detected:", len(all_detections))
    print("Detection results:", output_path)
    print("Annotated frames:", video_output_dir)

    return output_record



if __name__ == "__main__":
    # Process only the three personal CCTV evidence records.
    record_files = sorted(EVIDENCE_DIR.glob("cctv_*_record.json"))

    if not record_files:
        print("No personal CCTV evidence records found.")
        print("Run modules/video_processor.py first.")
    else:
        print(f"Found {len(record_files)} personal CCTV video record(s).")

        for record_path in record_files:
            print("\n" + "=" * 55)
            print("Processing evidence record:", record_path.name)
            print("=" * 55)

            try:
                output_record = detect_objects(record_path)

                print("\nDetection summary")
                print("Video:", output_record["video_name"])
                print("Frames processed:", output_record["frame_count"])
                print("Objects detected:", output_record["detection_count"])

            except Exception as error:
                print("Detection failed for:", record_path.name)
                print("Reason:", error)

        print("\nFinished processing personal CCTV videos.")
