
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


from modules.rag_engine import (
    search_events,
    generate_investigation_summary
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
EVIDENCE_DIR = DATA_DIR / "evidence"
DB_PATH = DATA_DIR / "database" / "cctv_investigation.db"
AUDIT_FILE = DATA_DIR / "audit_logs" / "integrity_checks.jsonl"


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CCTV Investigation Platform",
    page_icon="🔎",
    layout="wide"
)

st.title("AI-Powered CCTV Event Investigation")
st.caption(
    "Computer Vision | Event Search | Digital Evidence Integrity"
)


# --------------------------------------------------
# SHA-256 HASH CALCULATION
# --------------------------------------------------

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# INTEGRITY AUDIT LOGGING
# --------------------------------------------------

def log_integrity_check(
    record_path,
    checked_file,
    original_sha256,
    current_sha256,
    status
):
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_record": str(record_path),
        "checked_file": str(checked_file),
        "original_sha256": original_sha256,
        "current_sha256": current_sha256,
        "status": status
    }

    with open(AUDIT_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")


# --------------------------------------------------
# LOAD EVIDENCE RECORDS
# --------------------------------------------------

def load_records():
    records = []

    for record_path in EVIDENCE_DIR.glob("*_record.json"):
        try:
            with open(record_path, "r", encoding="utf-8") as file:
                record = json.load(file)

            record["_record_path"] = str(record_path)
            records.append(record)

        except (OSError, json.JSONDecodeError):
            continue

    return records


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_database_connection():
    if not DB_PATH.exists():
        return None

    return sqlite3.connect(DB_PATH)


records = load_records()


tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Event Investigation",
    "Evidence Verification",
    "AI Investigation Assistant"
])



# ==================================================
# TAB 1: DASHBOARD
# ==================================================

with tab1:
    st.header("Investigation Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Evidence Records", len(records))

    total_frames = sum(
        record.get(
            "extracted_frame_count",
            len(record.get("extracted_frames", []))
        )
        for record in records
    )

    col2.metric("Extracted Frames", total_frames)

    connection = get_database_connection()

    if connection:
        try:
            video_count = pd.read_sql_query(
                "SELECT COUNT(*) AS total FROM videos",
                connection
            ).iloc[0]["total"]

            detection_count = pd.read_sql_query(
                "SELECT COUNT(*) AS total FROM detections",
                connection
            ).iloc[0]["total"]

            col3.metric("Stored Detections", int(detection_count))

            st.subheader("Database Summary")
            st.write(
                f"Videos registered in database: {int(video_count)}"
            )

        except Exception as error:
            st.error(f"Could not read database tables: {error}")

        finally:
            connection.close()

    else:
        st.warning(
            "Database not found. Run your database manager module first."
        )

    st.subheader("Registered CCTV Evidence")

    if records:
        summary = []

        for record in records:
            summary.append({
                "Video": record.get("video_name", "Unknown"),
                "Duration (seconds)": record.get(
                    "duration_seconds", "N/A"
                ),
                "FPS": record.get("fps", "N/A"),
                "Frames extracted": record.get(
                    "extracted_frame_count",
                    len(record.get("extracted_frames", []))
                ),
                "Resolution": record.get("resolution", "N/A"),
                "Baseline SHA-256": record.get("sha256", "N/A")
            })

        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No evidence records found in data/evidence.")


# ==================================================
# TAB 2: EVENT INVESTIGATION
# ==================================================

with tab2:
    st.header("Search CCTV Events")

    connection = get_database_connection()

    if not connection:
        st.warning(
            "Database not found. Run the database manager first."
        )

    else:
        try:
            columns = pd.read_sql_query(
                "PRAGMA table_info(detections)",
                connection
            )["name"].tolist()

            if not columns:
                st.warning(
                    "The detections table was not found. "
                    "Import detections into the database first."
                )

            else:
                detection_data = pd.read_sql_query(
                    "SELECT * FROM detections LIMIT 1",
                    connection
                )

                if detection_data.empty:
                    st.info("No detections have been imported yet.")

                else:
                    st.caption("Available database fields:")
                    st.write(", ".join(columns))

                    class_col = next(
                        (
                            name for name in [
                                "object_class",
                                "class_name",
                                "label",
                                "class"
                            ]
                            if name in columns
                        ),
                        None
                    )

                    time_col = next(
                        (
                            name for name in [
                                "timestamp_seconds",
                                "timestamp",
                                "time_seconds"
                            ]
                            if name in columns
                        ),
                        None
                    )

                    if not class_col or not time_col:
                        st.warning(
                            "Your database uses different column names. "
                            "Check the displayed column names."
                        )

                    else:
                        class_values = pd.read_sql_query(
                            f'SELECT DISTINCT "{class_col}" '
                            f'FROM detections ORDER BY "{class_col}"',
                            connection
                        ).iloc[:, 0].dropna().tolist()

                        if not class_values:
                            st.info("No detected object classes are available.")

                        else:
                            selected_class = st.selectbox(
                                "Object to search for",
                                class_values
                            )

                            min_time, max_time = pd.read_sql_query(
                                f'SELECT MIN("{time_col}"), '
                                f'MAX("{time_col}") FROM detections',
                                connection
                            ).iloc[0]

                            min_time = float(min_time or 0)
                            max_time = float(max_time or 0)

                            slider_max = max(
                                max_time,
                                min_time + 1.0
                            )

                            start_time, end_time = st.slider(
                                "Timestamp range (seconds)",
                                min_value=min_time,
                                max_value=slider_max,
                                value=(min_time, slider_max)
                            )

                            if st.button("Search Events"):
                                results = pd.read_sql_query(
                                    f'SELECT * FROM detections '
                                    f'WHERE "{class_col}" = ? '
                                    f'AND "{time_col}" BETWEEN ? AND ? '
                                    f'ORDER BY "{time_col}"',
                                    connection,
                                    params=(
                                        selected_class,
                                        start_time,
                                        end_time
                                    )
                                )

                                st.subheader(
                                    f"Search Results: {len(results)} detections"
                                )

                                if results.empty:
                                    st.info(
                                        "No detections found for this search."
                                    )

                                else:
                                    st.dataframe(
                                        results,
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                    for _, row in results.iterrows():
                                        original_relative_path = row.get(
                                            "frame_path"
                                        )

                                        if not isinstance(
                                            original_relative_path, str
                                        ):
                                            st.warning(
                                                "Original frame path is missing."
                                            )
                                            continue

                                        original_path = Path(
                                            original_relative_path
                                        )

                                        if not original_path.is_absolute():
                                            original_path = (
                                                PROJECT_DIR / original_path
                                            )

                                        video_row = connection.execute(
                                            "SELECT video_name FROM videos "
                                            "WHERE id = ?",
                                            (int(row["video_id"]),)
                                        ).fetchone()

                                        if not video_row:
                                            st.warning(
                                                "Video record not found for "
                                                "this detection."
                                            )
                                            continue

                                        video_stem = Path(
                                            video_row[0]
                                        ).stem

                                        annotated_path = (
                                            DATA_DIR / "detections"
                                            / video_stem
                                            / original_path.name
                                        )

                                        st.divider()

                                        st.markdown(
                                            f"**{str(row['object_class']).title()}** "
                                            f"at {float(row['timestamp_seconds']):.2f} "
                                            "seconds"
                                        )

                                        st.write(
                                            f"Confidence: "
                                            f"{float(row['confidence']):.2%}"
                                        )

                                        st.write(
                                            f"Bounding box: "
                                            f"({row['x1']}, {row['y1']}) "
                                            f"to ({row['x2']}, {row['y2']})"
                                        )

                                        left, right = st.columns(2)

                                        with left:
                                            st.markdown(
                                                "**Original CCTV frame**"
                                            )

                                            if original_path.exists():
                                                st.image(
                                                    str(original_path),
                                                    use_container_width=True
                                                )
                                            else:
                                                st.warning(
                                                    "Original frame file not found."
                                                )

                                        with right:
                                            st.markdown(
                                                "**YOLO-annotated frame**"
                                            )

                                            if annotated_path.exists():
                                                st.image(
                                                    str(annotated_path),
                                                    use_container_width=True
                                                )
                                            else:
                                                st.warning(
                                                    "Annotated frame not found: "
                                                    f"{annotated_path}"
                                                )

        except Exception as error:
            st.error(f"Event search error: {error}")

        finally:
            connection.close()


# ==================================================
# TAB 3: DIGITAL EVIDENCE VERIFICATION
# ==================================================

with tab3:
    st.header("Digital Evidence Integrity Verification")

    if not records:
        st.warning("No evidence records are available.")

    else:
        record_names = [
            Path(record["_record_path"]).name
            for record in records
        ]

        selected_name = st.selectbox(
            "Select evidence record",
            record_names
        )

        selected_record = records[
            record_names.index(selected_name)
        ]

        record_path = Path(selected_record["_record_path"])
        original_path = VIDEO_DIR / selected_record["video_name"]
        baseline_hash = selected_record.get("sha256", "")

        st.write(
            "**Recorded video:**",
            selected_record["video_name"]
        )

        st.write("**Saved baseline SHA-256:**")
        st.code(baseline_hash)

        # ------------------------------------------
        # VERIFY ORIGINAL VIDEO
        # ------------------------------------------

        if st.button("Verify Original CCTV Video"):
            if not original_path.exists():
                status = "FILE_NOT_FOUND"
                current_hash = None

                log_integrity_check(
                    record_path,
                    original_path,
                    baseline_hash,
                    current_hash,
                    status
                )

                st.error("Original video file not found.")
                st.info("Verification attempt saved to the audit log.")

            else:
                current_hash = calculate_sha256(original_path)

                status = (
                    "MATCH"
                    if current_hash == baseline_hash
                    else "MISMATCH"
                )

                log_integrity_check(
                    record_path,
                    original_path,
                    baseline_hash,
                    current_hash,
                    status
                )

                st.write("**Current SHA-256:**")
                st.code(current_hash)

                if status == "MATCH":
                    st.success(
                        "MATCH — file matches the saved baseline."
                    )
                else:
                    st.error(
                        "MISMATCH — file differs from the saved baseline."
                    )

                st.info("Verification attempt saved to the audit log.")

        st.divider()
        st.subheader("Test Another File Against This Baseline")

        test_path_text = st.text_input(
            "Test file path",
            value="data/videos/test_copy.mp4"
        )

        # ------------------------------------------
        # VERIFY TEST FILE
        # ------------------------------------------

        if st.button("Verify Test File"):
            test_path = Path(test_path_text)

            if not test_path.is_absolute():
                test_path = PROJECT_DIR / test_path

            if not test_path.exists():
                status = "FILE_NOT_FOUND"
                test_hash = None

                log_integrity_check(
                    record_path,
                    test_path,
                    baseline_hash,
                    test_hash,
                    status
                )

                st.error(f"File not found: {test_path}")
                st.info("Verification attempt saved to the audit log.")

            else:
                test_hash = calculate_sha256(test_path)

                status = (
                    "MATCH"
                    if test_hash == baseline_hash
                    else "MISMATCH"
                )

                log_integrity_check(
                    record_path,
                    test_path,
                    baseline_hash,
                    test_hash,
                    status
                )

                st.write("**Test file SHA-256:**")
                st.code(test_hash)

                if status == "MATCH":
                    st.success(
                        "MATCH — test file matches the saved baseline."
                    )
                else:
                    st.error(
                        "MISMATCH — test file differs from the saved baseline."
                    )

                st.info("Verification attempt saved to the audit log.")

        # ------------------------------------------
        # VIEW RECORDED AUDIT LOG
        # ------------------------------------------

        st.divider()
        st.subheader("Integrity Verification Audit Log")

        
        if AUDIT_FILE.exists():
            try:
                with open(AUDIT_FILE, "r", encoding="utf-8") as file:
                    audit_entries = [
                        json.loads(line)
                        for line in file
                        if line.strip()
                    ]

                if audit_entries:
                    audit_df = pd.DataFrame(audit_entries)

                    display_columns = {
                        "checked_at_utc": "Verification Time (UTC)",
                        "checked_file": "File Checked",
                        "original_sha256": "Recorded SHA-256",
                        "current_sha256": "Calculated SHA-256",
                        "status": "Verification Status"
                    }

                    available_columns = [
                        column
                        for column in display_columns
                        if column in audit_df.columns
                    ]

                    audit_df = audit_df[available_columns].rename(
                        columns=display_columns
                    )

                    st.dataframe(
                        audit_df.iloc[::-1],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("The audit log is currently empty.")

            except (OSError, json.JSONDecodeError) as error:
                st.warning(f"The audit log could not be read: {error}")
        else:
            st.info(
                "No audit log exists yet. Verify a file to create one."
            )

  


with tab4:
    st.header("AI Investigation Assistant")

    st.write(
        "Ask questions about objects detected in your CCTV footage. "
        "The system retrieves relevant events and supporting evidence."
    )

    question = st.text_input(
        "Enter your investigation question",
        placeholder="When does a person appear in the CCTV footage?"
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Search CCTV Events"):
        if not question.strip():
            st.warning("Please enter an investigation question.")
        else:
            try:
                with st.spinner("Searching CCTV event records..."):
                    results = search_events(question, top_k=top_k)

                if not results:
                    st.info("No matching events were retrieved.")
                else:
                    summary = generate_investigation_summary(question, results)

                    st.subheader("Investigation Summary")
                    st.text(summary)

                    st.download_button(
                        label="Download Investigation Summary",
                        data=summary,
                        file_name="investigation_summary.txt",
                        mime="text/plain"
                    )

                    st.divider()

                    st.subheader("Retrieved Evidence")
                    st.caption(
                        "Results are ranked by semantic similarity. "
                        "Similarity is not the same as detection confidence."
                    )

                    for number, event in enumerate(results, start=1):
                        with st.expander(
                            f"Result {number}: "
                            f"{event['object_class']} at "
                            f"{event['timestamp_seconds']:.2f} seconds"
                        ):
                            st.write(
                                f"**Video:** {event['video_name']}"
                            )
                            st.write(
                                f"**Object:** {event['object_class']}"
                            )
                            st.write(
                                f"**Timestamp:** "
                                f"{event['timestamp_seconds']:.2f} seconds"
                            )
                            st.write(
                                f"**Detection confidence:** "
                                f"{event['confidence']:.2f}"
                            )
                            st.write(
                                f"**Semantic similarity:** "
                                f"{event['similarity']:.4f}"
                            )
                            st.write(
                                f"**Frame number:** {event['frame_index']}"
                            )

                            frame_path = Path(event["frame_path"])

                            if not frame_path.is_absolute():
                                frame_path = PROJECT_DIR / frame_path

                            if frame_path.exists():
                                
                                st.subheader("Visual Evidence")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.image(
                                        str(frame_path),
                                        caption="Original Frame",
                                        use_container_width=True
                                    )

                                with col2:
                                    annotated_path = (
                                        PROJECT_DIR
                                        / "data"
                                        / "detections"
                                        / Path(event["frame_path"]).parent.name
                                        / Path(event["frame_path"]).name
                                    )

                                    if annotated_path.exists():
                                        st.image(
                                            str(annotated_path),
                                            caption="YOLO Detection Frame",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("Annotated frame not found.")

                            else:
                                st.warning(
                                    f"Frame image not found: {frame_path}"
                                )

            except Exception as error:
                st.error(f"Investigation search failed: {error}")
          



# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Research prototype: a hash mismatch indicates a file difference, "
    "not necessarily malicious tampering or proof of authenticity."
)
