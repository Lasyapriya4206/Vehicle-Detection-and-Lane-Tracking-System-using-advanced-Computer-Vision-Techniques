import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import streamlit as st
import tempfile
import pandas as pd
import time
from ultralytics import YOLO

# ────────────────────────────────────────
# Load Model
# ────────────────────────────────────────
model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ────────────────────────────────────────
# Lane Configuration
# ────────────────────────────────────────
LANE_1 = 450
LANE_2 = 900

# ────────────────────────────────────────
# Persistent state
# ────────────────────────────────────────
previous_lane = {}
previous_time = {}
previous_position = {}
unique_vehicles = set()
lane_cross_events = []
last_event_for_vehicle = {}

lane_change_count = 0

# ────────────────────────────────────────
# Reset per-frame counters
# ────────────────────────────────────────
def reset_frame_state():
    return {
        "vehicle_count": 0,
        "lane_count": {1: 0, 2: 0, 3: 0},
    }

# ────────────────────────────────────────
# Get lane number
# ────────────────────────────────────────
def get_lane(center_x):
    if center_x < LANE_1:
        return 1
    elif center_x < LANE_2:
        return 2
    else:
        return 3

# ────────────────────────────────────────
# Speed estimation
# ────────────────────────────────────────
def estimate_speed(track_id, center_y):
    if track_id not in previous_position:
        previous_position[track_id] = center_y
        previous_time[track_id] = time.time()
        return 0

    distance = abs(center_y - previous_position[track_id])
    dt = time.time() - previous_time[track_id]

    previous_position[track_id] = center_y
    previous_time[track_id] = time.time()

    if dt < 0.001:
        return 0

    return int((distance / dt) * 0.05)

# ────────────────────────────────────────
# Traffic density
# ────────────────────────────────────────
def traffic_density(count):
    if count < 5:
        return "LOW"
    if count < 15:
        return "MEDIUM"
    return "HIGH"

# ────────────────────────────────────────
# Process one frame
# ────────────────────────────────────────
def process_frame(frame):
    global lane_change_count

    results = model.track(frame, persist=True, verbose=False)
    state = reset_frame_state()

    if results[0].boxes.id is None:
        density = traffic_density(0)
        draw_ui(frame, state["vehicle_count"], density, state["lane_count"])
        return frame, state["vehicle_count"], density, state["lane_count"]

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, track_id, cls in zip(boxes, ids, classes):
        if cls not in VEHICLE_CLASSES:
            continue

        unique_vehicles.add(track_id)
        state["vehicle_count"] += 1

        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        lane = get_lane(center_x)

        prev_lane = previous_lane.get(track_id, lane)
        if prev_lane != lane:
            transition = f"{prev_lane}->{lane}"
            if last_event_for_vehicle.get(track_id) != transition:
                lane_cross_events.append({
                    "time": time.time(),
                    "vehicle_id": track_id,
                    "from_lane": prev_lane,
                    "to_lane": lane
                })
                last_event_for_vehicle[track_id] = transition
                lane_change_count += 1

        previous_lane[track_id] = lane
        state["lane_count"][lane] += 1

        speed = estimate_speed(track_id, center_y)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"ID:{track_id} L{lane} {speed} px/s"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.line(frame, (LANE_1, 0), (LANE_1, frame.shape[0]), (255, 255, 0), 3)
    cv2.line(frame, (LANE_2, 0), (LANE_2, frame.shape[0]), (255, 255, 0), 3)

    density = traffic_density(state["vehicle_count"])
    draw_ui(frame, state["vehicle_count"], density, state["lane_count"])

    return frame, state["vehicle_count"], density, state["lane_count"]

def draw_ui(frame, total_vehicles, density, lane_count):
    y = 40
    cv2.putText(frame, f"Vehicles: {total_vehicles}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    y += 40

    cv2.putText(frame, f"L1: {lane_count[1]}   L2: {lane_count[2]}   L3: {lane_count[3]}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
    y += 40

    cv2.putText(frame, f"Lane Changes: {lane_change_count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    y += 40

    cv2.putText(frame, f"Traffic: {density}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

# ────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Vehicle Detection and Lane Tracking System")
st.title("Vehicle Detection and Lane Tracking System")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov", "mpeg4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    if st.button("▶️ Start Analysis"):
        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("Cannot open video file.")
        else:
            frame_placeholder = st.empty()
            log_data = []

            lane_change_count = 0
            unique_vehicles.clear()
            lane_cross_events.clear()
            previous_lane.clear()
            previous_time.clear()
            previous_position.clear()
            last_event_for_vehicle.clear()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 20.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, veh_count, density, lane_cnt = process_frame(frame)

                log_data.append({
                    "timestamp": time.time(),
                    "total_vehicles": veh_count,
                    "lane1": lane_cnt[1],
                    "lane2": lane_cnt[2],
                    "lane3": lane_cnt[3],
                    "density": density
                })

                frame_placeholder.image(processed_frame, channels="BGR")
                out.write(processed_frame)

                time.sleep(0.03)

            cap.release()
            out.release()

            st.success(f"✅ Analysis finished — Total unique vehicles: **{len(unique_vehicles)}**")

            df_logs = pd.DataFrame(log_data)
            df_events = pd.DataFrame(lane_cross_events)

            combined = pd.concat([
                df_logs.assign(record_type="frame_log"),
                df_events.assign(record_type="lane_change")
            ], ignore_index=True)

            csv_bytes = combined.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Full CSV Log",
                data=csv_bytes,
                file_name="traffic_analysis_log.csv",
                mime="text/csv"
            )

            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.download_button(
                label="🎥 Download Output Video",
                data=video_bytes,
                file_name="processed_output.mp4",
                mime="video/mp4"
            )

        try:
            os.unlink(tfile.name)
        except:
            pass