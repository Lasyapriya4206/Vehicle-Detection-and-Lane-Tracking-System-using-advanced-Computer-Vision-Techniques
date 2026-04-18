🚗 Vehicle Detection and Lane Tracking System

A Streamlit-based real-time traffic monitoring system using YOLOv8 for vehicle detection, lane identification, lane change detection, speed estimation, and traffic density analysis.

📌 Project Overview

This project analyzes traffic videos and performs:

Vehicle detection (car, motorcycle, bus, truck)
Lane detection using predefined lane lines
Lane change detection with event logging
Speed estimation for each vehicle
Traffic density classification
Real-time visualization using Streamlit
CSV log and processed video export

It is useful for intelligent traffic monitoring, road safety studies, and smart city development.

🧠 Key Features
✔ Vehicle Detection

Powered by YOLOv8 (yolov8n.pt) for lightweight and fast detection.

✔ Lane Identification

Three lanes separated at:

Lane 1: < 450px
Lane 2: 450–900px
Lane 3: > 900px
✔ Lane Change Detection

System identifies transitions like:
1 → 2, 2 → 3, 3 → 1
and logs them with timestamps.

✔ Speed Estimation

Speed is estimated in pixels per second based on positional change.

✔ Traffic Density Levels
LOW
MEDIUM
HIGH
✔ Output Downloads
Processed video
Full CSV log (vehicle counts + lane change events)
🛠️ Installation
1️⃣ Install Python Libraries

Run this command:

pip install ultralytics opencv-python numpy pandas streamlit

If needed, also install:

pip install pillow
2️⃣ Make sure your model file is available

Download yolov8n.pt and keep it in the same folder as your project code.

▶️ How to Run the Project

Run the Streamlit application:

streamlit run app.py

Then:

Upload a traffic video (mp4 / avi / mov / mpeg4).
Click Start Analysis.
View:
Bounding boxes
Lane numbers
Speed
Lane change count
Traffic density
Download:
CSV log
Processed output video

📂 Project Structure Example
project-folder/
│── app.py
│── yolov8n.pt
│── README.md
│── sample_videos/
│── requirements.txt

📊 Output Files Explained
📘 traffic_analysis_log.csv

Contains:

Frame timestamps
Total vehicle count
Lane-wise vehicle count
Traffic density
Lane change events
Vehicle ID with lane transitions

🎥 processed_output.mp4
Video showing:

Bounding boxes
Lane lines
Vehicle ID + Lane + Speed
Traffic statistics
📘 Important Files in Code
process_frame()

Handles detection, tracking, lane assignment, and speed estimation.

estimate_speed()

Calculates speed per vehicle.

get_lane()

Decides which lane the vehicle is in.

draw_ui()

Displays text stats on video.

🚀 Future Enhancements
Real-world speed calculation (km/h)
Accident or sudden braking detection
Heatmap-based congestion prediction
Dynamic lane configuration
Vehicle classification accuracy improvements
