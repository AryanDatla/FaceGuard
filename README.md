# **🛡️ FaceGuard: Biometric Security & Attendance System**

FaceGuard is a professional-grade biometric employee monitoring and attendance system. It leverages state-of-the-art computer vision and deep learning to provide real-time face recognition, multi-layered anti-spoofing protection, and automated attendance logging via a sleek Streamlit interface.

[![Python 3.12](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://scikit-learn.org/stable/)
[![Streamlit Badge](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![DeepFace](https://img.shields.io/badge/-DeepFace-FF69B4)](https://github.com/serengil/deepface)

------

## **🚀 Key Features**

### 1. Security-First Live Scanner

**Real-Time Recognition:** High-resolution face detection and matching using the Facenet model and Cosine distance metrics.

**Multi-Layered Anti-Spoofing:** Protects against presentation attacks (photos, videos, or masks) using:

**DeepFace MiniFASNet:** Neural-network-based liveness detection.

**LBP Texture Analysis:** Detects "flatness" in printed photos vs. real skin texture.

**FFT Frequency Check:** Identifies loss of high-frequency detail common in digital screens.

**Geometry Gate:** Rejects extreme-angle or profile-only attacks.

**Audio Feedback:** Integrated Text-to-Speech (TTS) for "Access Granted" or "Spoof Detected" alerts.



### **2. Admin Command Center**

**Live Metrics:** At-a-glance view of total staff, present employees, security alerts, and spoofing attempts.

**Data Management:** Searchable dataframes for attendance logs, security events, and liveness audit trails.

**Late Reporting:** Automatically flags arrivals after a configurable cutoff time (e.g., 09:00 AM).


### **3. Smart Enrollment Portal**

**1:N Duplicate Prevention:** Automatically checks new enrollments against the existing database to prevent double-entry or identity theft.

**Flexible Capture:** Supports both live webcam snapshots and high-quality image uploads.

**Validation:** Built-in checks to ensure exactly one face is present and clearly visible before saving.

----

## **🏗️ System Architecture**

FaceGuard is designed with a decoupled architecture that separates the presentation layer from the core biometric processing engine.

### **1. Presentation Layer (Streamlit)**

**Admin Dashboard:** Manages state handling for authentication and view switching.

**Live Scanner UI:** A dedicated high-performance view for real-time video processing and visual feedback.

**Analytics Engine:** Processes CSV data into visual summaries and searchable reports.


### **2. Logic & Security Engine (att.py)**

**Recognition Module:** Orchestrates DeepFace models (Facenet) for 128-d feature extraction and cosine similarity matching.

**Anti-Spoofing Pipeline:** A multi-layered verification stack (DeepFace MiniFASNet, LBP Texture Analysis, and FFT Frequency Checks) that must all return a "Real" status before recognition is attempted.

**Attendance Manager:** Handles the logic for cooldown periods, late-arrival flagging, and duplicate enrollment prevention.


### **3. Data Persistence Layer**

**Biometric Vault:** A structured directory (employee_db/) storing reference face templates.

**Relational Logs:** A series of flat-file CSVs acting as a lightweight database for attendance, security alerts, and spoofing audits.

**Security Archive:** Automatic capture and storage of "Intruder" and "Spoof" attempts for forensic review.

----

## **🔄 Operational Flow**

The system follows a synchronous processing pipeline for every frame captured during the "Live Scanner" session to ensure maximum security.

### **Phase 1: Pre-Processing & Detection**

**Frame Acquisition:** OpenCV captures a raw frame from the video stream.

**Face Localization:** The system attempts to locate a face using the configured detector (OpenCV/MTCNN). If no face is found, the frame is discarded immediately.


### **Phase 2: The Security Gate (Anti-Spoofing)**

Before identity is even checked, the face must pass through three distinct filters:

**Neural Check:** MiniFASNet analyzes the face for "liveness" patterns.

**Texture Check:** Local Binary Patterns (LBP) detect if the surface has the mathematical texture of skin or paper.

**Frequency Check:** Fast Fourier Transform (FFT) looks for the high-frequency "screen door" effect found in digital displays.


### ***Phase 3: Identity Matching**

**Feature Extraction:** If the face is deemed "Real," a mathematical embedding is generated.

**Vector Comparison:** This embedding is compared against the employee_db using Cosine Similarity.

**Threshold Validation:** If the distance is below the THRESHOLD (e.g., 0.40), a match is confirmed.


### **Phase 4: Action & Feedback**

**Validation Check:** The system verifies if the user is in a "Cooldown" state.

**Logging:** Data is appended to attendance.csv with a "Late" or "On-Time" flag based on the system clock.

**User Interface:** The UI displays a green success box, and the Text-to-Speech (TTS) engine announces "Access Granted."

**Failure Protocol:** If recognition fails or a spoof is detected, a snapshot is saved to the intruder_captures folder, and a security alert is logged.

----

## **🛠️ Tech Stack**

**Frontend:** Streamlit

**Face Recognition:** DeepFace (Facenet model)

**Computer Vision:** OpenCV

**Backend:** Python (Modular architecture)

**Data Storage:** CSV-based logging for easy portability and auditability.

----

## **📦 Installation & Setup**

**1. Prerequisite:** Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.

**2. Install Dependencies**
```
Bash
pip install streamlit opencv-python pandas deepface tf-keras python-dotenv pyttsx3
```

**3. Environment Configuration**
Create a .env file in the root directory to secure your administrative credentials:
```
Code snippet
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password
```

**4. Directory Structure**
The system will automatically create these folders on first run, but ensure the structure looks like this:
```
FaceGuard
├── app.py
├── att.py
├── employee_db
├── intruder_captures
└── attendance.csv
```

----

## **🖥️ Usage**

Launch the Application:
```
Bash
streamlit run app.py
```
**Live Scanner:** Open the "Live Scanner" tab to begin monitoring. This is a public-facing entrance screen.

**Admin Login:** Use the sidebar to log in. Once authenticated, the Command Center, Enrollment, and Analytics tabs will be unlocked.

**Enrolling Staff:** 
- Navigate to Employee Enrollment.
- Enter an ID and Name.
- Capture or upload a photo. The system will run a 1:N check to ensure the person isn't already in the database.

----

## **⚙️ Configuration Tuning**

You can adjust system sensitivity in att.py:

**THRESHOLD:** Adjust the recognition distance (Default: 0.40). Lower is stricter.

**COOLDOWN_SEC:** Time between allowing the same person to mark attendance again (Default: 30s).

**LATE_AFTER:** The time after which arrivals are marked as "Late" (Default: "09:00").

**ANTI_SPOOFING_ENABLED:** Toggle the liveness engine on/off.

## **🔒 Security Note**

This system captures and stores biometric data locally. Ensure the employee_db and log files are stored on an encrypted drive or within a secured network environment to comply with local data privacy regulations.
