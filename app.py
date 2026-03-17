import streamlit as st
import cv2
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
import att

# Load environment variables from .env file
load_dotenv() 

# --- Page Configuration ---
st.set_page_config(
    page_title="FaceGuard",
    page_icon="🛡️",
    layout="wide"
)

# --- Secure Admin Credentials ---
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin") 
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "default_password")

# --- Initialize Session State ---
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = None

# --- Global Data Initialization ---
att.ensure_csvs()
all_employees = att.get_enrolled_employees()
today_str = datetime.now().strftime("%Y-%m-%d")

# --- Sidebar Navigation ---
st.sidebar.title("🛡️ FaceGuard")
st.sidebar.markdown("---")

# Navigation Options
menu_options = ["Live Scanner", "Command Center", "Employee Enrollment", "Analytics & Reports", "System Registry"]
menu = st.sidebar.selectbox("Navigation", menu_options)

# --- Admin Login Section in Sidebar ---
st.sidebar.markdown("---")
if not st.session_state.admin_logged_in:
    st.sidebar.subheader("Admin Login")
    user = st.sidebar.text_input("Username")
    passwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if user == ADMIN_USERNAME and passwd == ADMIN_PASSWORD:
            st.session_state.admin_logged_in = True
            st.sidebar.success("Logged in as Admin")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")
else:
    st.sidebar.success("Admin Mode Active")
    if st.sidebar.button("Logout"):
        st.session_state.admin_logged_in = False
        st.rerun()

# --- Access Control Logic ---
restricted_tabs = ["Command Center", "Employee Enrollment", "Analytics & Reports", "System Registry"]

if menu in restricted_tabs and not st.session_state.admin_logged_in:
    st.title("🔒 Access Restricted")
    st.warning(f"The '{menu}' tab requires Administrative privileges. Please login via the sidebar.")
    st.stop()

# --- 1. Live Scanner (Public Access) ---
if menu == "Live Scanner":
    st.title("🛡️ Security Entrance Scanner")
    
    # Initialize scanner state
    if 'scanner_active' not in st.session_state:
        st.session_state.scanner_active = False

    # Layout: Bigger central column for the camera feed
    c1, c2, c3 = st.columns([0.1, 8, 0.1])
    
    with c2:
        # Toggle scanner state with buttons
        if not st.session_state.scanner_active:
            if st.button("🚀 Start Scanner", width="stretch", type="primary"):
                st.session_state.scanner_active = True
                st.rerun()
        else:
            if st.button("🛑 Stop Scanner", width="stretch", type="secondary"):
                st.session_state.scanner_active = False
                st.rerun()

        # Larger window container
        FRAME_WINDOW = st.image([], width="stretch") 
        
    if st.session_state.scanner_active:
        worker = att.RecognitionWorker()
        cap = cv2.VideoCapture(0)
        
        # Set higher resolution if hardware supports it
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while st.session_state.scanner_active:
            ret, frame = cap.read()
            if not ret: 
                st.error("Failed to access webcam.")
                break

            frame = cv2.flip(frame, 1)
            worker.submit(frame)
            
            latest_status = worker.get_result()
            att.draw_hud(frame, len(all_employees))
            att.draw_status_bar(frame, latest_status)

            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, width="stretch")
            
        cap.release()
    else:
        st.info("Scanner is currently offline. Click 'Start Scanner' to begin monitoring.")

# --- 2. Command Center (Admin Only) ---
elif menu == "Command Center":
    st.title("System Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Total Staff", len(all_employees))
    
    with m2:
        if os.path.exists(att.ATTENDANCE_CSV):
            df_att = pd.read_csv(open(att.ATTENDANCE_CSV, encoding='utf-8', errors='replace'))
            today_data = df_att[df_att['Date'] == today_str]
            unique_present = today_data['Employee ID'].nunique()
            st.metric("Present Today", unique_present)
        else:
            st.metric("Present Today", 0)
            
    with m3:
        if os.path.exists(att.LOG_CSV):
            df_log = pd.read_csv(open(att.LOG_CSV, encoding='utf-8', errors='replace'))
            denied = len(df_log[(df_log['Date'] == today_str) & (df_log['Result'] == "DENIED")])
            st.metric("Security Alerts", denied)
        else:
            st.metric("Security Alerts", 0)

    with m4:
        if os.path.exists(att.SPOOF_LOG_CSV):
            df_spoof = pd.read_csv(open(att.SPOOF_LOG_CSV, encoding='utf-8', errors='replace'))
            spoof_count = len(df_spoof[df_spoof['Date'] == today_str])
            st.metric("⚠ Spoof Attempts", spoof_count)
        else:
            st.metric("⚠ Spoof Attempts", 0)
            
    with m5: st.metric("System Status", "Operational")

# --- 3. Employee Enrollment (Admin Only) ---
elif menu == "Employee Enrollment":
    st.title("Enrollment Portal")

    e_id   = st.text_input("Employee ID")
    e_name = st.text_input("Full Name")
    mode   = st.selectbox("Method", ["Webcam Capture", "Upload Image"])

    # File uploader must live outside st.form — render it conditionally here
    uploaded_file = None
    if mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"],
            help="Upload a clear, front-facing photo. One face only."
        )

    if st.button("Start Enrollment", type="primary"):
        if not e_id or not e_name:
            st.warning("Please fill in both Employee ID and Full Name.")
        else:
            import tempfile, shutil
            temp_path = os.path.join(tempfile.gettempdir(), f"temp_{e_id}.jpg")

            # ── Webcam path ───────────────────────────────────────────────────
            if mode == "Webcam Capture":
                att._enroll_from_webcam(temp_path, e_name)

            # ── Upload path ───────────────────────────────────────────────────
            elif mode == "Upload Image":
                if uploaded_file is None:
                    st.warning("Please upload an image first.")
                    st.stop()
                # Save uploaded bytes → temp JPEG
                import numpy as np
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    st.error("Could not decode image. Please upload a valid JPG or PNG.")
                    st.stop()
                # Validate exactly one face is present
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(faces) == 0:
                    st.error("❌ No face detected in the uploaded image. Please use a clear, front-facing photo.")
                    st.stop()
                if len(faces) > 1:
                    st.error(f"❌ {len(faces)} faces detected. Please upload an image with exactly one person.")
                    st.stop()
                cv2.imwrite(temp_path, img)
                # Show preview
                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    caption="Uploaded photo — 1 face detected ✓",
                    width=200
                )

            # ── Duplicate check + save ────────────────────────────────────────
            if os.path.exists(temp_path):
                with st.spinner("Checking for existing enrollment..."):
                    is_dup, d_id, d_name, dist = att._is_face_duplicate_1n(temp_path)

                if is_dup:
                    st.error(f"❌ REJECTED: This face is already enrolled as **{d_name}** (ID: `{d_id}`)")
                    os.remove(temp_path)
                else:
                    dest = os.path.join(att.DB_PATH, f"{e_id}_{e_name.replace(' ', '_')}.jpg")
                    shutil.move(temp_path, dest)
                    att.clear_deepface_cache()
                    st.success(f"✅ Successfully enrolled **{e_name}** (ID: `{e_id}`)")
                    st.rerun()
            else:
                if mode == "Webcam Capture":
                    st.warning("No photo was captured. Enrollment cancelled.")

# --- 4. Analytics & Reports (Admin Only) ---
elif menu == "Analytics & Reports":
    st.title("Data Management")
    t1, t2, t3, t4 = st.tabs(["Attendance", "Security Logs", "Late Report", "Spoof Attempts"])

    with t1:
        if os.path.exists(att.ATTENDANCE_CSV):
            st.dataframe(pd.read_csv(open(att.ATTENDANCE_CSV, encoding='utf-8', errors='replace')), width="stretch")
        else:
            st.info("No attendance data found.")

    with t2:
        if os.path.exists(att.LOG_CSV):
            st.dataframe(pd.read_csv(open(att.LOG_CSV, encoding='utf-8', errors='replace')), width="stretch")
        else:
            st.info("No recognition log yet.")

    with t3:
        if os.path.exists(att.ATTENDANCE_CSV):
            df_att = pd.read_csv(open(att.ATTENDANCE_CSV, encoding='utf-8', errors='replace'))
            today_rows = df_att[df_att['Date'] == today_str].copy()

            if today_rows.empty:
                st.info("No attendance records for today.")
            else:
                try:
                    from datetime import datetime as dt
                    cutoff = dt.strptime(att.LATE_AFTER, "%H:%M").time()
                    today_rows['_arrival'] = today_rows['Time'].apply(
                        lambda x: dt.strptime(x, "%H:%M:%S").time() if len(x) > 5
                        else dt.strptime(x, "%H:%M").time()
                    )
                    late_df = today_rows[today_rows['_arrival'] > cutoff][
                        ['Employee ID', 'Name', 'Date', 'Time', 'Status']
                    ].reset_index(drop=True)
                except Exception:
                    late_df = pd.DataFrame()

                if late_df.empty:
                    st.success(f"No late arrivals today (cutoff: {att.LATE_AFTER})")
                else:
                    st.warning(f"{len(late_df)} late arrival(s) — cutoff: {att.LATE_AFTER}")
                    st.dataframe(late_df, width="stretch")
        else:
            st.info("No attendance data found.")

    with t4:
        if os.path.exists(att.SPOOF_LOG_CSV):
            df_spoof = pd.read_csv(open(att.SPOOF_LOG_CSV, encoding='utf-8', errors='replace'))
            today_spoof = df_spoof[df_spoof['Date'] == today_str]
            st.metric("Spoof Attempts Today", len(today_spoof))
            if not df_spoof.empty:
                st.dataframe(df_spoof.sort_values("Date", ascending=False), width="stretch")
            else:
                st.success("No spoof attempts recorded.")
        else:
            st.info("No spoof log found yet.")

# --- 5. System Registry (Admin Only) ---
elif menu == "System Registry":
    st.title("Enrolled Personnel")

    if not all_employees:
        st.info("No employees enrolled yet.")
    else:
        for emp in all_employees:
            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.markdown(f"`{emp['id']}`")
            with col2:
                st.write(emp['name'].replace('_', ' '))
            with col3:
                if st.session_state.confirm_delete == emp['id']:
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("Confirm", key=f"confirm_{emp['id']}", type="primary"):
                            photo_path = os.path.join(att.DB_PATH, emp['file'])
                            if os.path.exists(photo_path):
                                os.remove(photo_path)
                            # validates the caller before touching the cache.
                            att.clear_deepface_cache()
                            st.session_state.confirm_delete = None
                            st.success(f"Removed {emp['name'].replace('_', ' ')}")
                            st.rerun()
                    with cc2:
                        if st.button("Cancel", key=f"cancel_{emp['id']}"):
                            st.session_state.confirm_delete = None
                            st.rerun()
                else:
                    if st.button("Delete", key=f"del_{emp['id']}", type="secondary"):
                        st.session_state.confirm_delete = emp['id']
                        st.rerun()
