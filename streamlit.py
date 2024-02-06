import streamlit as st
import cv2

# Streamlit settings
st.set_page_config(layout="wide")
st.title("OpenCV + Streamlit Integration")

# OpenCV script settings
video_path = '/Users/onurdenizoguncu/Desktop/PROJ201/ProjOpenCv/abacus_capture_min.mp4'
cap = cv2.VideoCapture(video_path)


# Read frames from the OpenCV script
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display frame in Streamlit
    col1.image(frame, channels="BGR", use_column_width=True)

    # Display additional information
    col2.write("Additional Information:")
    # Add more information as needed

    # Check for Streamlit events (e.g., button clicks)
    if st.button("Stop"):
        break

# Release the video capture when done
cap.release()
