import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import streamlit as st

model = YOLO('yolov8s.pt')

control_area1 = [(452, 347), (555, 186), (503, 165), (394, 324)]
control_area2 = [(388, 322), (496, 162), (473, 154), (364, 310)]

cap = cv2.VideoCapture('/Users/onurdenizoguncu/Desktop/PROJ201/ProjOpenCv/cospaceEntranceCapture-Grayscale-360p.mp4')

my_file = open("/Users/onurdenizoguncu/Desktop/PROJ201/ProjOpenCv/coco.txt", "r")

data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()

people_enters = {}
people_exits = {}

entering = set()
exiting = set()

st.set_page_config(layout="wide")
st.title("Co-Space Entrance")

# Cache Streamlit elements to update only on changes
entered_cache = st.empty()
exited_cache = st.empty()
present_cache = st.empty()
entering_cache = st.empty()
exiting_cache = st.empty()
image_cache = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)

    start_point = (30, 30)
    end_point = (290, 191)
    color = (255, 0, 0)
    thickness = -1

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(control_area2, np.int32), (x4, y4), False)

        if results >= 0:
            people_enters[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in people_enters:
            results1 = cv2.pointPolygonTest(np.array(control_area1, np.int32), (x4, y4), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                entering.add(id)

        results2 = cv2.pointPolygonTest(np.array(control_area1, np.int32), (x4, y4), False)

        if results2 >= 0:
            people_exits[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in people_exits:
            results3 = cv2.pointPolygonTest(np.array(control_area2, np.int32), (x4, y4), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                exiting.add(id)

    cv2.polylines(frame, [np.array(control_area1, np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(control_area2, np.int32)], True, (255, 0, 0), 2)

    enter_count = len(entering)
    exit_count = len(exiting)
    present = len(entering) - len(exiting)

    # Update Streamlit content only when there's a change
    entered_cache.text(f"Entered: {enter_count}")
    exited_cache.text(f"Exited: {exit_count}")
    present_cache.text(f"Present: {present}")
    image_cache.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()