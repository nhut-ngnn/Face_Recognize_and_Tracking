import cv2
from src.models.detect_model import Face_Detection
from src.models.recognize_model import Face_Recognition
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import sqlite3

def getProfile(id):
    conn = sqlite3.connect("FaceDatabase.db")
    cursor = conn.execute("SELECT * FROM users WHERE ID=?", (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

detector = Face_Detection()

recog = Face_Recognition()
recog.fit_model("FaceDatabase")

tracker = DeepSort(max_age=20, n_init=1, nms_max_overlap=1.0, max_iou_distance=0.7)

if __name__ == "__main__":

    cam = cv2.VideoCapture("C:/Users/admin/Downloads/GreyD_test.mp4")  #Add path of video
    ret, frame = cam.read()
    detector.set_frame_len(frame)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    
    while(True):
        ret, frame = cam.read() 
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray_frame)
        re, acc = recog.recognize(gray_frame, faces)
        print(acc)
        tracks = tracker.update_tracks(re, frame=frame)
        
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                
                profile = getProfile(class_id)
                if profile != None:
                    label = "{}: {}".format(profile[0], profile[1])
                else:
                    label = "{}".format(class_id)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        out.write(frame)
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
    cv2.destroyAllWindows()