import cv2
from src.models.detect_model import Face_Detection
from src.models.recognize_model import load_model, Face_Recognition

from deep_sort_realtime.deepsort_tracker import DeepSort

import numpy as np

detector = Face_Detection()

recog = Face_Recognition()
recog.fit_model("Face Database")

tracker = DeepSort(max_age=20)

if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    detector.set_frame_len(frame)

    while(True):
        ret, frame = cam.read() 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.flip(frame,1)

        faces = detector.detectMultiScale(gray_frame)
        re = recog.recognize(gray_frame, faces)


        tracks = tracker.update_tracks(re, frame=frame)


        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                        
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
        
                label = "{}".format(class_id)
        
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (0,255,0), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
    cv2.destroyAllWindows()