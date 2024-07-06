import cv2
import os
import sqlite3
from src.models.detect_model import Face_Detection

def insertOrUpdate(id, name):
    conn = sqlite3.connect("FaceDatabase.db")
    cursor = conn.execute('SELECT * FROM users WHERE ID=?', (id,))
    isRecordExist = cursor.fetchone() is not None

    if isRecordExist:
        cmd = "UPDATE users SET Name=? WHERE ID=?"
        conn.execute(cmd, (name, id))
    else:
        cmd = "INSERT INTO users (ID, Name) VALUES (?, ?)"
        conn.execute(cmd, (id, name))

    conn.commit()
    conn.close()
    
    
def capture_faces():
    file_path = "FaceDatabase/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    ids = input("Input IDs: ")
    name = input('Input Name: ')
    insertOrUpdate(ids, name)
    
    user_folder = os.path.join(file_path, ids)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    print("Record Face, press q to exit!")

    sample_num = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample_num += 1
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (128, 128))
            cv2.imwrite(os.path.join(user_folder, f"{name}{sample_num}.jpg"), resized_face)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif sample_num >= 300:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()
