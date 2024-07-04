import cv2
import os
from src.models.detect_model import Face_Detection

def capture_faces():
    file_path = "Raw Database/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    name = input('Input Name: ')

    user_folder = os.path.join(file_path, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    print("Record Face, press q to exit!")

    sample_num = 0

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample_num += 1
            cv2.imwrite(os.path.join(user_folder, f"{name}{sample_num}.jpg"), gray[y:y + h, x:x + w])
        cv2.imshow('frame', img)
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break
        elif sample_num > 300:
            break

    cam.release()
    cv2.destroyAllWindows()

def save_image(fr, dest):
    model = Face_Detection()
    names = os.listdir(path=fr)

    for name in names:
        imgs = os.listdir(f"{fr}/{name}")
        os.makedirs(f"{dest}/{name}")
        for img in imgs:
            
            _img = cv2.imread(f"{fr}/{name}/{img}")
            model.set_frame_len(_img)

            faces = model.detectMultiScale(_img)
            
            for (x1, y1, x2, y2) in faces:
                face = cv2.resize(_img[y1:y2, x1:x2], (128, 128))
                cv2.imwrite(img=face, filename=f"./{dest}/{name}/{img}")

if __name__ == "__main__":
    capture_faces()
    save_image("Raw Database", "Face Database")
