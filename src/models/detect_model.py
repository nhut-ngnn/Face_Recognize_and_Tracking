import cv2
import numpy as np


class Face_Detection(cv2.CascadeClassifier):
    def __init__(self, type=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
        super().__init__(type)
        self.frame_len = 0.1
        if self.empty():
            raise IOError("Unable to load the face cascade classifier xml file")
    def set_frame_len(self, frame):
        width, length, deep = frame.shape
        self.frame_len = np.sqrt(width**2 + length**2)

    def detectMultiScale(self, frame):
        faces = super().detectMultiScale(frame)
        if faces is tuple():
            return []
        faces[:,2:3] = faces[:,0:1] + faces[:,2:3]
        faces[:,3:4] = faces[:,1:2] + faces[:,3:4]

        distances = faces[:, 0:2] - faces[:, 2:4]
        distances = np.sqrt((distances**2).sum(axis=1))

        faces = faces[distances > 0.1*self.frame_len]


        return faces

