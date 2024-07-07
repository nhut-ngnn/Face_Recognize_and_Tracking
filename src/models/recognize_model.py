from sklearn.svm import SVC 
import pickle
import cv2
from cv2 import resize
import os
import numpy as np
from sklearn.decomposition import PCA

from src.preprocessing.extract_feature import extract_lbp_features, extract_hog_features


def load_model(path:str):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_model(model, path:str):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


class Face_Recognition(SVC):

    def __init__(self) -> None:
        super().__init__(kernel="linear", probability=True)
        self.pca_transf = PCA()
        self.input_img = (128, 128)
        self.threshold = 0.8
    
    
    def _reshape_img(self, img):
        return resize(img, self.input_img)

    def recognize(self, frame, faces):
        recog_faces = []
        for (x1, y1, x2, y2) in faces:
            face = frame[y1:y2, x1:x2]

            name = self.predict(face)[0]
            name_prob = self.predict_proba(face)[0]
            name_prob = np.max(name_prob)

            if name_prob < self.threshold:
                name = "Unknown"
                name_prob = 0
            recog_faces.append([[x1, y1, x2-x1, y2 - y1], f"{name_prob:.2f}", name])

        return recog_faces
    
    def _get_features(self, img):
        lbp_features = extract_lbp_features(img)
        hog_features = extract_hog_features(img)

        features = np.concatenate((lbp_features, hog_features))
        return features


    def _convert_img_to_feature(self, img):
        img = self._reshape_img(img)
        X = self._get_features(img)
        X = self.pca_transf.transform([X])
        return X

    def _get_data_from_db(self, face_db):
        X = []
        y = []

        names = os.listdir(path=face_db)
        for name in names:
            imgs = os.listdir(f"{face_db}/{name}")
            for img in imgs:
                _img = cv2.imread(f"{face_db}/{name}/{img}")
                _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

                embeding = self._get_features(_img)

                X.append(embeding)
                y.append(name)
        return np.array(X), np.array(y)


    def fit_model(self, face_db) -> np.ndarray:        
        X, y = self._get_data_from_db(face_db=face_db)
        X = self.pca_transf.fit_transform(X, y)
        self.fit(X, y)
    
    def predict(self, img):
        X = self._convert_img_to_feature(img)
        return super().predict(X)
    
    def predict_proba(self, img) -> np.ndarray:
        X = self._convert_img_to_feature(img)
        return super().predict_proba(X)
    



