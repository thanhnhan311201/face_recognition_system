import os
import pickle
from PIL import Image
import numpy as np
import cv2
from scipy import spatial
import time
import glob
from tqdm import tqdm
import json

try:
    import feature_extractor
    import face_detector
except:
    from src import feature_extractor
    from src import face_detector

class face_recognition_system:
    def __init__(self, dataset_path, image_folder):
        self.extractor_method = 'Xception_FE'
        self.extractor = feature_extractor.Xception_FE()

        self.detector = face_detector.FaceDetector()

        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        self.image_folder = image_folder

        self.feature_folder_path = dataset_path + 'feature_vectors/'
        if not os.path.exists(self.feature_folder_path):
            os.mkdir(self.feature_folder_path)

    def index_dataset(self):
        for img_path in tqdm(os.listdir(self.image_folder)):
            name = img_path.split('.')[0]
            vector_file = f'{self.feature_folder_path}/{name}.pkl'
            
            img_path_full = os.path.join(self.image_folder + img_path)
            img = cv2.imread(img_path_full)

            if self.detector.detect(img) is None:
                return 0

            x_min, y_min, x_max, y_max = self.detector.detect(img)
            PIL_image = Image.open(img_path_full).crop((x_min, y_min, x_max, y_max))
                
            try:
                feature_vector = self.extractor.extract(PIL_image)
            except:
                continue

            pickle.dump(feature_vector, open(vector_file, "wb"))

        return 'Done!'

    def verify_face(self, img):
        res_dict = {}

        max_similarity = 0.0
        for vector_file in os.listdir(self.feature_folder_path):
            feature_vectors = pickle.load(open(self.feature_folder_path + vector_file,"rb"))

            if isinstance(img, str):
                img = Image.open(img)

            feature_img = self.extractor.extract(img)

            # calculate the cosine similarity
            cos_sim = 1 - spatial.distance.cosine(feature_vectors, feature_img)

            name = vector_file.split('.')[0]
            temp = {'similary': cos_sim}
            res_dict[name] = temp

            if cos_sim > max_similarity:
                max_similarity = cos_sim
                face_name = vector_file.split('.')[0]

        if max_similarity < 0.5:
            return 'Unknown'

        return face_name, max_similarity, res_dict

    def recognize_face_via_image(self, img):
        if isinstance(img, str):
            PIL_img = Image.open(img)
            img = cv2.imread(img)

        if self.detector.detect(img) is None:
            return 0

        x_min, y_min, x_max, y_max = self.detector.detect(img)
        processed_img = PIL_img.crop((x_min, y_min, x_max, y_max))
        res_img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if self.verify_face(processed_img) == "Unknown":
            face_name = "Unknown"
            cv2.putText(res_img, face_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            face_name, similarity, res_dict = self.verify_face(processed_img)
            cv2.putText(res_img, f'{face_name}: {round(similarity, 2)}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        res_file = open("log.json", "w")
        json.dump(res_dict, res_file, indent=6)
        res_file.close()

        cv2.imshow("Result", res_img)
        cv2.waitKey(0)

        return res_img

    def recognize_face_via_camera(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()

            if ret is None:
                continue

            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            if self.detector.detect(frame) is None:
                continue

            x_min, y_min, x_max, y_max = self.detector.detect(frame)
            processed_frame = Image.fromarray(np.uint8(frame[y_min:y_max, x_min:x_max])).convert('RGB')
            res_frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if self.verify_face(processed_frame) == "Unknown":
                face_name = "Unknown"
                cv2.putText(res_frame, face_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                face_name, similarity, _ = self.verify_face(processed_frame)
                cv2.putText(res_frame, f'{face_name}: {round(similarity, 2)}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Result", res_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()