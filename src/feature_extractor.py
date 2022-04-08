from tensorflow.keras.applications import xception

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

import numpy as np

class Xception_FE:
    def __init__(self):
        base_model = xception.Xception(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    def extract(self, img):
        img = img.resize((299, 299))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = xception.preprocess_input(x)
        feature = self.model.predict(x)[0]
        feature = feature / np.linalg.norm(feature)

        return feature