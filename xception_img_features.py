from keras.applications import xception
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import re

### Globals
os.chdir('/Users/grigoriykoytiger/Desktop/avito/avito')
TEST_MODE = True
IMAGES_DIR = '../data/train_jpg_0'
OUTPUT_FILE = '../data/derived/xception_img_features.hdf5'

def get_features(base_model, model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = base_model.preprocess_input(x)
    predictions = model.predict(x)
    #global average pooling 2d
    predictions = np.mean(predictions, axis=(1,2))[-1]
    return predictions

if __name__ == '__main__':
    image_files = [x.path for x in os.scandir(IMAGES_DIR)]

    if TEST_MODE:
        image_files = image_files[:1000]
    
    model = xception.Xception(weights='imagenet', include_top=False)
    image_features = [get_features(xception, model, image_file) for image_file in image_files]
    image_names = [re.search("000.+\.", image_file).group(0)[0:-1] for image_file in image_files]

    image_features = pd.DataFrame(data=np.vstack(image_features),
                                  index=image_names)

    image_features.to_hdf(OUTPUT_FILE, key='img_features')
