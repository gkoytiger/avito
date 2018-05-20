from keras.applications import xception
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import re
import glob
from zipfile import ZipFile

### Globals
TEST_MODE = False
ZIP_FILES = ['/home/greg/.kaggle/competitions/avito-demand-prediction/train_jpg.zip',
             '/home/greg/.kaggle/competitions/avito-demand-prediction/test_jpg.zip']
DATA_FOLDER = '/home/greg/.kaggle/competitions/avito-demand-prediction/'
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, 'derived', 'xception_img_features')

def get_features(base_model, model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = base_model.preprocess_input(x)
    predictions = model.predict(x)
    #global average pooling 2d
    predictions = np.mean(predictions, axis=(1,2))[-1]
    return predictions

if __name__ == '__main__':
    model = xception.Xception(weights='imagenet', include_top=False)

    for zip_file in ZIP_FILES:
        zipf = ZipFile(zip_file, 'r')
        image_files = zipf.namelist()
        image_features = []
        image_names = []

        if TEST_MODE:
            image_files = image_files[:100]
        for image_file in image_files:
            imagef = zipf.open(image_file)
            try: img = image.load_img(imagef, target_size=(224, 224))
            except: continue 
            image_features.append(get_features(xception, model, img))
            image_names.append(image_file[:-4])

        image_features = pd.DataFrame(data=np.vstack(image_features),
                                    index=image_names)
        file_name = os.path.join(OUTPUT_FOLDER, os.path.basename(zip_file)[:-4] + '.hdf5')
        if os.path.exists(file_name): os.remove(file_name)
        image_features.to_hdf(file_name, key='img_features', complevel=3)
