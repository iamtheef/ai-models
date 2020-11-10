from keras.models import load_model
from keras_preprocessing import image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


img = 'dataset/single_prediction/cat_or_dog_1.jpg'

def make_prediction(img_path):
    loaded = load_model('model.h5')
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    return prediction

print(make_prediction(img))