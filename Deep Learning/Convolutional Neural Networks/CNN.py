import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
import numpy as np

tf.__version__

# preprocessing the training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# class mode options is about how many different outputs we expect ('binary'/'categorical')
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# initialising the cnn
cnn = tf.keras.models.Sequential()

# adding convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# adding pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# adding a second convolution and pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# flatteing
cnn.add(tf.keras.layers.Flatten())

# full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# compiling the model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training the model
# cnn.fit(x=training_set, validation_data=test_set, epochs=25)

test_image = image.load_img('dataset/single_prediction/cat_or_dog_5.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
        prediction = 'dog'
else:
        prediction = 'cat'

print(prediction)


