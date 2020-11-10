import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

tf.__version__
class CNN:
        def __init__(self):
                # preprocessing the training set
                self.train_datagen = ImageDataGenerator(
                        rescale=1. / 255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

                # class mode options is about how many different outputs we expect ('binary'/'categorical')
                self.training_set = self.train_datagen.flow_from_directory(
                        'dataset/training_set',
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='binary')

                # preprocessing the test set
                self.test_datagen = ImageDataGenerator(rescale=1. / 255)
                self.test_set = self.train_datagen.flow_from_directory(
                        'dataset/test_set',
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='binary')
                # initialising the cnn
                self.cnn = tf.keras.models.Sequential()

                # adding convolution layer
                self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

                # adding pooling layer
                self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

                # adding a second convolution and pooling layer
                self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
                self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

                # flatteing
                self.cnn.add(tf.keras.layers.Flatten())

                # full connection
                self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

                # output layer
                self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

                # compiling the model
                self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                # training the model
                self.cnn.fit(x=self.training_set, validation_data=self.test_set, epochs=25)

                # we can see what is what by looking into the class_indices attribute
                # print(self.training_set.class_indices)
        def save_model(self):
                self.cnn.save(
                        'model.h5',
                        overwrite=True,
                        include_optimizer=True,
                )

new_cnn = CNN()
new_cnn.save_model()

