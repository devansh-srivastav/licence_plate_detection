from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

# Initialising CNN
classifier = Sequential()

# Adding Convolutional Layers and Pooling Layers
classifier.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flatenning Layer
classifier.add(Flatten())

# Full Connection
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 36, activation = 'softmax'))

# Compiling cnn
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting cnn
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('characters/train',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')

test_set = test_datagen.flow_from_directory('characters/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                    steps_per_epoch=4680//32,
                    epochs=15,
                    validation_data=test_set,
                    validation_steps=720//32,
                    workers = 16)


# Saving model
classifier.save('ocr.h5', include_optimizer = True)