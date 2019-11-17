import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
# import tensorflowjs as tfjs
import matplotlib.pyplot as plt

from tqdm import tqdm

data = []
data.append([np.load('resources/numpy_bitmap/car.npy'), 'car'])
data.append([np.load('resources/numpy_bitmap/bed.npy'), 'bed'])
data.append([np.load('resources/numpy_bitmap/apple.npy'), 'apple'])

#plt.imshow(X=data[3].reshape((28,28)))
#plt.show()

REBUILD_DATA = True

LABELS = {
    'car': 0,
    'apple': 1,
    'bed': 2
}

def create_training_data():
    features = []
    labels = []
    for doodle in tqdm(data):
        for img in doodle[0]: #[:200]:
            features.append(img) #.reshape(28,28)
            labels.append(np.eye(len(LABELS))[LABELS[doodle[1]]])
            # training_data.append([img.reshape((28,28)), np.eye(2)[LABELS[doodle[1]]]])  #img 1 label
    # np.random.shuffle(training_data)
    # np.save("car_bed.npy", training_data)
    return features, labels

if REBUILD_DATA:
    data, labels = create_training_data()
else:
    training_data = np.load('car_bed.npy', allow_pickle=True)

data = np.array(data)

# Normalize the data
data = data / 255

X = data
y = labels

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Dense(2))
# model.add(Activation('softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#
# model.fit(X_train, y_train, batch_size=64, epochs=10)



model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Convert the 3D features (height, width, features) to 1D feature map
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=32)

print("SCORE:")
print("Test loss: ", test_loss)
print("Test accuracy", test_acc)


print(history.history)
print(history.history.keys())
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


print("Saving model...")
model.save('cnn_model_3_' + str(time.time()) + '.h5')
# tfjs.converters.save_keras_model(model, "model.json")
