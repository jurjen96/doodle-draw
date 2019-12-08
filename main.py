import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tqdm import tqdm

REBUILD_DATA = True

LABELS = ['car', 'sailboat', 'bird', 'broom', 'butterfly', 'candle', 'clock',
          'flashlight', 'flower', 'airplane', 'house', 'violin', 'sock', 'saw']

def create_training_data():
    features = []
    labels = []
    dataset = []

    print("Loading the dataset")
    for label in tqdm(LABELS):
        dataset.append([np.load('data/' + label + '.npy'), label])

    print("Appending it to the labels and features")
    for doodle in tqdm(dataset):
        for img in doodle[0][:60000]:
            features.append(img)
            labels.append(np.eye(len(LABELS))[LABELS.index(doodle[1])])

    print("Saving the features and labels")
    np.save('object_features', np.array(features))
    np.save('object_labels', np.array(labels))
    return features, labels

if REBUILD_DATA:
    data, labels = create_training_data()
else:
    data = np.load('object_features.npy', allow_pickle=True)
    labels = np.load('object_labels.npy', allow_pickle=True)

data = np.array(data)

# Normalize the data
data = data / 255

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = np.array(y_train)
y_test = np.array(y_test)


# Display the first image of the training set
# print(y_train[0])
# plt.imshow(X=X_train[0].reshape((28,28)))
# plt.show()

# To save some memory:
del data
del labels

model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten()) # Convert the 3D features (height, width, features) to 1D feature map
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(LABELS), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=256,
                    validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=256)

print("SCORE:")
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


print("Saving model...")
model.save('cnn_object_model_' + str(time.time()) + '.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
