import numpy as np
import cv2
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

def _load_printed_digits(base_dir):
    x = []
    y = []

    for label in range(10):
        label_dir = os.path.join(base_dir, str(label))
        if not os.path.isdir(label_dir):
            continue

        for name in os.listdir(label_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            path = os.path.join(label_dir, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            x.append(img)
            y.append(label)

    if not x:
        return None, None

    x = np.array(x, dtype=np.float32).reshape(-1, 28, 28, 1)
    x /= 255.0
    y = to_categorical(np.array(y, dtype=np.int32), num_classes=10)
    return x, y


def train():
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    
    x_train /= 255
    x_test /= 255

    # One-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    num_classes = y_test.shape[1]

    printed_dir = os.path.join("data", "printed_digits")
    printed_x, printed_y = _load_printed_digits(printed_dir)
    if printed_x is not None:
        print(f"Loaded printed digit samples: {printed_x.shape[0]} from {printed_dir}")
        target = 20000
        repeat = int(np.ceil(target / float(printed_x.shape[0])))
        printed_x = np.repeat(printed_x, repeat, axis=0)
        printed_y = np.repeat(printed_y, repeat, axis=0)

        x_train = np.concatenate([x_train, printed_x], axis=0)
        y_train = np.concatenate([y_train, printed_y], axis=0)

        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

    print("Building model...")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("Training model (this might take a while)...")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=1)
    
    print("Saving model to model.h5...")
    model.save('model.h5')
    print("Done!")

if __name__ == "__main__":
    train()
