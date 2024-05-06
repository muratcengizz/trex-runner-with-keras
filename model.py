import os
import glob
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

imgs = glob.glob(pathname="./images/*.png")

width = 125
height = 50

def preprocess(imgs, width, height):
    X, y = [], []
    for img in imgs:
        filename = os.path.basename(p=img)
        label = filename.split('_')[0]
        #im = np.array(object=Image.open(fp=img).convert(mode='L').resize(width, height))
        im = np.array(object=Image.open(fp=img).convert(mode="L").resize(size=(width, height)))
        im = im / 255
        X.append(im)
        y.append(label)
    return X, y

def reshape_images(X, width, height):
    X = np.array(X)
    X = X.reshape(X.shape[0], width, height, 1)
    return X

def visualization(labels):
    sns.countplot(x=labels)
    plt.show()

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y=values)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def train_test_splitt(X, y, test_size, random_state):
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_X, test_X, train_Y, test_Y


def create_cnn():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=3, activation='softmax'))
    return model

def train_model(model, train_X, train_Y):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=50, batch_size=64)
    return model

def evaluate_model(model, train_X, train_Y, test_X, test_Y):
    score_train = model.evaluate(train_X, train_Y)
    score_test = model.evaluate(test_X, test_Y)
    return score_train, score_test

def print_weight_file(model, model_name):
    model.save(f'{model_name}.h5')

def process():
    global imgs, width, height
    X, y = preprocess(imgs=imgs, width=width, height=height)
    X = reshape_images(X=X, width=width, height=height)
    y = onehot_labels(values=y)
    train_X, test_X, train_Y, test_Y = train_test_splitt(X=X, y=y, test_size=0.25, random_state=10)
    model = create_cnn()
    model = train_model(model=model, train_X=train_X, train_Y=train_Y)
    score_train, score_test = evaluate_model(model=model, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
    print_weight_file(model=model, model_name='trex')
    print(f"Train Score: {score_train[1]*100:.2f}\nTest Score: {score_test[1]*100:.2f}")

process()

