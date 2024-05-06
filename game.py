import cv2
import time
import pyautogui
import numpy as np
import keyboard
from keras.models import load_model, model_from_json


def takeScreenShot():
    screenshot = pyautogui.screenshot()
    np_frame = np.array(screenshot)
    frame = cv2.cvtColor(src=np_frame, code=cv2.COLOR_RGB2GRAY)
    return frame

def cutScreenShot():
    #coordinates = {'top': 444, 'left':650, 'width':250, 'height':100}
    x, y, w, h = 650, 444, (650+250), (444+100)
    frame = takeScreenShot()
    frame = frame[y:h, x:w]
    cv2.imshow(winname="zz", mat=frame)
    if cv2.waitKey(1) == ord("q"): pass
    try:
        height, width, channel = frame.shape
    except:
        height, width = frame.shape
        channel = 1
    return frame, (height, width, channel)

def reshape_frame(frame, shape):
    height, width, channel = shape
    shape = (1, height, width, 1)
    frame = frame / 255.
    X = np.array([frame])
    frame = np.reshape(a=frame, newshape=shape)
    
    return frame

def load_modell():
    model = model_from_json(open('model_new.json', 'r').read())
    #model = load_model("trexai_arch1.h5")
    model.load_weights('trexai_arch1.h5')
    return model

def predict(model, frame):
    # 0:up 1:down 2:right
    result = model.predict(frame)
    classes = np.argmax(result)
    print(result, classes)
    return classes


def process():
    model = load_modell()
    while True:
        frame, shape = cutScreenShot()
        frame = reshape_frame(frame=frame, shape=shape)
        
        klass = predict(model=model, frame=frame)
        

        classes = {"up":0, "down":1, "right":2}

        if klass == 0:
            keyboard.press('up')
            time.sleep(0.05)
            keyboard.release('up')
        elif klass == 1:
            keyboard.press('down')
            time.sleep(0.05)
            keyboard.release('down')
        elif klass == 2:
            pass

        #cv2.imshow(winname="cutting", mat=frame)
        #if cv2.waitKey(0) == ord("q"): exit()
        
        

process()