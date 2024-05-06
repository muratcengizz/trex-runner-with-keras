import keyboard
import uuid
import time
from PIL import Image
from mss import mss

coordinates = {'top': 444, 'left':650, 'width':250, 'height':100}
sct = mss()

i = 0

def record_screen(record_id, key):
    global i 
    i += 1

    print(f'{key}: {i}')
    img = sct.grab(coordinates)
    im = Image.frombytes(mode='RGB', size=img.size, data=img.rgb)
    im.save(f"./images/{key}_{record_id}_{i}.png")

is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey('esc', exit)

record_id = uuid.uuid4()

while True:
    if is_exit: break

    try:
        if keyboard.is_pressed(hotkey=keyboard.KEY_UP):
            record_screen(record_id=record_id, key='up')
            time.sleep(0.1)
        elif keyboard.is_pressed(hotkey=keyboard.KEY_DOWN):
            record_screen(record_id=record_id, key='down')
            time.sleep(0.1)
        elif keyboard.is_pressed(hotkey="right"):
            record_screen(record_id=record_id, key='right')
            time.sleep(0.1)
    except Exception as e:
        print(e)
        continue