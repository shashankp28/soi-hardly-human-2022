import cv2
import numpy as np
import tensorflow as tf


EMO_DICT = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
EMO_COLOR_DICT = {0: (0, 0, 255), 2: (60, 20, 9), 1: (71,122,75), 3: (0,255,0), 4: (255,0,0), 5: (255,255,255), 6: (0,0,0)}


def model_loader(faceNet_paths, emotor_path):
    faceNet = cv2.dnn.readNet(faceNet_paths[0], faceNet_paths[1])
    emotor = tf.keras.models.load_model(emotor_path)
    return faceNet, emotor

    
def convert_to_square(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    square_length = ((xmax - xmin) + (ymax - ymin)) // 2 // 2
    square_length *= 1.1

    xmin = int(center_x - square_length)
    ymin = int(center_y - square_length)
    xmax = int(center_x + square_length)
    ymax = int(center_y + square_length)
    return xmin, ymin, xmax, ymax

def detect_emotion(frame, faceNet, emotor):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.66:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            startX, startY, endX, endY = convert_to_square(startX, startY, endX, endY)
            face = frame[startY:endY, startX:endX]
            try: 
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
            except: break
            face=np.array(face)
            face=face.reshape(1,48,48,1)/255
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    preds = []
    
    if len(faces) > 0:
        for face in faces:
            output= emotor.predict(face).tolist()
            preds.append(output[0])
    return (locs, preds)