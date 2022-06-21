# import the necessary packages
import torch
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from model import resmasking_dropout1
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
from torchvision.transforms import transforms

transform = transforms.Compose(
    transforms=[transforms.ToPILImage(), transforms.ToTensor()]
)

EMO_DICT = {0: "neutral", 1: "angry", 2: "disgust", 3: "fear", 4: "happy", 5: "sad", 6: "surprise"}

    
def convert_to_square(xmin, ymin, xmax, ymax):
    # convert to square location
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    square_length = ((xmax - xmin) + (ymax - ymin)) // 2 // 2
    square_length *= 1.1

    xmin = int(center_x - square_length)
    ymin = int(center_y - square_length)
    xmax = int(center_x + square_length)
    ymax = int(center_y + square_length)
    return xmin, ymin, xmax, ymax

def detect_mask(frame, faceNet, maskNet):
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
            face = cv2.resize(face, (224, 224))
            face = transform(face)
            face = torch.unsqueeze(face, dim=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    preds = []
    
    if len(faces) > 0:
        for face in faces:
            output = torch.squeeze(maskNet(face), 0)
            proba = torch.softmax(output, 0)
            preds.append(proba.tolist())
        print(preds)

    return (locs, preds)


prototxtPath = r"deploy.prototxt.txt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = resmasking_dropout1()
maskNet.load_state_dict(torch.load("Z_resmasking_dropout1_rot30_2019Nov30_13.32")['net'])
maskNet.eval()


# maskNet = load_model("mask_detector.model")

print("Initiating video stream...")
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_mask(frame, faceNet, maskNet)
    '''
    # Change this part (pred == list containing 7 probabilities See EMO_DICT for labels)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    '''
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
