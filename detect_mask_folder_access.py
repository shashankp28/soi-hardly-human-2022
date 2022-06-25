import cv2
import os

#import imutils
#from imutils.video import VideoStream

from services.utils import (EMO_COLOR_DICT, EMO_DICT, detect_emotion,
                            model_loader)

prototxtPath = os.path.join(os.path.dirname(os.curdir),"models/deploy.prototxt.txt")
weightsPath = os.path.join(os.path.dirname(os.curdir),"models/res10_300x300_ssd_iter_140000.caffemodel")

face_detector_paths = [prototxtPath, weightsPath]
emotion_detectot_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

faceNet, emotor = model_loader(face_detector_paths, emotion_detectot_path)

'''#Video Streaming
print("Initiating video stream...")
vs = VideoStream(src=0).start()
'''


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

files = load_images_from_folder("C:\\Users\\tkhal\\Documents\\GitHub\\soi-hardly-human-2022\\input")
index=0
for frame in files:
    (locs, preds) = detect_emotion(frame, faceNet, emotor)
    # Change this part (pred == list containing 7 probabilities See EMO_DICT for labels)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        '''(mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # display the label and bounding box rectangle on the output
        # frame'''
        index = preds[0].index(max(preds[0]))
        label = "{}: {:.2f}%".format(EMO_DICT[index], preds[0][index]*100 )
        cv2.putText(frame,label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, EMO_COLOR_DICT[index], 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), EMO_COLOR_DICT[index], 2)
    cv2.imshow("Frame", frame)
    cv2.imwrite("C:\\Users\\tkhal\\Documents\\GitHub\\soi-hardly-human-2022\\output\\{index}.png",frame)
    index+=1
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
