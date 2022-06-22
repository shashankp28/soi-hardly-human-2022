import cv2
import imutils
from imutils.video import VideoStream
from services.utils import EMO_COLOR_DICT, EMO_DICT, detect_emotion, model_loader



prototxtPath = r"models/deploy.prototxt.txt"
weightsPath = r"models/res10_300x300_ssd_iter_140000.caffemodel"

face_detector_paths = [prototxtPath, weightsPath]
emotion_detectot_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

faceNet, emotor = model_loader(face_detector_paths, emotion_detectot_path)


print("Initiating video stream...")
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
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
        cv2.putText(frame, EMO_DICT[index], (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, EMO_COLOR_DICT[index], 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), EMO_COLOR_DICT[index], 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
