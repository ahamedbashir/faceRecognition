from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = ap.parse_args()

detector = cv2.CascadeClassifier(args.cascade)

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L)
time.sleep(1.0)
total = 0

while True:
    ret, frame = cap.read()
    orig = frame
    frame = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30,30))

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("k"):
        p = os.path.sep.join([args.output, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1 
    elif key == ord("q"):
        break

print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
cap.release()
