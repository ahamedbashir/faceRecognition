from imutils.video import VideoStream
import cv2
import time
import imutils
import face_recognition
import pickle

cap = cv2.VideoCapture(0, cv2.CAP_V4L)
time.sleep(1.0)
data = pickle.loads(open("encodings.pickle", "rb").read())

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    rects = face_recognition.face_locations(frame, model="hog")
    encodings = face_recognition.face_encodings(frame, rects)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        indexes = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        if len(indexes) > 0:
            for i in indexes:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = (max(counts, key=counts.get),'unknown')[counts[max(counts)] < 15]
            names.append(name)

    for ((top, right, bottom, left), name) in zip(rects, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    cv2.namedWindow("Frame")
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    
    
cv2.destroyAllWindows()
cap.release()
