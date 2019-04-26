import cv2
import imutils
from tkinter import *
from PIL import Image, ImageTk
import face_recognition
import pickle
import os
import time
from imutils.video import VideoStream

# train module
import trainFace

cap = cv2.VideoCapture(0, cv2.CAP_V4L)
data = pickle.loads(open("encodings.pickle", "rb").read())
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

display_flag=False
recognize_flag=False
capture_flag = False
train_flag=False

# Actions
def displayAction():
    global display_flag
    global recognize_flag
    global train_flag
    global capture_flag
    capture_flag = False
    recognize_flag = False
    train_flag = False
    if not display_flag:
        display_flag = True
        display()

def recognizeAction():
    global display_flag
    global recognize_flag
    global train_flag
    global capture_flag
    capture_flag = False
    display_flag = False
    train_flag = False
    if not recognize_flag:
        recognize_flag = True
        recognize()
        
def trainAction():
    global display_flag
    global recognize_flag
    global train_flag
    global capture_flag
    capture_flag = False
    display_flag = False
    recognize_flag = False
    if not train_flag:
        train_flag = True
        train()
        
def captureAction():
        global display_flag
        global recognize_flag
        global train_flag
        display_flag = False
        recognize_flag = False
        train_flag = False
        if entry.get():
                capture()
        
# Repetitive functions
def display():
    if display_flag:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        top_container.itk  = frame
        top_container.configure(image=frame)
        top_container.after(10, display)

def recognize():
    if recognize_flag:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        rects = face_recognition.face_locations(frame, model="hog")
        encodings = face_recognition.face_encodings(frame, rects)
        names = []
        name = "unknown"
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            indexes = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            if len(indexes) > 0:
                for i in indexes:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
                names.append(name)
        for ((top, right, bottom, left), name) in zip(rects, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        top_container.itk  = frame
        top_container.configure(image=frame)
        top_container.after(10, recognize)

def capture() :
        name = entry.get()
        print(name)
        pathDir = "dataset/"+name
        cascade ="haarcascade_frontalface_default.xml"

        detector = cv2.CascadeClassifier(cascade)
    
        if not os.path.exists(pathDir):
                os.mkdir(pathDir)
        # vs = VideoStream(src=0).start()
        time.sleep(1.0)
        total = len(next(os.walk(pathDir))[2])
        print(total)
        ret, frame = cap.read()
        orig = frame.copy()

        rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30,30))

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        key = cv2.waitKey(1) & 0xFF
        
        total+= 1
        p = os.path.sep.join([pathDir, "{}.png".format(str(total).zfill(5))])
        print(p)
        cv2.imwrite(p, orig)
        
        print("[INFO] {} face images stored".format(total))

def train():
    if train_flag:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for(x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        top_container.itk  = frame
        top_container.configure(image=frame)
        top_container.after(10, train)
        trainFace.encodeFace()
    
# Main Function
root = Tk()
root.title("Face Recognition")
root.attributes("-type","dialog")
top_container = Label(root, text='Face Recognition', bg="grey")
bottom_container = Label(root, bg="black")
text_container = Label(root,bg="black")
button_stream = Button(bottom_container, text="Stream", command=displayAction)
button_recognize = Button(bottom_container, text="Recognize", command=recognizeAction)
button_captureFace = Button(bottom_container, text="Capture", command = captureAction)
button_train = Button(bottom_container, text="Train", command=trainAction)

entry = Entry(root)
button_stream.pack(side="left")
button_recognize.pack(side="left")
button_captureFace.pack(side="left")

Label(text_container, text="Name :").pack(side="left")
entry = Entry(text_container, width=30)
entry.pack(side="left")

button_train.pack(side="left")
top_container.pack(expand=1, fill=BOTH)
bottom_container.pack(fill=X)
text_container.pack(fill=X)
displayAction()
root.mainloop()
cap.release()
