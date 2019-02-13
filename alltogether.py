import cv2
import imutils
import tkinter as tk
from PIL import Image, ImageTk
import face_recognition
import pickle

cap = cv2.VideoCapture(0, cv2.CAP_V4L)
data = pickle.loads(open("encodings.pickle", "rb").read())
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

display_flag=False
recognize_flag=False
train_flag=False

# Actions
def display_action():
    global display_flag
    global recognize_flag
    global train_flag
    recognize_flag = False
    train_flag = False
    if not display_flag:
        display_flag = True
        display()

def recognize_action():
    global display_flag
    global recognize_flag
    global train_flag
    display_flag = False
    train_flag = False
    if not recognize_flag:
        recognize_flag = True
        recognize()
        
def train_action():
    global display_flag
    global recognize_flag
    global train_flag
    display_flag = False
    recognize_flag = False
    if not train_flag:
        train_flag = True
        train()
        
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
    
# Main Function
root = tk.Tk()
root.attributes("-type","dialog")
top_container = tk.Label(root, bg="grey")
bottom_container = tk.Label(root, bg="black")
button_stream = tk.Button(bottom_container, text="Stream", command=display_action)
button_recognize = tk.Button(bottom_container, text="Recognize", command=recognize_action)
button_train = tk.Button(bottom_container, text="Train", command=train_action)
button_stream.pack(side="left", padx=10)
button_recognize.pack(side="left", padx=10)
button_train.pack(side="left", padx=10)
top_container.pack(expand=1, fill=tk.BOTH)
bottom_container.pack(fill=tk.X)
display_action()
root.mainloop()
cap.release()
