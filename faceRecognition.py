#facerecognition.py
import cv2;
import imutils;
from imutils import paths;
import tkinter as tk;
from PIL import Image, ImageTk;
import face_recognition;
import pickle;
from tkinter import Entry;
import os;
import threading;
import time;

cap = cv2.VideoCapture(0, cv2.CAP_V4L);
pickleFound = True;
try:
    data = pickle.loads(open("encodings.pickle", "rb").read());
except (OSError, IOError) as e:
    pickleFound = False
    data = 3;
    pickle.dump(data, open("encodings.pickle", "wb"));

# data = pickle.loads(open("encodings.pickle", "rb").read());
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

display_flag=False;
recognize_flag=False;
new_face_flag=False;


def faceRecognition():
    try:
        # Actions
        def display_action():
            global display_flag;
            global recognize_flag;
            global new_face_flag;
            recognize_flag = False;
            new_face_flag = False;
            bottom_container.pack_forget();
            if not display_flag:
                display_flag = True;
                display();

        def recognize_action():
            global display_flag;
            global recognize_flag;
            global new_face_flag;
            global data;
            global pickleFound;
            if not pickleFound:
                    new_face_flag = True;
                    print("Created New subscriptable object", end=" ");
                    train();
                    pickleFound = True;

            display_flag = False;
            new_face_flag = False;
            bottom_container.pack_forget();
            if not recognize_flag:
                recognize_flag = True;
                data = pickle.loads(open("encodings.pickle", "rb").read());
                recognize();
                
        def new_face_action():
            global display_flag;
            global recognize_flag;
            global new_face_flag;
            display_flag = False;
            recognize_flag = False;
            bottom_container.pack(fill=tk.X);
            if not new_face_flag:
                new_face_flag = True;
                new_face();
                
        # Repetitive functions
        def display():
            if display_flag:
                ret, frame = cap.read();
                frame = imutils.resize(frame, width=400);
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
                frame = Image.fromarray(frame);
                frame = ImageTk.PhotoImage(image=frame);
                top_container.itk  = frame;
                top_container.configure(image=frame);
                top_container.after(10, display);

        def recognize():
            if recognize_flag:
                ret, frame = cap.read();
                frame = imutils.resize(frame, width=400);
                rects = face_recognition.face_locations(frame, model="hog");
                encodings = face_recognition.face_encodings(frame, rects);
                names = [];
                name = "unknown";
                for encoding in encodings:
                    matches = face_recognition.compare_faces(data["encodings"], encoding);
                    indexes = [i for (i, b) in enumerate(matches) if b];
                    counts = {};
                    if len(indexes) > 0:
                        for i in indexes:
                            name = data["names"][i];
                            counts[name] = counts.get(name, 0) + 1;
                        name = (max(counts, key=counts.get),"unknown")[counts[max(counts, key=counts.get)] < data["names"].count(max(counts, key=counts.get))*0.7];
                        names.append(name);
                    else:
                        name = "unknown";
                        names.append(name);
                for ((top, right, bottom, left), name) in zip(rects, names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2);
                    y = top - 15 if top - 15 > 15 else top + 15;
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2);
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
                frame = Image.fromarray(frame);
                frame = ImageTk.PhotoImage(image=frame);
                top_container.itk  = frame;
                top_container.configure(image=frame);
                top_container.after(10, recognize);

        def new_face():
            if new_face_flag:
                ret, frame = cap.read();
                frame = imutils.resize(frame, width=400);
                rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30,30));
                for(x, y, w, h) in rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2);
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
                frame = Image.fromarray(frame);
                frame = ImageTk.PhotoImage(image=frame);
                top_container.itk  = frame;
                top_container.configure(image=frame);
                top_container.after(10, new_face);

        #Other functions
        def auto_cap():
            try:
                if new_face_flag:
                    if text_entry.get()=="":
                        print("Please type your name in the text field");
                    else:
                        if not os.path.isdir('dataset'):
                            os.mkdir('dataset');
                        directory = os.path.sep.join(['dataset',text_entry.get()]);
                        if not os.path.isdir(directory):
                            os.mkdir(directory);
                        total = len(list(paths.list_images(directory)));
                        # if total >= 50:
                        #     print("There is plenty of images for you in the database");
                        #     return;
                        os.chdir(directory);
                        l = list(paths.list_images('.'));
                        l.sort();
                        count = 0;
                        for i in l:
                            os.rename(i, str(count).zfill(5)+".png");
                            count+=1;
                        os.chdir('../..');
                        print("Ready!");
                        time.sleep(1);
                        print("Set!");
                        time.sleep(1);
                        print("Action!");
                        time.sleep(1);
                        total = total % 50; # to update face images
                        while total <= 50:
                            time.sleep(0.5);
                            ret, frame = cap.read();
                            frame = imutils.resize(frame, width=400);
                            rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30,30));
                            if len(rects)==1:
                                cv2.imwrite(os.path.sep.join([directory, "{}.png".format(str(total).zfill(5))]), frame);
                                print("Imaging progress: {}/{} images".format(total,50));
                                total+=1;
                            # time.sleep(0.5);
                        print("Finished!");
            except:
                pass;
                            
                    
        def start_thread():
            threading.Thread(target=auto_cap).start();

        def train():
            if new_face_flag:
                imagepaths = list(paths.list_images('dataset/'));
                known_names = [];
                known_encodings = [];
                for (count, ip) in enumerate(imagepaths):
                    print("encoding images {} / {}".format(count + 1, len(imagepaths)));
                    name = ip.split(os.path.sep)[-2];
                    image = cv2.imread(ip);
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);

                    box = face_recognition.face_locations(rgb, model="hog");
                    encodings = face_recognition.face_encodings(rgb, box);

                    for encoding in encodings:
                        known_encodings.append(encoding);
                        known_names.append(name);

                print("writing to the disk");
                data = {"names": known_names, "encodings": known_encodings};
                f = open("encodings.pickle", "wb");
                f.write(pickle.dumps(data));
                f.close();
                print("finished!");
            
        # Main Function
        root = tk.Tk();
        root.title("Face Recognition");
        root.attributes("-type","dialog");
        top_container = tk.Label(root, bg="grey");
        middle_container = tk.Label(root, bg="black");
        bottom_container = tk.Label(root, bg="black");
        button_stream = tk.Button(middle_container, text="Stream", command=display_action);
        button_recognize = tk.Button(middle_container, text="Recognize", command=recognize_action);
        button_new_face = tk.Button(middle_container, text="New Face", command=new_face_action);
        text_entry = Entry(bottom_container, text="");
        button_capture = tk.Button(bottom_container, text="Capture", command=start_thread);
        button_train = tk.Button(bottom_container, text="Train", command=train);
        button_stream.pack(side="left", padx=(0,20));
        button_recognize.pack(side="left", padx=(0,20));
        button_new_face.pack(side="left", padx=(0,20));
        text_entry.pack(side="left", padx=(0,20), pady=(10,0));
        button_capture.pack(side="left", padx=(0,20), pady=(10,0));
        button_train.pack(side="left", padx=(0,20), pady=(10,0));
        top_container.pack(expand=1, fill=tk.BOTH);
        middle_container.pack(fill=tk.X);
        display_action();
        root.mainloop();
        cap.release();
        
    except:
        print("Failed to execute the program");

if __name__ == "__main__":
    faceRecognition();