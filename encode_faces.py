from imutils import paths
import argparse
import os
import cv2
import face_recognition
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to dataset")
ap.add_argument("-m", "--model", required=True,
                help="Model to locate faces")
ap.add_argument("-e", "--encodings", required=True,
                help="Data file to write to disk")
args = ap.parse_args()

imagepaths = list(paths.list_images(args.dataset))

known_names = []
known_encodings = []
for (count, ip) in enumerate(imagepaths):
    print("encoding images {} / {}".format(count + 1, len(imagepaths)))
    name = ip.split(os.path.sep)[-2]
    image = cv2.imread(ip)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=args.model)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

print("writing to the disk")
data = {"names": known_names, "encodings": known_encodings}
f = open(args.encodings, "wb")
f.write(pickle.dumps(data))
f.close()
print("finished!")
