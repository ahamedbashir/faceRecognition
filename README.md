# Face Recognition
> A team project built for Senior Project I II by  
- Bashir
- Shofi
- Ucha  

## Instructions: 
### To Build Dataset Manually:
The following command will take images using system camera:  
```
python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/personName
```
Where '--output' is the argument to the path to which the newly taken images will be saved.  

### To Encode Images:
```
python encode_faces.py --dataset dataset --encodings encodings.pickle --model cnn
```
Where '--dataset' is the path to all images '--encodings' is the name of the encoded file will be, and '--model' is which model to use

### To Test Images Saved Locally:
```
python recognize_faces_image.py --encodings encodings.pickle --image testExamples/image.png
```
Where '--encodings' is tells which encoding file to use, and '--image' tells to to which the app should test  
