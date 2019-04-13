import math
import numpy
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import sys
import cv2
from face_recognition.cli import image_files_in_folder



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict(X_img_path, knn_clf=None, model_path=None,threshold=0.50):
   
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Loading KNN model 
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Loading image file
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img,number_of_times_to_upsample=2)

    
    if len(X_face_locations) == 0:
        return []

    # Finding face
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= threshold for i in range(len(X_face_locations))]
  
    #a[] = float((1-closest_distances[0])*100);
    #print((1-closest_distances[0])*100);
    
    

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    


def show_prediction_labels_on_image(img_path, predictions,image_file):
   
    pil_image = Image.open(img_path).convert("RGB")
    # a=cv2.imread(pil_image)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        
        name = name.encode("UTF-8")

        
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        

    
    del draw

    
    #pil_image.show()
    #cv2.imshow('image',a)
    pil_image.save(os.path.join('static/imagek',image_file))
    
    



