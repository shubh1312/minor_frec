import cv2
import numpy
import os, glob

#Cropped_Folder = os.path.basename('cropped')
path = os.path.abspath('.')
fpath = os.path.join(path, 'static','cropped')
def fdec(image,crp_name):

    res = os.path.abspath('./static/cropped')
    if glob.glob1(res, '*.jpg') != []:
        for fil in glob.glob1(res, '*.jpg'):
            print(os.path.join(res, fil))
            os.remove(os.path.join(res, fil))

    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(image)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    faces=face_cascade.detectMultiScale\
        (gray_img,
         scaleFactor=1.25,
         minNeighbors=5)
    #print(faces)
    count=1
    for x,y,w,h in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,226,0),1)
        crop_img = img[y:y + h, x:x + w]
        #print(crop_img)
        #cv2.imshow('single_faces',crop_img)
        #cv2.waitKey(2000)
        #print(os.path.join(fpath,'face'+str(count)+'.jpg'))
        cv2.imwrite(os.path.join(fpath,crp_name+str(count)+'.jpg'),crop_img)
        count+=1

    #cv2.imshow('faces',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return 0
