from tensorflow.keras.models import load_model
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
model = load_model('landmark-model.h5')
vid = cv2.VideoCapture(0) 

def detect_points(face_img):
        me  = np.array(face_img)/255
        x_test = np.expand_dims(me, axis=0)
        x_test = np.expand_dims(x_test, axis=3)

        y_test = model.predict(x_test)
        label_points = (y_test*50)+100
        return label_points

  
while(True): 
      
    ret, img  = vid.read() 
    m,n,o = img.shape
    rgb  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi = gray[y-70:y+h+70, x-70:x+w+70]
        org_shape = roi.shape
        roi_copy = cv2.resize(roi,(224,224))
        predicted_key_pts = detect_points(roi_copy)
        predicted_key_pts = predicted_key_pts.astype(int).reshape(-1,2)
        predicted_key_pts[:, 0] = predicted_key_pts[:, 0] * org_shape[0] / 224 + x-70
        predicted_key_pts[:, 1] = predicted_key_pts[:, 1] * org_shape[1] / 224 + y-70

                # cv2.rectangle(image_with_detections_1, (x, y), (x + w, y + h), (0, 0, 255), 3)

        for (x_point, y_point) in zip(predicted_key_pts[:, 0], predicted_key_pts[:, 1]):
            cv2.circle(img, (x_point, y_point), 3, (0, 255, 0), -1)


    
    cv2.imshow('img', img)
    try:
        img2 = np.zeros((480,640,1))
        for (x_point, y_point) in zip(predicted_key_pts[:, 0], predicted_key_pts[:, 1]):
                cv2.circle(img2, (x_point, y_point), 3, (255, 255, 225), -1)
        
        cv2.imshow('img2',img2)
    except:
            continue                             

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

