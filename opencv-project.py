import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
video_capture = cv2.VideoCapture(0)
skip = 0

face_data = []
dataset_path = './data/'
file_name = input('Enter the name of the person : ')


while video_capture.isOpened():
    _,img = video_capture.read()
    gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img,1.1,5)
    faces = sorted(faces,key=lambda x:x[2]*x[3],reverse=True)
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

        offset = 10
        face_section = img[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))


    cv2.imshow('Face',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

np.save(dataset_path+file_name+'.npy',face_data)
print('data successfully saved at ' + dataset_path+file_name+'.npy')

video_capture.release()
cv2.destroyAllWindows()

