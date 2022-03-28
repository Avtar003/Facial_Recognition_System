# import cv2
#
# def draw_boundary(img,classifier,scaleFactor,minNeighbour,color,text):
#     gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = classifier.detectMultipleScale(gray_image,scaleFactor,minNeighbour)
#     cords = []
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
#         cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
#         cords = [x,y,w,h]
#     return cords,img
#
#
# def detect(img,faceCascade):
#     cords,img = draw_boundary(img,faceCascade,1.1,4,(117, 13, 34),'Face')
#     return img
#
#
# video_capture = cv2.VideoCapture(0)
# faceCascade = cv2.CascadeClassifier("D:\yadav\Coding Blocks\opencv\haarcascade_frontalface_alt")
# # face_cascade = cv2.CascadeClassifier("D:\yadav\Coding Blocks\opencv\haarcascade_frontalface_alt")
#
# while True:
#     rect,img = video_capture.read()
#     img = detect(img,faceCascade)
#     cv2.imshow('Face', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()


# import cv2
#
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
# video_capture = cv2.VideoCapture(0)
#
# while video_capture.isOpened():
#     _,img = video_capture.read()
#     gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(img,1.1,5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(90, 204, 95),2)
#     cv2.imshow("Face",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()

# import cv2
#
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
# video_capture = cv2.VideoCapture(0)
#
# while video_capture.isOpened():
#     _,img = video_capture.read()
#     gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray_frame,1.1,5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(68, 212, 219),2)
#     cv2.imshow("Faces",gray_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
video_capture = cv2.VideoCapture(0)
skip = 0
#face_section = 0
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
    #cv2.imshow('Section',face_section)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

np.save(dataset_path+file_name+'.npy',face_data)
print('data successfully saved at ' + dataset_path+file_name+'.npy')

video_capture.release()
cv2.destroyAllWindows()

