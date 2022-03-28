import numpy as np
import cv2
import os

def distance(x,y):
    return np.sqrt(((x-y)**2).sum())

def knn(train,test,k=7):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i,-1]
        d = distance(test,ix)
        dist.append([d,iy])
    dk = sorted(dist,key=lambda x:x[0])[:k]
    labels = np.array(dk)[:,-1]
    output = np.unique(labels,return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]


videp_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
face_data = []
dataset_path = './data/'
skip = 0
labels = []
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
trainset = np.concatenate((face_dataset,face_labels),axis=1)



while True:
    ret,img = videp_capture.read()
    if ret == False:
        continue
    faces = faceCascade.detectMultiScale(img,1.1,5)
    gray_scale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for face in faces:
        x,y,w,h = face
        offset = 10
        face_section = img[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        out = knn(trainset,face_section.flatten())
        prediction = names[int(out)]
        cv2.putText(img,prediction,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (156, 19, 117), 2)
    cv2.imshow('Faces',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videp_capture.release()
cv2.destroyAllWindows()


