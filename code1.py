import numpy
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im

import os
folder = r"D:\Actors"
files = os.listdir(folder)
files

len(files)

fd = cv2.CascadeClassifier(r"D:\kaizen_training\kaizen\haarcascade_frontalface_alt.xml")
def get_face(img):
    corners = fd.detectMultiScale(img,1.3,4)
    if len(corners)==0:
        return None,None
    else:
        (x,y,w,h) = corners[0]
        img = img[y:y+w,x:x+h] #cropping the image
        img = cv2.resize(img,(100,100))
        return(x,y,w,h),img
    

trainimg = []
trainlb = []
for filename in files:
    filepath = folder + "\\" + filename #path of subfolder
    
        
    files2 = os.listdir(filepath)
        
    for filename2 in files2:
        filepath2 = folder +"\\" + filename + "\\" + filename2
            
        img = im.imread(filepath2)
        corner,img = get_face(img)
        trainimg.append(img)
        trainlb.append(filename)
    

print(trainlb)



trainimg = numpy.array(trainimg)
trainlb = numpy.array(trainlb)
print(trainimg.shape)
print(trainlb.shape)

#preprocessing the image
#scale the images
trainimg = trainimg/255

#reshape the image data


#onehot encode the labels
from sklearn.preprocessing import OneHotEncoder
trainlb = OneHotEncoder().fit_transform(trainlb.reshape(25,1)).toarray()

print(trainimg.shape)
print(trainlb.shape)

#building the model using keras - CNN

from keras import models,layers
model = models.Sequential()
#add first convolutional and maxpooling layer
model.add(layers.Conv2D(filters=20,kernel_size=(3,3),
                       activation = 'relu', input_shape = (100,100,3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#add the second convolutional layer
model.add(layers.Conv2D(filters=40,kernel_size=(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#add the flatten layer
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(5,activation= 'softmax'))

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

#train the model
model.fit(trainimg,trainlb,epochs=10,batch_size=9,shuffle =True,verbose=True)



model.save("model.h5")

