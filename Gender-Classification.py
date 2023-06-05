#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import zipfile 
import pandas as pd


# In[2]:


os.listdir('Male-Female face dataset')


# In[3]:


os.listdir('Male-Female face dataset\\Male Faces')


# In[4]:


os.listdir('Male-Female face dataset\\Female Faces')


# In[5]:


len(os.listdir('Male-Female face dataset\\Male Faces'))


# In[6]:


len(os.listdir('Male-Female face dataset\\Female Faces'))


# In[7]:


from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
img = load_img('Male-Female face dataset\\Female Faces\\0 (1282).jpg')
img


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join('Male-Female face dataset')
path


# In[9]:


batch_size =100
epochs = 10
IMG_HIEGHT = 150
IMG_WIDHT = 150


# In[10]:


image_generator = ImageDataGenerator(rescale=1./255)
img_iter = image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=path,
                                                           shuffle=True,
                                                           target_size=(IMG_HIEGHT,IMG_WIDHT),
                                                           class_mode="binary",
                                                            )


# In[11]:


img_iter.class_indices


# In[12]:


sample_image,labels = next(img_iter)
sample_image.shape


# In[13]:


labels


# In[14]:


sample_image


# In[15]:


plt.imshow(sample_image[0])


# In[16]:


def plotImages(images_arr):
    fig, axes = plt.subplots(2,8,figsize=(10,10))
    axes = axes.flatten()
    for img,ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[17]:


plotImages(sample_image[0:20])


# In[18]:


model = Sequential([
    Conv2D(15,3,padding='same',activation='relu',input_shape=(IMG_HIEGHT,IMG_WIDHT,3)), #150
    MaxPooling2D(), #75
    Conv2D(30,3,padding='same',activation='relu'), #75
    MaxPooling2D(),#37
    Conv2D(30,3,padding='same',activation='relu'),#37
    MaxPooling2D(),#18
    Flatten(),
    Dense(512,activation='relu'),
    Dense(1,activation='sigmoid')
])


# In[19]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[20]:


history = model.fit(img_iter,epochs=epochs)


# In[21]:


acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(epochs)
plt.figure(figsize=(4,4))
plt.plot(epochs_range,acc,label='Accuracy')
plt.plot(epochs_range,loss,label='Loss')
plt.legend(loc=0)
plt.show()


# In[22]:


test_img1 = load_img('male1.jpg',target_size=(IMG_HIEGHT,IMG_WIDHT))
test_img1


# In[24]:


test_array1 = img_to_array(test_img1)
test_array1 = test_array1.reshape(1,IMG_HIEGHT,IMG_WIDHT,3)

model.predict(test_array1)


# In[25]:


out1=model.predict(test_array1)


# In[26]:


int(out1[0][0])


# In[ ]:





# In[27]:


#detect gender
if int(out1[0][0])==0: 
    print("gender is Female")
else:
    print("gender is Male")


# In[28]:


class_names=['Female','Male']
#testing cascad
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_cam = img
    test_cam = cv2.resize(test_cam,(IMG_HIEGHT,IMG_WIDHT))
    test_cam = test_cam.reshape(1,IMG_HIEGHT,IMG_WIDHT,3)
    #iden = []
    pred1 = model.predict(test_cam)
    #iden.append(int(pred[0][0]))
    print(pred1)
    

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print(faces.shape)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, str(class_names[int(pred1[0][0])]), (x+5,y-5), font, 1, (255,0,0), 4)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()


# In[29]:


model.save("gender_class.h5")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




