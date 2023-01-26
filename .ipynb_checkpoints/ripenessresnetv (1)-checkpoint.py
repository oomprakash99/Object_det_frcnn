import os
import matplotlib.pyplot as plt
from PIL import Image
import math
dir_example = "./Users/oomprakash/Desktop/python"
classes = os.listdir(dir_example)
print(classes)


# In[15]:


dir_example = "./Users/oomprakash/Desktop/python/train2014"

classes = os.listdir(dir_example)
print(classes)


# In[16]:


dir_example = "./Users/oomprakash/Desktop/python/test"
classes = os.listdir(dir_example)
print(classes)


# In[17]:


train = "./Users/oomprakash/Desktop/python/train2014"


# In[18]:


test = "./Users/oomprakash/Desktop/python/test"


# In[19]:


image1 = Image.open( './Users/oomprakash/Desktop/python/train2014/COCO_train2014_000000000081.jpg')


# In[20]:


plt.imshow(image1)


# In[ ]:





# In[21]:


# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# Create a generator
train_datagen = ImageDataGenerator(
  rescale=1./255
)
train_datagen = train_datagen.flow_from_directory(
        train,
        batch_size=32,
        target_size=(300, 300),
        class_mode='sparse')

#Printing the training set
labels = (train_datagen.class_indices)
print(labels,'\n')


# In[9]:


# Create a generator
test_datagen = ImageDataGenerator(
  rescale=1./255
)
test_datagen = test_datagen.flow_from_directory(
        test,
        batch_size=32,
        target_size=(300, 300),
        class_mode='sparse')

#Printing the test set
labels = (test_datagen.class_indices)
print(labels,'\n')


# In[132]:


for image_batch, label_batch in train_datagen:
  break
image_batch.shape, label_batch.shape


# In[133]:


for image_batch, label_batch in test_datagen:
  break
image_batch.shape, label_batch.shape


# In[137]:


model=Sequential()
#model.add(ResNet50(weights='imagenet'))
Convolution blocks
model.add(Conv2D(32, kernel_size = (3,3), padding='same',input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 

model.add(Conv2D(64, kernel_size = (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 

model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Flatten())
Classification layers

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
(/model.layers[0].layers)
model.summary()


# In[138]:


model.compile(optimizer = 'RMSprop', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[144]:


history = model.fit(train_datagen,epochs=100,steps_per_epoch=118//32)
# Get training and test loss histories
training_loss = history.history['loss']
training_accuracy=history.history['accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_accuracy) + 1)

# Visualize loss history
plt.plot(epoch_count, training_accuracy, 'r--')
plt.legend(['Training accurasy model:Resnet50 optimizer:RMSprop,relu'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();


# In[142]:


print(train_datagen.class_indices)
Labels = '\n'.join(sorted(train_datagen.class_indices.keys()))

with open('Labels.txt', 'w') as file:
  file.write(Labels)
class_names = list(labels.keys())


# In[143]:


from tensorflow.keras.preprocessing import image
import numpy as np
test_img = './Users/oomprakash/Desktop/python/test/COCO_train2014_000000000030.jpg'
img=image.load_img(test_img, target_size = (300,300))
img=image.img_to_array(img, dtype=np.uint8)
img=np.array(img)/255.0
prediction = model.predict(img[np.newaxis, ...])

print("Probability: ",np.max(prediction[0], axis=-1))
predicted_class = class_names[np.argmax(prediction[0], axis=-1)]
print("Classified: ",predicted_class,'\n')

plt.axis('off')
plt.imshow(img.squeeze())
plt.title("Loaded Image")


# In[121]:


from tensorflow.keras.preprocessing import image
import numpy as np
test_img = './imagecopies/14.jpg'
img=image.load_img(test_img, target_size = (300,300))
img=image.img_to_array(img, dtype=np.uint8)
img=np.array(img)/255.0
prediction = model.predict(img[np.newaxis, ...])

print("Probability: ",np.max(prediction[0], axis=-1))
predicted_class = class_names[np.argmax(prediction[0], axis=-1)]
print("Classified: ",predicted_class,'\n')

plt.axis('off')
plt.imshow(img.squeeze())
plt.title("Loaded Image")


# In[ ]:





# In[ ]:




