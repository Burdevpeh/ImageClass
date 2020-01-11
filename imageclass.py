#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


fashion_mnist = keras.datasets.fashion_mnist


# In[3]:


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[4]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[5]:


len(train_labels)


# In[6]:



test_images.shape
len(test_labels)


# In[7]:


train_images = train_images / 255.0


# In[8]:


test_images = test_images / 255.0


# In[19]:


plt.figure(figsize=(50,50))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]]) // view with class labels
plt.show()


# In[20]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[21]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[22]:


model.fit(train_images, train_labels, epochs=10)


# In[23]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print("Test loss is", test_loss)


# In[57]:


predictions = model.predict(test_images)
for i in range(9999):
    print(predictions[i])
for i in range(num_images):
    plot_value_array(i, predictions[i], test_labels)

