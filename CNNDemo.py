# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 22:08:33 2023

@author: joshu
"""

'''
   Copyright 2023 Joshua Henderson

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

#for more information on CNNs, you can reference my work:
    # https://link.springer.com/chapter/10.1007/978-3-031-23599-3_6
    # https://red.library.usd.edu/honors-thesis/254/

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##############################
# Get some data to work with
##############################
# https://www.kaggle.com/datasets/tongpython/cat-and-dog
# 1. Download the data from the link above and extract it to a local folder
# 2. You will need to rename the folders inside of the root folder to "Train" and "Test"
# 3. Then, within those folders, you'll notice there is only one folder and then the cats and dogs are embedded one folder layer deeper
#       You will need to move the cats and dogs folders from the level they're at to the "Train" and "Test" folders
# 4. Within those cats and dogs folders, there is one file that isn't an image, so you'll need to go and delete that too
#
# So... your end product will be within the root folder (stored in the variable direct) you'll have two folders: Train and Test
# within each of those folders, you'll have two more folders: cats and dogs (these can be named whatever)

direct = r"DIRECTORY_GOES_HERE" # Directory of root folder of dataset (split into Train and Test Folders within that directory)

train_dataset_dir = direct+"/Train"
test_dataset_dir  = direct+"/Test" # hidden from the model during training

# normalize the images as they are read in
# note: you can add other data augmentation functions to the ImageDataGenerator if you want
# Here is a page with what you can add to the ImageDataGenerator: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
train = ImageDataGenerator(rescale = 1/255)
test = ImageDataGenerator(rescale = 1/255)

# create a stream of images, resized to 256x256, grouped into batches of size 8, and classified into 2 categories
train_dataset = train.flow_from_directory(train_dataset_dir, target_size = (256,256), batch_size=8, class_mode='binary')
test_dataset  = test.flow_from_directory(test_dataset_dir, target_size = (256,256), batch_size=8, class_mode='binary')

##############################
# Create a model
##############################

model = Sequential() # create a new sequential model

# Add a 2D convolution layer with 16 features, a 3x3 filter size, relu activation, and padding
model.add(Conv2D(16, (3,3), activation='relu', padding = 'same', input_shape=(256,256,3)))
# Add a Max Pool layer with a 2x2 pooling window
model.add(MaxPool2D(2, 2))

# These are the same as above, except the convolution has 32 features
model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(2, 2))

# These are the same as above, except the convolution has 64 features
model.add(Conv2D(64, (3,3), activation='relu', padding = 'same'))
model.add(MaxPool2D(2, 2))

# These are the same as above, except the convolution has 128 features
model.add(Conv2D(128, (3,3), activation='relu', padding = 'same'))
model.add(MaxPool2D(2, 2))

# Flatten into a single vector
model.add(Flatten())

# Add a dense layer to 128 nodes (to extract relationships between features)
model.add(Dense(128))

# Add a dense layer to one node with a sigmoid function (to extract a prediction)
model.add(Dense(1, activation='sigmoid'))

# Compile the model using adam optimization, calculating loss using binary cross-entropy loss, and collect the metrics listed
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy', AUC(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])

# Print out a summary of the model
model.summary()

##############################
# Train the model
##############################
# Train the model on the training dataset over 10 epochs, using the test dataset to validate the results
model.fit(train_dataset, steps_per_epoch=len(train_dataset), epochs=10, validation_data=test_dataset)
