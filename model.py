import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, Input, Lambda, Cropping2D
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import csv
import numpy as np
import PIL
from PIL import Image
import glob
import random

#not used
def get_image(image_path):
    image = Image.open(image_path)

    return image.convert('RGB')

features = []
targets = []

data_base_directory = '../car_simulator_training'

def local_image_path(image_path):
    filename = image_path.split('/')[-1]

    return data_base_directory + '/IMG/' + filename

sampled_rows = []
with open(data_base_directory + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        center, left, right, steering_angle, throttle, _, speed = line
        sampled_rows.append(line)

random.shuffle(sampled_rows)
valid_start_index = int(len(sampled_rows) * 0.9)
train_rows = sampled_rows[0:valid_start_index]
valid_rows = sampled_rows[valid_start_index:-1]
test_rows = valid_rows[len(valid_rows) // 2:-1]
valid_rows = valid_rows[0:len(valid_rows) // 2]


def get_images(samples, batch_size=36):
    features.clear()
    targets.clear()
    while True:
        random.shuffle(samples)
        for line in samples:
            center, left, right, steering_angle, throttle, _, speed = line
            steering_angle = float(steering_angle)


            # fix overrepresentation of small steering angles
            # by strongly favoring images with angles, with a small leak in the filter
            if np.random.random() > (np.power(np.abs(steering_angle), 1.4) + 0.003):
                continue
    
            try:
                center_image = get_image(local_image_path(center))
                left_image = get_image(local_image_path(left))
                right_image = get_image(local_image_path(right))
    
            except Exception as e:
              print(e)
            
            else:
                #center image
                features.append(np.array(center_image).astype(np.float32))
                targets.append(steering_angle)
                flipped_center = center_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                features.append(np.array(flipped_center).astype(np.float32))
                targets.append(-steering_angle)

                #add in occasional images from the side cameras
                if np.random.random() < 0.1:    
                    #left image
                    features.append(np.array(left_image).astype(np.float32))
                    targets.append(steering_angle + 0.25)
                    features.append(np.array(left_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)).astype(np.float32))
                    targets.append(steering_angle - 0.25)
        
                    #right image
                    features.append(np.array(right_image).astype(np.float32))
                    targets.append(steering_angle - 0.25)
                    features.append(np.array(right_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)).astype(np.float32))
                    targets.append(steering_angle + 0.25)
    
            if len(features) >= batch_size:
                output = (np.array(features), np.array(targets))
                yield output
                features.clear()
                targets.clear()

    
    return None
 
shape = (160, 320, 3)

batch_size = 36
batches_per_epoch = 100 #use an arbitrary value since I am using a generator which can run indefinitely
epochs = 25

model = Sequential()

def preprocess(x):
    #importing inside the function allows lambdas to be serialized
    import tensorflow as tf

    normalized = x / 255.
    hsv = tf.image.rgb_to_hsv(normalized)
    h_channel = tf.slice(hsv, [0, 0, 0, 0], [-1, -1, -1, 1])
    s_channel = tf.slice(hsv, [0, 0, 0, 1], [-1, -1, -1, 1])
    v_channel = tf.slice(hsv, [0, 0, 0, 2], [-1, -1, -1, 1])

    return h_channel

    # the below was an experiment in setting up dedicated channels
    # for lane lines and dirt.
    # I got better (and faster) performance with just using the h channel above.

    # extract colors of lines and boundary dirt
    true_matrix = tf.ones_like(h_channel)
    false_matrix = tf.zeros_like(h_channel)
    yellow_center = 58. / 255.
    red_max = 15. / 255.
    brown_center = 40. / 255.

    yellow_matches = tf.less(tf.abs(h_channel - yellow_center), 3. / 255.)
    red_matches = tf.less(h_channel, red_max)
    brown_matches = tf.less(tf.abs(h_channel - brown_center), 5. / 255.)
    is_black = tf.less(v_channel, 2. / 255.)

    yellow_channel = tf.where(yellow_matches, true_matrix, false_matrix) - 0.5
    red_channel = tf.where(red_matches, true_matrix, false_matrix) - 0.5
    brown_channel = tf.where(red_matches, true_matrix, false_matrix) - 0.5
    black_channel = tf.where(is_black, true_matrix, false_matrix) - 0.5

    # this code returned exclusively matching pixels for brown, yellow, and red.
    # the performance was no better.
    #colors_match = tf.logical_or(is_black, tf.logical_or(brown_matches, tf.logical_or(yellow_matches, red_matches)))
    #return tf.where(colors_match, true_matrix, false_matrix) - 0.5

    with_extra_channels = tf.concat([normalized, s_channel, yellow_channel, red_channel, brown_channel, black_channel], 3)

    return with_extra_channels

model.add(Lambda(preprocess, input_shape=(160, 320, 3)))

# remove the top 65 pixels from each image
model.add(Cropping2D(cropping=((65, 0), (0, 0))))

model.add(Conv2D(8, 7, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(16, 7, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32, 5, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, 5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Conv2D(64, 3, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Conv2D(64, 1, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Flatten())

# this fully-connected layer did not improve model performance, so I removed it
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# using a recurrent layer did not work well, because
# it biased the network toward straight driving.
# To use it, randomization must be removed in the training data.
#model.add(Reshape((1, 32)))
#model.add(LSTM(32, input_shape=(1,32), return_sequences=True))
#model.add(LSTM(32, return_sequences=True))
#model.add(LSTM(32))
#model.add(Dense(32))
#model.add(Activation('relu'))

model.add(Dense(1))

optimizer = Adam(lr=0.0001, decay=0.01)
model.compile(optimizer, 'mean_squared_error', ['accuracy'])

# model.fit() requires all of the data to be in memory, which didn't work as I added training data
#model.fit(features, targets, validation_split=0.2, epochs=epochs, shuffle=True, batch_size=batch_size)

# saving after each epoch allowed me to test the model regularly during training
# to verify improvement and stop early.
checkpointer = ModelCheckpoint(filepath='model-live.h5', verbose=1)
model.fit_generator(get_images(train_rows, batch_size), steps_per_epoch=batches_per_epoch, epochs=epochs, validation_data=get_images(valid_rows, batch_size), validation_steps=5, callbacks=[checkpointer])

# evaluate on test set
test_x = []
test_y = []
test_generator = get_images(test_rows)
for i in range(10):
    batch_x, batch_y = next(test_generator)

    test_x.extend(batch_x)
    test_y.extend(batch_y)

# I rarely got to this point, since I was testing directly on the simulator.
test_loss = model.evaluate(np.array(test_x), np.array(test_y))
print("Test loss: {}".format(test_loss))

model.save('model.h5')
