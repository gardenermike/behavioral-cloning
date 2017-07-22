import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, Input, Lambda, Cropping2D, GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D, ZeroPadding2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply
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
import cv2

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image


#not used
def get_image(image_path):
    image = Image.open(image_path)

    return image.convert('RGB')

features = []
targets = []

data_base_directory = '../track_2_2'
#data_base_directory = '../car_training_2'

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

def add_random_shadow(image):
    x_size = 320
    y_size = 160

    left = np.random.random() * x_size
    right = np.random.random() * x_size

    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:,:,1]
    y_array, x_array = np.mgrid[0:y_size, 0:x_size]
    shadow_mask[x_array * (left - right) - y_size * (y_array - left) >= 0] = 1

    brightness_threshold = 0.5
    random_bright = 1. - (np.random.random() * brightness_threshold)

    mask_true = shadow_mask == 1
    mask_false = shadow_mask == 1
    if np.random.random() > 0.5:
        condition = mask_true
    else:
        condition = mask_false

    image_hls[:,:,1][condition] = image_hls[:,:,1][condition] * random_bright

    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

def add_random_shear(image):
    x_size = 320
    y_size = 160
    x_center = x_size // 2

    pts1 = np.float32([[0, y_size], [x_size, y_size], [x_center, 0]])

    rotation_angle = (np.random.random() * np.pi) - (np.pi / 2)
    delta_x = np.sin(rotation_angle) * y_size
    delta_y = np.cos(rotation_angle)
    pts2 = np.float32([[0, y_size], [x_size, y_size], [x_center + delta_x, delta_y]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (x_size, y_size))

    return (rotation_angle / 2, dst)

def get_images(samples, batch_size=36, augment=True):
    features.clear()
    targets.clear()
    while True:
        random.shuffle(samples)
        for line in samples:
            center, left, right, steering_angle, throttle, _, speed = line
            steering_angle = float(steering_angle)


            # fix overrepresentation of small steering angles
            # by strongly favoring images with angles, with a small leak in the filter
            #if augment and np.random.random() > (np.power(np.abs(steering_angle), 0.2) + 0.005):
            #    continue

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
                if augment:
                    flipped_center = center_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                    features.append(np.array(flipped_center).astype(np.float32))
                    targets.append(-steering_angle)
    
                    if np.random.random() < 0.2:
                        #left image
                        features.append(np.array(left_image).astype(np.float32))
                        targets.append(steering_angle + 0.25)
                        features.append(np.array(left_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)).astype(np.float32))
                        targets.append(-steering_angle - 0.25)
    
                        #right image
                        features.append(np.array(right_image).astype(np.float32))
                        targets.append(steering_angle - 0.25)
                        features.append(np.array(right_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)).astype(np.float32))
                        targets.append(-steering_angle + 0.25)

            if len(features) >= batch_size:
                with_shadows = features #[add_random_shadow(image) for image in features]
                for i in range(len(features)):
                    rotation_angle, sheared_image = add_random_shear(with_shadows[i])
                    with_shadows[i] = sheared_image
                    targets[i] = targets[i] + rotation_angle
                output = (np.array(with_shadows), np.array(targets))
                yield output
                features.clear()
                targets.clear()


    return None

#get_images()
#features = np.array(features)
#targets = np.array(targets)
#print("{} images, {} targets".format(len(features), len(targets)))

#shape = features[0].shape
shape = (160, 320, 3)

#file_count = len(glob.glob(data_base_directory + "/IMG/*.jpg"))
batch_size = 36
#images_per_row = 6
batches_per_epoch = 100 #use an arbitrary value now that using a generator #len(train_rows) * images_per_row // batch_size
#validation_batches_per_epoch = len(valid_rows) * images_per_row // batch_size
epochs = 50

def preprocess(x):
    #importing inside the function allows lambdas to be serialized
    import tensorflow as tf

    normalized = x / 255.

    #return tf.image.resize_images(normalized, (224, 224)) - 0.5

    #normalized = tf.image.resize_images(normalized, (224, 224))

    #added = tf.reduce_sum(normalized, axis=3, keep_dims=True)
    #return added / tf.reduce_max(added)

    hsv = tf.image.rgb_to_hsv(normalized)

    return tf.concat([normalized, hsv], 3) - 0.5

    h_channel = tf.slice(hsv, [0, 0, 0, 0], [-1, -1, -1, 1])
    s_channel = tf.slice(hsv, [0, 0, 0, 1], [-1, -1, -1, 1])
    v_channel = tf.slice(hsv, [0, 0, 0, 2], [-1, -1, -1, 1])

    false_matrix = tf.zeros_like(h_channel)
    true_matrix = tf.ones_like(h_channel)

    v_channel_thresholded = tf.greater(v_channel, 1. / 255. * 30.)
    v_channel_with_threshold = tf.where(v_channel_thresholded, v_channel, false_matrix)

    r_channel = tf.slice(normalized, [0, 0, 0, 0], [-1, -1, -1, 1])
    g_channel = tf.slice(normalized, [0, 0, 0, 1], [-1, -1, -1, 1])
    b_channel = tf.slice(normalized, [0, 0, 0, 2], [-1, -1, -1, 1])
    #green_heavy = tf.logical_and(tf.greater(g_channel, r_channel), tf.greater(g_channel, b_channel))
    #green_mask = tf.where(green_heavy, false_matrix, true_matrix)
    green_heavy = tf.logical_and(tf.logical_and(tf.greater(g_channel, r_channel), tf.greater(g_channel, b_channel)), tf.less(v_channel, 75. / 255.))
    green_mask = tf.where(green_heavy, true_matrix / 2., true_matrix)
    v_channel_minus_green = tf.multiply(v_channel_with_threshold, green_mask)

    max_value = tf.reduce_max(v_channel_minus_green)
    v_channel_minus_green = v_channel_minus_green / max_value

    return tf.concat([v_channel_minus_green, s_channel, h_channel, r_channel, g_channel, b_channel, v_channel, v_channel_with_threshold], 3) - 0.5

    return v_channel

    #return tf.concat([h_channel, s_channel, v_channel], 3)
    #return s_channel

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

    #colors_match = tf.logical_or(is_black, tf.logical_or(brown_matches, tf.logical_or(yellow_matches, red_matches)))

    #return tf.where(colors_match, true_matrix, false_matrix) - 0.5

    with_extra_channels = tf.concat([normalized, s_channel, yellow_channel, red_channel, brown_channel, black_channel], 3)

    return with_extra_channels

inputs = Input(shape=(160, 320, 3))
preprocessed = Lambda(preprocess, input_shape=(160, 320, 3))(inputs)
y_crop_top_pixels = 100
y_crop_bottom_pixels = 30
x_crop_pixels = 90
cropped = Cropping2D(cropping=((y_crop_top_pixels, y_crop_bottom_pixels), (x_crop_pixels, x_crop_pixels)))(preprocessed)
def stretch_y(x):
    import tensorflow as tf
    return tf.image.resize_images(x, (140, 224))
#cropped = Lambda(stretch_y)(cropped)

#preprocessed = Lambda(preprocess)(inputs)

#resized = ZeroPadding2D(padding=((224 - 160 + y_crop_top_pixels + y_crop_bottom_pixels) // 2, 0))(cropped)

"""
base_model = InceptionV3(input_tensor=preprocessed, weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='elu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='elu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='elu')(x)
x = Dense(32, activation='elu')(x)
x = Dense(1)(x)
"""

# apply a soft attention layer
#flattened_cropped = Flatten()(cropped)
#attention_probabilities = Dense((160 - y_crop_top_pixels) * 320, activation='softmax', name='attention_probs')(flattened_cropped)
#attention_multiplied = Multiply()([flattened_cropped, attention_probabilities])
#attention_applied = Reshape((160 - y_crop_top_pixels, 320, 1))(attention_multiplied)
#attention_applied = BatchNormalization()(attention_applied)


conv1 = Conv2D(8, 7, padding='same', activation='elu')(cropped)
#conv1 = SeparableConv2D(8, 7, padding='same', activation='elu')(cropped)
conv1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

#conv2 = Conv2D(16, 7, padding='same', activation='elu')(conv1)
conv2 = SeparableConv2D(16, 7, padding='same', activation='elu')(conv1)
conv2 = MaxPooling2D(pool_size=(1, 2), padding='same')(conv2)

#conv3 = Conv2D(32, 5, padding='same', activation='elu')(conv2)
conv3 = SeparableConv2D(32, 5, padding='same', activation='elu')(conv2)
#conv3 = MaxPooling2D(pool_size=(1, 2),  padding='same')(conv3)

#conv4 = Conv2D(64, 5, padding='same')(conv3)
conv4 = SeparableConv2D(64, 5, padding='same')(conv3)
#conv4 = BatchNormalization()(conv4)
conv4 = Activation('elu')(conv4)
conv4 = Dropout(0.3)(conv4)

#conv5 = Conv2D(64, 3, padding='same')(conv4)
conv5 = SeparableConv2D(64, 3, padding='same')(conv4)
#conv5 = BatchNormalization()(conv5)
conv5 = Activation('elu')(conv5)
conv5 = MaxPooling2D(pool_size=(1, 2), padding='same')(conv5)
conv5 = Dropout(0.3)(conv5)

conv6 = Conv2D(64, 3, padding='same', activation='elu')(conv5)
for i in range(4):
    conv6 = Conv2D(64, 3, padding='same', activation='elu')(conv6)
    conv6 = Dropout(0.2)(conv6)

#conv6 = Conv2D(64, 1, padding='same')(conv5)
conv7 = SeparableConv2D(64, 1, padding='same')(conv6)
conv7 = BatchNormalization()(conv7)
conv7 = Activation('elu')(conv7)
conv7 = Dropout(0.2)(conv7)

conv8 = SeparableConv2D(64, 1, padding='same')(conv7)
conv8 = BatchNormalization()(conv8)
conv8 = Activation('elu')(conv8)
conv8 = Dropout(0.2)(conv8)

flattened = Flatten()(conv8)

# this may need to be regularized more or removed again
dense0 = Dense(192, activation='elu')(flattened)
dense0 = Dropout(0.2)(dense0)

dense1 = Dense(64)(dense0)
#dense1 = BatchNormalization()(dense1)
dense1 = Activation('elu')(dense1)
dense1 = Dropout(0.2)(dense1)

dense2 = Dense(32, activation='elu')(dense1)
dense2 = Dropout(0.2)(dense2)

dense3 = Dense(16, activation='elu')(dense2)
dense3 = Dropout(0.2)(dense3)

dense4 = Dense(8, activation='elu')(dense3)

#output = Dense(1, activation='tanh')(dense2)
output = Dense(1)(dense4)

#model = Model(inputs=inputs, outputs=output)
model = Model(inputs=inputs, outputs=output)

# TODO: train your model here
optimizer = Adam(lr=0.0001, decay=0.01)
model.compile(optimizer, 'mean_squared_error', ['accuracy'])
#model.fit(features, targets, validation_split=0.2, epochs=epochs, shuffle=True, batch_size=batch_size)

checkpointer = ModelCheckpoint(filepath='model-live.h5', verbose=1)
model.fit_generator(get_images(train_rows, batch_size), steps_per_epoch=batches_per_epoch, epochs=epochs, validation_data=get_images(valid_rows, batch_size, augment=False), validation_steps=10, callbacks=[checkpointer])

# evaluate on test set
test_x = []
test_y = []
test_generator = get_images(test_rows, augment=False)
for i in range(10):
    batch_x, batch_y = next(test_generator)

    test_x.extend(batch_x)
    test_y.extend(batch_y)

test_loss = model.evaluate(np.array(test_x), np.array(test_y))
print("Test loss: {}".format(test_loss))

model.save('model.h5')
