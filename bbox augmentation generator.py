import xmltodict
import os
import glob
import xmltodict
import xlwt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import numpy as np
import random
from random import randint
import PIL
from PIL import Image
import cv2
import xlrd
import pandas as pd
import math
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D
from keras.models import load_model
from datetime import datetime
import time

count2 = 0
datagen = ImageDataGenerator()
xlfilepath = "E:/machine learning/combineAug/combineAug.xls"
df = pd.read_excel(xlfilepath)
df_list = df.to_numpy()
file_list = np.delete(df_list, 0, 0)
np.random.shuffle(file_list)
shape = np.shape(file_list)
rows = shape[0]
num_list = [None] * rows

for i in range(rows):
    num_list = i
count = 0
limit = int(rows / 64)


def debug_func(img_arr, xmin, ymin, xmax, ymax):
    S = np.shape(img_arr)
    height = S[0]
    width = S[1]
    x_scale = width / 224
    y_scale = height / 224
    im = Image.fromarray(img_arr)
    im2 = im.resize((224, 224))
    im3 = np.array(im2)
    xmin = xmin * x_scale
    ymin = ymin * y_scale
    xmax = xmax * x_scale
    ymax = ymax * y_scale

    return im3, xmin, ymin, xmax, ymax


def image_generator(xlfilepath, batch_size=64):
    global file_list
    global count
    global count2
    global rows
    augment = False
    while True:  # Select files (paths/indices) for the batch
        if count >= (rows - (batch_size + 1)):
            count = randint(0, batch_size)

        batch_path_indexes = np.random.choice(a=num_list, size=batch_size, replace=False)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for index in range(batch_size):
            number = randint(0, 100)
            if number < 10:
                augment = True
            #augment=False
            input, output = get_in_out(file_list[index + count, 0], file_list[index + count, 1],
                                       file_list[index + count, 2],
                                       file_list[index + count, 3], file_list[index + count, 4],
                                       [224, 224], augment)
            augment = False
            batch_input += [input]
            batch_output += [output]  #
            # Return a tuple of (input, output) to feed the network
        count = count + batch_size
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        yield batch_x, batch_y


# get image path augment image, get new points and preprocess the image
def get_in_out(img_path, xmin, ymin, xmax, ymax, target_size, augment):
    im = Image.open(img_path)
    # if augment not
    if augment == False:
        target_height = target_size[0]
        target_width = target_size[1]
        shape = np.shape(np.array(im))
        original_height = shape[0]
        original_width = shape[1]
        x_scale = target_height / original_height
        y_scale = target_width / original_width
        # determin new scale values
        xmin_scaled = xmin * x_scale
        ymin_scaled = ymin * y_scale
        xmax_scaled = xmax * x_scale
        ymax_scaled = ymax * y_scale
        # resize image
        im2 = im.resize((target_height, target_width))
        output_points = [xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled,0]
        im2 = im.resize((224, 224))
        im3=np.array(im2)/255
        return im3, output_points
    img_arr = np.array(im)
    new = aug(img_arr, xmin, ymin, xmax, ymax, target_size)
    new_img = new[0]
    new_points = new[1]
    xmin2 = new_points[0]
    ymin2 = new_points[1]
    xmax2 = new_points[2]
    ymax2 = new_points[3]

    xmin_scaled = xmin2
    ymin_scaled = ymin2
    xmax_scaled = xmax2
    ymax_scaled = ymax2

    if xmin_scaled <= 0:
        xmin_scaled = 0
    if ymin_scaled <= 0:
        ymin_scaled = 0
    if xmax_scaled > 1:
        xmax_scaled = .99
    if ymax_scaled > 1:
        ymax_scaled = .99
    new_img_processed = new_img / 255

    output_points = [xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled, 0]

    return new_img_processed, output_points


def translate(x, y, height, width):
    x = x + (width / 2)
    y = y + (height / 2)
    y = -1 * (y - height)
    return x, y


def max_exist(x, y, side):
    T = translate(x, y, 240, 320)
    A = x ** 2 + y ** 2
    # B = math.sqrt(A)
    C = side ** 2
    if A > C:
        return True
    else:
        return False


def angle_find_vectors(xa, ya, xb, yb):
    T = translate(xa, yb, 240, 320)
    A = [xa, ya]
    B = [xb, yb]
    # C = [abs(xa), abs(ya)]
    # D = [abs(xb), abs(yb)]
    C = math.sqrt((xa ** 2) + (ya ** 2))
    D = math.sqrt((xb ** 2) + (yb ** 2))

    F = np.dot(A, B)
    G = C * D
    # if for whateve reason it is one (outside the domain of acos)
    # return zero
    if abs(F / G) >= 1:
        return 0

    H = math.acos(F / G)
    return H


def angle_find2(x, y, side, hs, ws, height_width):
    T = translate(x, y, 240, 320)
    exist = False
    angle = 1.5708
    width = 1
    height = 1
    a = height_width
    exist = max_exist(x, y, side)
    if not exist:
        return angle

    one_side = x ** 2 + y ** 2
    side_sq = side ** 2
    other_side = math.sqrt(one_side - side_sq)

    if height_width == 0:
        width = ws * side
        height = other_side * hs
    elif height_width == 1:
        width = ws * other_side
        height = hs * side
    else:
        print("something went wrong")
    T2 = translate(width, height, 240, 320)
    angle = angle_find_vectors(x, y, width, height)
    return angle


# height width
def angle_find(x, y, height, width):
    T = translate(x, y, 240, 320)
    angle_clock = 1.5708
    angle_counter = 1.5708
    angle_temp = [1.5708, 1.5708, 1.5708, 1.5708]

    q = which_quadrant(x, y)
    if q == 1:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, 1, 1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, -1, -1, 1)
        angle_temp[3] = angle_find2(x, y, width, 1, -1, 0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    elif q == 2:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, 1, 1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, 1, 1, 1)
        angle_temp[3] = angle_find2(x, y, width, 1, -1, 0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    elif q == 3:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, -1, -1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, -1, -1, 1)
        angle_temp[3] = angle_find2(x, y, width, -1, 1, 0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    elif q == 4:
        angle_temp[0] = angle_find2(x, y, height, 1, -1, 1)
        angle_temp[1] = angle_find2(x, y, width, -1, -1, 0)
        angle_clock = min(angle_temp[0], angle_temp[1])

        angle_temp[2] = angle_find2(x, y, height, 1, 1, 1)
        angle_temp[3] = angle_find2(x, y, width, -1, 1, 0)
        angle_counter = min(angle_temp[2], angle_temp[3])
    return angle_clock, angle_counter


def which_quadrant(x, y):
    if x <= 0:
        if y > 0:
            return 1
        else:
            return 3
    if x < 0:
        if y > 0:
            return 2
        else:
            return 4
    else:
        return 1


def max_rotation(xmin, ymin, xmax, ymax, angle, height, width):
    # make all points relative to the center of rotation
    center_x = width / 2
    center_y = height / 2

    # rightsideup
    ymin0 = height - ymin
    ymax0 = height - ymax

    xmin_r = xmin - center_x
    ymin_r = ymin0 - center_y
    xmax_r = xmax - center_x
    ymax_r = ymax0 - center_y

    # create four points
    x1 = xmin_r
    x2 = xmin_r
    x3 = xmax_r
    x4 = xmax_r

    y1 = ymin_r
    y2 = ymax_r
    y3 = ymin_r
    y4 = ymax_r

    angle_clock = 1.57079632679
    angle_counter = 1.57079632679

    temp_clock, temp_counter = angle_find(x1, y1, center_y, center_x)
    angle_clock1 = min(angle_clock, temp_clock)
    angle_counter1 = min(angle_counter, temp_counter)

    temp_clock, temp_counter = angle_find(x2, y2, center_y, center_x)
    angle_clock2 = min(angle_clock1, temp_clock)
    angle_counter2 = min(angle_counter1, temp_counter)

    temp_clock, temp_counter = angle_find(x3, y3, center_y, center_x)
    angle_clock3 = min(angle_clock2, temp_clock)
    angle_counter3 = min(angle_counter2, temp_counter)

    temp_clock, temp_counter = angle_find(x4, y4, center_y, center_x)
    angle_clock4 = min(angle_clock3, temp_clock)
    angle_counter4 = min(angle_counter3, temp_counter)

    # convert from radians to degrees
    angle_clock_con = ((180 / 3.141592) * angle_clock4)
    angle_counter_con = ((-180 / 3.141592) * angle_counter4)

    angle_clock_i = int(angle_clock_con)
    angle_counter_i = int(angle_counter_con)

    return angle_clock_i, angle_counter_i


def rotation_transform(x, y, radian):
    y_transform = y * math.cos(radian) + -x * math.sin(radian)
    x_transform = y * math.sin(radian) + x * math.cos(radian)
    val = [x_transform, y_transform]
    return x_transform, y_transform


# finds new points after rotation transformation
def post_rotation_points(xmin, ymin, xmax, ymax, angle, height, width):
    center_x = width / 2
    center_y = height / 2

    # compensate for y
    ymin0 = height - ymin
    ymax0 = height - ymax
    # make all points relative to the center of rotation
    xmin_r = xmin - center_x
    ymin_r = ymin0 - center_y
    xmax_r = xmax - center_x
    ymax_r = ymax0 - center_y

    # create four points
    x1 = xmin_r
    x2 = xmin_r
    x3 = xmax_r
    x4 = xmax_r

    y1 = ymin_r
    y2 = ymax_r
    y3 = ymin_r
    y4 = ymax_r

    # convert the angle into radians
    radian = angle * (math.pi / 180)
    x1_r, y1_r, = rotation_transform(x1, y1, radian)
    x2_r, y2_r, = rotation_transform(x2, y2, radian)
    x3_r, y3_r, = rotation_transform(x3, y3, radian)
    x4_r, y4_r, = rotation_transform(x4, y4, radian)

    # compute the new points

    # take away relative position of center
    x1_a = x1_r + center_x
    x2_a = x2_r + center_x
    x3_a = x3_r + center_x
    x4_a = x4_r + center_x

    y1_a = y1_r + center_y
    y2_a = y2_r + center_y
    y3_a = y3_r + center_y
    y4_a = y4_r + center_y

    # recompensate the ys
    y1_b = -1 * (y1_a - height)
    y2_b = -1 * (y2_a - height)
    y3_b = -1 * (y3_a - height)
    y4_b = -1 * (y4_a - height)

    p1 = [x1_a, y1_b]
    p2 = [x2_a, y2_b]
    p3 = [x3_a, y3_b]
    p4 = [x4_a, y4_b]
    P = [p1, p2, p3, p4]

    xmin_a = int(min(x1_a, x2_a, x3_a, x4_a))
    ymin_a = int(min(y1_b, y2_b, y3_b, y4_b))
    xmax_a = int(max(x1_a, x2_a, x3_a, x4_a))
    ymax_a = int(max(y1_b, y2_b, y3_b, y4_b))
    out = [xmin_a, ymin_a, xmax_a, ymax_a]

    # return new values
    return xmin_a, ymin_a, xmax_a, ymax_a


# determines the zoom limit for a point to not be zoomed out
def zoom_point_limit(point, max_point):
    limit = .1
    distance_max = 1
    centerpoint = int(max_point / 2)
    # distance from center=0
    distance_center = abs(point - centerpoint)

    # putting in a certain tolerance point
    if limit < .5:
        limit = .5

    limit = (distance_center / centerpoint)
    return limit


# determines points after zoom
def new_zoom_points(point, zoom_level, max_point):
    distance_max = 1
    new_point = 1
    center_point = int(max_point / 2)
    # distance from center
    distance_center = abs(point - center_point)
    # if point is bellow the center
    if point < center_point:
        new_point = center_point - int((distance_center / zoom_level))
    # if the point is above center
    elif point > center_point:
        new_point = center_point + int((distance_center / zoom_level))
    else:
        new_point = center_point + int((distance_center / zoom_level))
    return new_point


def display(shift_img, xmin_shift, ymin_shift, xmax_shift, ymax_shift):
    xmin_shift = int(xmin_shift)
    ymin_shift = int(ymin_shift)
    xmax_shift = int(xmax_shift)
    ymax_shift = int(ymax_shift)
    S = np.shape(shift_img)
    height = S[0]
    width = S[1]

    U = np.zeros((height, width, 3))
    P = 0
    U[0:height, 0:width, 0:3] = shift_img[0:height, 0:width, 0:3]

    U[ymin_shift, xmin_shift] = [P, P, P]
    U[ymax_shift, xmax_shift] = [P, P, P]

    U[ymin_shift + 1, xmin_shift] = [P, P, P]
    U[ymax_shift + 1, xmax_shift] = [P, P, P]

    U[ymin_shift, xmin_shift + 1] = [P, P, P]
    U[ymax_shift, xmax_shift + 1] = [P, P, P]

    U[ymin_shift - 1, xmin_shift] = [P, P, P]
    U[ymax_shift - 1, xmax_shift] = [P, P, P]

    U[ymin_shift, xmin_shift - 1] = [P, P, P]
    U[ymax_shift, xmax_shift - 1] = [P, P, P]
    U = np.uint8(U)
    UU = Image.fromarray(U)
    UU.show()


def cutoff(img_arr, xmin, ymin, xmax, ymax):
    S = np.shape(img_arr)
    x_diff = xmax - xmin
    y_diff = ymax - ymin

    height = S[0]
    width = S[1]

    # take away chunk from pic
    img_chunked = np.zeros((height, width, 3))
    img_chunked[0:height, 0:width] = img_arr[0:height, 0:width]
    blank = np.zeros((ymax - ymin, xmax - xmin, 3))
    # blank out covered area
    img_chunked[ymin:ymax, xmin: xmax] = blank

    # take a random chunk from chunked
    y_limit = int(height - (ymax))
    x_limit = int(width - (xmax))

    random_y = randint(1, y_limit)
    random_x = randint(1, x_limit)

    random_chunk = img_chunked[random_y + ymin:random_y + ymax, random_x + xmin:random_x + xmax]
    # insert chunk in missing piece
    A = img_chunked[ymin:ymax, xmin: xmax]
    B = random_chunk
    img_chunked[ymin:ymax, xmin: xmax] = random_chunk
    img_chunked = np.uint8(img_chunked)
    # return new image
    return img_chunked


# fic any pacularities in data
def problem_points(xmin, ymin, xmax, ymax, height, width):
    xm = xmin + 0
    ym = ymin + 0
    xmx = xmax + 0
    ymx = ymax + 0

    problem = False
    # check if negative
    if xmin < 0:
        xmin = 0
        problem = True
    if ymin < 0:
        ymin = 0
        problem = True
    if xmax < 0:
        xmax = 0
        problem = True
    if ymax < 0:
        ymax = 0
        problem = True

    # check if over
    if xmin >= width:
        xmin = width - 1
        problem = True
    if ymin >= height:
        ymin = height - 1
        problem = True
    if xmax >= width:
        xmax = width - 1
        problem = True
    if ymax >= height:
        ymax = height - 1
        problem = True

    # check if any are equal
    # convert to integers
    xmin_int = int(xmin)
    ymin_int = int(ymin)
    xmax_int = int(xmax)
    ymax_int = int(ymax)

    if xmin_int == xmax_int:
        xmin = 0
        xmax = width - 1
        problem = True

    if ymin_int == ymax_int:
        ymin = 0
        ymax = height - 1
        problem = True

    # if problem:
    #    print("issue with points", str(xm), str(ym), str(xmx), str(ymx))
    #    print("fixed to", str(xmin), str(ymin), str(xmax), str(ymax))

    return xmin, ymin, xmax, ymax


def resize_and_points(img_arr, xmin, ymin, xmax, ymax, maxsize):
    max_height = maxsize[0]
    max_width = maxsize[1]

    original_shape = np.shape(img_arr)

    original_height = original_shape[0]
    original_width = original_shape[1]

    S0 = np.zeros((original_height, original_width, 3))
    S0[0:original_height, 0:original_width, 0:3] = img_arr[0:original_height, 0:original_width, 0:3]
    S0 = np.uint8(S0)
    SA = Image.fromarray(S0)
    SA.thumbnail((max_height, max_width))
    SC = np.array(SA)

    new_shape = np.shape(SC)
    new_height = new_shape[0]
    new_width = new_shape[1]

    x_scale_factor = new_width / original_width
    y_scale_factor = new_height / original_height

    # compensate the zoom points by the scale factors
    xmin_scaled = xmin * x_scale_factor
    xmax_scaled = xmax * x_scale_factor
    ymin_scaled = ymin * y_scale_factor
    ymax_scaled = ymax * y_scale_factor
    return SC, xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled


# target size has to be list
def aug(img_arr_pre, xmin, ymin, xmax, ymax, target_size):
    global datagen

    # img_arr_pre, xmin, ymin, xmax, ymax=debug_func(img_arr_pre, xmin, ymin, xmax, ymax)
    # return img_arr_pre, xmin, ymin, xmax, ymax
    img_shape = np.shape(img_arr_pre)
    height = img_shape[0]
    width = img_shape[1]
    # make sure any weird behaivor is under control
    if xmin <= 0:
        xmin = 0

    if ymin <= 0:
        ymin = 0

    if xmax >= width:
        xmax = width - 1

    if ymax >= height:
        ymax = height - 1
    # process the image array
    # remove extra dimension if png file
    img_arr = img_arr_pre[0:height, 0:width, 0:3]

    # determine macimum zoom range
    xmin_limit = zoom_point_limit(xmin, width)
    xmax_limit = zoom_point_limit(xmax, width)
    ymin_limit = zoom_point_limit(ymin, height)
    ymax_limit = zoom_point_limit(ymax, height)

    zoom_x_limit = max(xmin_limit, xmax_limit)
    zoom_y_limit = max(ymin_limit, ymax_limit)
    # from this range choose random zoom level
    zoom_x = random.uniform(zoom_x_limit, 1.5)
    zoom_y = random.uniform(zoom_y_limit, 1.5)

    #zoom_x = random.uniform(.9, 1.1)
    #zoom_y = random.uniform(.9, 1.1)

    # determine new points from this zoom
    xmin_zoom = new_zoom_points(xmin, zoom_x, width)
    xmax_zoom = new_zoom_points(xmax, zoom_x, width)
    ymin_zoom = new_zoom_points(ymin, zoom_y, height)
    ymax_zoom = new_zoom_points(ymax, zoom_y, height)
    zoompoints = [xmin_zoom, ymin_zoom, xmax_zoom, ymax_zoom]
    # zoom in or out of image
    zoom_img = datagen.apply_transform(x=img_arr, transform_parameters={'zx': zoom_y, 'zy': zoom_x})
    # scale the image down to target size
    # determine the scale factors
    x_scale_factor = target_size[1] / width
    y_scale_factor = target_size[0] / height

    # compensate the zoom points by the scale factors
    xmin_scaled = xmin_zoom * x_scale_factor
    xmax_scaled = xmax_zoom * x_scale_factor
    ymin_scaled = ymin_zoom * y_scale_factor
    ymax_scaled = ymax_zoom * y_scale_factor

    # determine new height and width
    height_scaled = target_size[0]
    width_scaled = target_size[1]
    # resize image to target size
    zoom_img2 = Image.fromarray(zoom_img)
    scaled_img_0 = zoom_img2.resize((target_size[0], target_size[1]))
    scaled_img = np.array(scaled_img_0)

    # display(scaled_img,xmin_scaled,ymin_scaled,xmax_scaled,ymax_scaled)

    # rotate image
    # determine maximum range image can be rotated
    angle_clock, angle_clock_counter = max_rotation(xmin, ymin, xmax, ymax, 90, height_scaled, width_scaled)
    # draw out random number from that range
    if angle_clock - angle_clock_counter <= 0:
        angle = 0
    else:
        angle = randint(angle_clock_counter, angle_clock)
    # rotate image
    # scaled_seperated=np.zeros((height_scaled,width_scaled,3))
    # scaled_seperated[0:height_scaled,0:width_scaled]=scaled_img[0:height_scaled,0:width_scaled]
    rotated_img = datagen.apply_transform(x=scaled_img, transform_parameters={'theta': angle})
    # determine new points afer rotation
    xmin_rotated, ymin_rotated, xmax_rotated, ymax_rotated = post_rotation_points(xmin_scaled, ymin_scaled, xmax_scaled,
                                                                                  ymax_scaled, angle, height_scaled,
                                                                                  width_scaled)
    # display(rotated_img,  xmin_rotated, ymin_rotated, xmax_rotated, ymax_rotated )

    # shift image
    # determine maximum range of shift
    right_shift_limit = width_scaled - xmax_rotated - 1
    left_shift_limit = -1 * xmin_rotated
    up_shift_limit = -1 * ymin_rotated  # height - new_ymax
    down_shift_limit = height_scaled - ymax_rotated - 1  # -1 * new_ymin

    # chose random integer from this range
    left = int(min(left_shift_limit, right_shift_limit))
    right = int(max(left_shift_limit, right_shift_limit))
    down = int(min(up_shift_limit, down_shift_limit))
    up = int(max(up_shift_limit, down_shift_limit))
    if right - left <= 0:
        horizontal_shift = 0
    else:
        horizontal_shift = (randint(left, right))

    if up - down <= 0:
        vertical_shift = 0
    else:
        vertical_shift = -(randint(down, up))
    # = (randint(left, right))
    # vertical_shift = -(randint(down, up))

    # determine new points from these shifts
    shift_img = np.zeros((height_scaled, width_scaled, 3))
    shift_img[0:height_scaled, 0:width_scaled] = rotated_img[0:height_scaled, 0:width_scaled]
    # shift image
    shift_img = np.roll(shift_img, horizontal_shift, axis=1)
    shift_img = np.roll(shift_img, -vertical_shift, axis=0)
    # determine new points and truncate into integers
    xmin_shift = (xmin_rotated + horizontal_shift * 1)
    xmax_shift = (xmax_rotated + horizontal_shift * 1)
    ymin_shift = (ymin_rotated - vertical_shift * 1)
    ymax_shift = (ymax_rotated - vertical_shift * 1)

    # fix any irregularities
    xmin_shift, ymin_shift, xmax_shift, ymax_shift = problem_points(xmin_shift, ymin_shift, xmax_shift, ymax_shift,
                                                                    height_scaled, width_scaled)
    test_points = [xmin_shift, ymin_shift, xmax_shift, ymax_shift]
    # display(shift_img,xmin_shift, ymin_shift,xmax_shift, ymax_shift)

    # turn values in ratio of height and width
    xmin_ratio = xmin_shift / width_scaled
    xmax_ratio = xmax_shift / width_scaled
    ymin_ratio = ymin_shift / height_scaled
    ymax_ratio = ymax_shift / height_scaled
    # construct return array
    new_points = [xmin_ratio, ymin_ratio, xmax_ratio, ymax_ratio]
    # take chunk out
    # img_chunked = cutoff(shift_img, xmin_shift, ymin_shift, xmax_shift, ymax_shift)
    # finally apply brightness level transform
    bright = random.uniform(.9, 1.1)
    new_img = datagen.apply_transform(x=shift_img, transform_parameters={'brightness': bright})
    # unsigned integers
    # img_chunked = np.uint8(img_chunked)
    shift_img = np.uint8(shift_img)
    # display(new_img,xmin_shift,ymin_shift,xmax_shift,ymax_shift)

    return new_img, new_points


#############################################################################################
# all below is the normal stuff
def calculate_iou(target_boxes, pred_boxes):
    xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
    yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
    xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
    yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
    interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
    boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    return mse + (1 - iou)


def iou_metric(y_true, y_pred):
    return calculate_iou(y_true, y_pred)


input_dim = 224
input_shape = (input_dim, input_dim, 3)
dropout_rate = 0.5
alpha = 0.2



model_layers = [
    # adding guassian noise
    keras.layers.GaussianNoise(stddev=.1, input_shape=input_shape),
    keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),

    keras.layers.Dense(4800),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Dropout(.2),
    keras.layers.Dense(1240),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Dense(1240),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Dense(1240),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Dense(1240),
    keras.layers.LeakyReLU(alpha=alpha),
    keras.layers.Dense(5),
    keras.layers.LeakyReLU(alpha=alpha),
]

model = load_model("E:/machine learning/saved models/combineAug2.h5",
                   custom_objects={'custom_loss': custom_loss, 'iou_metric': iou_metric})
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.0001),
    loss=custom_loss,
    metrics=[iou_metric]
)
train_generator = image_generator(xlfilepath, batch_size=64)
model.fit_generator(train_generator, steps_per_epoch=341, epochs=10, verbose=1)
model.save("E:/machine learning/saved models/combineAug2.h5")
