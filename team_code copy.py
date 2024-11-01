#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import torch.nn as nn

from helper_code import *
import os, sys
from digital_model import UNetModel
from classify_model import Net1D #Inceptionv4
from inception_resnet import Inception_ResNetv2
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import pandas as pd
import scipy.interpolate as interp
from scipy.signal import find_peaks
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import subprocess

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    
    # 1、generate training_data,预估需要存储73G ，花费时间45小时
    command_1 = ['python', 'gen_ecg_images_from_data_batch.py', '-i', data_folder, '-o', data_folder]
    gen_pic = subprocess.run(command_1, capture_output=True, text=True)
    
    # 2、add pic to .hea
    command_2 = ['python', 'prepare_image_data.py', '-i', data_folder, '-o', data_folder]
    add_pic = subprocess.run(command_2, capture_output=True, text=True)
    
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can remove this part of the code.

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    digitization_features = list()
    classification_features = list()
    classification_labels = list()
    layout_labels = list()
    
    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        
        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        features = extract_features(record)
        digitization_features.append(features)

        
        # Some images may not be labeled...
        labels = load_labels(record)
        
        # layout label
        layout_label = load_label_layout(record)
        layout_labels.append(layout_label)
        
        
        if any(label for label in labels):
            classification_features.append(features)
            classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    digitization_model = np.mean(features)




    exit()
    # Train the classification model. If you are not training a classification model, then you can remove this part of the code.
    # train classify model
    
    # encode labels
    labels_dic = ['NORM','Acute MI','Old MI','STTC','CD','HYP','PAC','PVC','AFIB/AFL','TACHY','BRADY']
    # 创建一个字典来映射每个标签到一个唯一的索引
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    print("Label to index mapping:")
    print(label_to_index)
    encoded_data_labels = encode_labels(classification_labels, label_to_index) # label 
    print(encoded_data_labels)
    
    
    
    # This very simple model trains a random forest model with these very simple features.
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # n_epoch = 50
    # batch_size = 16
    # dataset = MyDataset(X_train, Y_train,transform=transform)
    # dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # model = Inception_ResNetv2()
    # model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    # loss_func = torch.nn.BCEWithLogitsLoss()
    
    # train model
    # total_train_loss = [] # loss
    # temp = []
    # step = 0
    # for _ in tqdm(range(n_epoch), desc="epoch", leave=False):
    #     # train
    #     model.train()
    #     prog_iter = tqdm(dataloader, desc="Training", leave=False , ncols=80)
    #     for batch_idx, batch in enumerate(prog_iter):
    #         input_x, input_y = tuple(t.to(device) for t in batch)
            
    #         # 需要在这里进行维度的修改
    #         pred = model(input_x)
    #         loss = loss_func(pred, input_y.float())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         step += 1
    #         temp.append(loss.data.cpu())
            
    #     total_train_loss.append(np.mean(temp))
    #     # print("第{}轮loss:".format(_+1),np.mean(temp))
    #     temp.clear()
    #     scheduler.step(_)
    
    # # save classify model
    # save_checkpoint(model,optimizer,model_folder)
    
    
    # train layout model 
    # layout labels
    labels_layout = list()
    
    
    
    classification_features = np.vstack(classification_features)
    classes = sorted(set.union(*map(set, classification_labels)))
    classification_labels = compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    classification_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done.')
        print()
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_filename = os.path.join(model_folder, 'digitization_model_v5.2.pth')
    saved_model = UNetModel()
    saved_model.load_state_dict(torch.load(digitization_filename, map_location='cuda:0'))
    digitization_model = saved_model

    # classify
    classification_filename = os.path.join(model_folder, 'picture_checkpoint_resnet_44.pth')
    saved_model_classify = Inception_ResNetv2()
    checkpoints = torch.load(classification_filename, map_location='cuda:0')
    model_dict = saved_model_classify.state_dict()
    state_dict = {k: v for k, v in checkpoints['model_state_dict'].items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    saved_model_classify.load_state_dict(model_dict)
    
    classification_model = {}
    classification_model["classify"] = saved_model_classify
    
    # layout model
    layout_filename = os.path.join(model_folder, '2024_8_10_layout.pth')
    layout_model = Inception_ResNetv2(classes=3)
    model_state_dict = torch.load(layout_filename, map_location='cuda:0')
    layout_model.load_state_dict(model_state_dict)
    classification_model["layout"] = layout_model
    # add model_folder
    classification_model["model_folder"] = model_folder
    
    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.

    images = load_images(record) # pil image open
    image = images[0].convert("RGB")
    image_classify = images[0].convert("RGB") # 用于分类


    # inference 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = digitization_model
    model.to(device)
    model.eval()
    image, larger_dim, pad_width, pad_height = prepare_images(image)
    image = image.to(device)
    output = model(image)
    # resize back to orignal size and remove the padding part
    output = resize_back(output, larger_dim, pad_width, pad_height)

    orignal_height = larger_dim - pad_height
    orignal_width = larger_dim - pad_width
    
    output = F.sigmoid(output)
    output = output.to("cpu")

    output = convert_format(output[0])
    output = output.unsqueeze(0)

    # get dependency
    header_str = load_header(record)
    sampling_hz = get_sampling_frequency(header_str)
    sampling_hz = int(sampling_hz)
    # sampling_hz = 100
    dependency_dict = get_dependency(image_size=larger_dim, sampling_hz=sampling_hz)
    pixel_per_grid = dependency_dict['pixel_per_grid']
    pixel_num_full = dependency_dict['pixel_num_full']
    sampling_num = dependency_dict['sampling_num']
    # flatten output
    output_flat = torch.zeros((orignal_height, orignal_width))
    for i in range(13):
        output_flat += output[0][i]
    pixel_array = prepare_tensor_for_display(output_flat)
    pixel_array = post_processing(pixel_array)
    # get offset
    offset_horizonal = get_offset(pixel_array)
    # assert offset_horizonal > 0
    
    #layout
    my_pixel_array = 255 - pixel_array
    array_3d = np.stack([my_pixel_array] * 3, axis=0) 
    model2 = classification_model["layout"]
    model2.to(device)
    model2.eval()
    with torch.no_grad():
        features = extract_features1(array_3d)
        features = features.to(device)
        pred_score = model2(features)
    
    layout = torch.argmax(pred_score, dim=1)
    layout = layout.to('cpu')
    layout = layout.tolist() # 0 :(3行4列) 1：（12行一列）2：（6行2列）
    layout = layout[0]
    # print(layout)
    
    # per lead signal
    sqi_flag = True
    signal = np.zeros((sampling_num, 12))
    # set boarder to zero
    output = set_boarder_to_zero(output)
    for i in range(12):
        try:
            is_II_full = False
            if layout != 1 and i == 1:
                # handle II lead and potential II full lead
                II_1_sum = torch.sum(output[0][1])
                II_2_sum = torch.sum(output[0][12])

                if II_1_sum > II_2_sum:
                    II_idx = 1
                else:
                    II_idx = 12
                    is_II_full = True

                image_float = output[0][II_idx].detach().numpy().astype(np.float32)
                img_sqi = check_img_sqi(image_float)
                if sqi_flag and not img_sqi :
                    signal_predict = np.zeros(sampling_num)
                else:
                    signal_predict, stt_c_offset, stp_c_offset, row_peak_idx = pixel2signal(output[0][II_idx], \
                                                                        offset_horizonal, pixel_num_full, \
                                                                        larger_dim, sampling_num, \
                                                                        pixel_per_grid)
        
            else:
                # handle other leads
                image_float = output[0][i].detach().numpy().astype(np.float32)
                img_sqi = check_img_sqi(image_float)
                if sqi_flag and not img_sqi :
                    signal_predict = np.zeros(sampling_num)
                else:
                    signal_predict, stt_c_offset, stp_c_offset, row_peak_idx = pixel2signal(output[0][i], \
                                                                            offset_horizonal, pixel_num_full, \
                                                                            larger_dim, sampling_num, \
                                                                            pixel_per_grid)

            signal[:, i] = signal_predict
        except Exception as e:
            print(e)
            print(f'Error: {record}')
            signal[:, i] = np.zeros(sampling_num)

    # classification
    model1 = classification_model["classify"]
    model1.to(device)
    model1.eval()
    # thresholds_list = [0.23, 0.01, 0.01, 0.03, 0.12, 0.03, 0.01, 0.01, 0.01, 0.05, 0.01]
    thresholds_list = [0.13, 0.03, 0.09999999999999999, 0.09999999999999999, 0.060000000000000005, 0.060000000000000005, 0.01, 0.02, 0.04, 0.02, 0.02]

    # extract 
    with torch.no_grad():
        features = extract_for_classify(classification_model["model_folder"],image_classify) # (1,3,299,299)
        features = features.to(device)
        pred = torch.sigmoid(model1(features))
        score = (pred.cpu().data.numpy())
    
    label = (score > np.array(thresholds_list)) # tensor
    label = label.squeeze().tolist()
    
    
    # decode labels
    labels_dic = ['NORM','Acute MI','Old MI','STTC','CD','HYP','PAC','PVC','AFIB/AFL','TACHY','BRADY']
    # create index
    label_to_index = {label: idx for idx, label in enumerate(labels_dic)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}  
    
    
    labels = decode_labels(label, index_to_label)
    print(labels)
    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features1(picture): # extract feature
    """
        Args:
            picture : ndarray shape(3,512,512)
        return:
            data_output : torch shape(1,3,299,299) 
    """
    image = Image.fromarray(picture.transpose(1, 2, 0))  # 从 (C, H, W) 转换为 (H, W, C)
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # 调整图像尺寸为2000x1000
        transforms.ToTensor(),  # 将图像转换为Tensor
    ])
    im = transform(image)
    
    data_output = im.unsqueeze(0)
    return data_output

def extract_for_classify(model_folder,picture):
    """
        picture : pic
        return :    
    """
    
    full_signal_time_length = 10 # second
    pixel_per_grid = 39.37 / 5 # pixel / grid, the smaller one
    grid_length = 1 # mm / grid    pixel_per_mm = pixel_per_grid / grid_length # pixel / mm
    paper_speed = 25 # mm / s
    grid_time_length = grid_length / paper_speed  # s / grid (0.04s/grid)
    #1. pixel_num_full = full_signal_time_length * paper_speed  * pixel_per_mm
    pixel_num_full = int(full_signal_time_length / grid_time_length * pixel_per_grid + 0.5) # 
    sampling_hz = 100
    sampling_num = full_signal_time_length * sampling_hz
    
    # load model
    digitization_filename = os.path.join(model_folder, 'digitization_model_v3.pth')
    saved_model = UNetModel()
    saved_model.load_state_dict(torch.load(digitization_filename, map_location='cuda:0'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = saved_model
    model.to(device)
    
    image = picture
    
    # preprocess the image, # original size: h=1700, w=2200
    im_w, im_h = image.size 
    adjust_ratio = im_w / 2200.

    pixel_per_grid = pixel_per_grid * adjust_ratio
    
    larger_dim = max(im_w, im_h)
    pad_width = larger_dim - im_w
    pad_height = larger_dim - im_h
    tsf_pad = transforms.Pad((0, 0, pad_width, pad_height), fill=0.)
    tsf_resize = transforms.Resize((512, 512))

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tsf_pad,
        tsf_resize,
    ])

    image = transform_image(image)
    image = image.float()
    image = image[None, ...]
    # inference 
    image = image.to(device)
    output = model(image)
    output = F.sigmoid(output)
    output = output.to("cpu")

    output = convert_format(output[0])
    output = output.unsqueeze(0)
    # save flatten output
    # flat the output [13, 512, 512] -> [512, 512]
    output_flat = torch.zeros((512, 512))
    for i in range(13):
        output_flat += output[0][i]

    pixel_array = prepare_tensor_for_display(output_flat)
    pixel_array = 255 - pixel_array
    
    
    array_3d = np.stack([pixel_array] * 3, axis=0) 
    # 修改维度
    out_pic = Image.fromarray(array_3d.transpose(1, 2, 0))  # 从 (C, H, W) 转换为 (H, W, C)
    transform_out = transforms.Compose([
        transforms.Resize((299, 299)),  # 调整图像尺寸为2000x1000
        transforms.ToTensor(),  # 将图像转换为Tensor
    ])
    im = transform_out(out_pic)
    data_output = im.unsqueeze(0)
    
    return data_output # tensor

def extract_features(record):
    images = load_images(record)
    # 读取图片
    print(len(images))
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)

# digital related

def get_dependency(image_size=2200, sampling_hz=100):
    # dependency
    full_signal_time_length = 10 # second
    
    pixel_per_grid = 39.37 / 5 / (2200/image_size)# pixel / grid, the smaller one
    grid_length = 1 # mm / grid    pixel_per_mm = pixel_per_grid / grid_length # pixel / mm
    
    paper_speed = 25 # mm / s
    grid_time_length = grid_length / paper_speed  # s / grid (0.04s/grid)

    #1. pixel_num_full = full_signal_time_length * paper_speed  * pixel_per_mm
    pixel_num_full = int(full_signal_time_length / grid_time_length * pixel_per_grid + 0.5) # 

    # 
    sampling_num = full_signal_time_length * sampling_hz

    return {
        'pixel_per_grid': pixel_per_grid,
        'pixel_num_full': pixel_num_full,
        'sampling_num': sampling_num
    }

class RescaleChannels(object):
    def __call__(self, sample):
        return 2 * sample - 1

def resize_back(tensor, larger_dim, pad_width, pad_height):
    # resize back to orignal size and remove the padding part
    tensor = F.interpolate(tensor, size=(larger_dim, larger_dim), mode='nearest')
    tensor = tensor[:, :, 0:larger_dim-pad_height, 0:larger_dim-pad_width]
    return tensor

def prepare_images(image):
    # preprocess the image, # original size: h=1700, w=2200
    im_w, im_h = image.size 
    larger_dim = max(im_w, im_h)
    pad_width = larger_dim - im_w
    pad_height = larger_dim - im_h

    tsf_pad = transforms.Pad((0, 0, pad_width, pad_height), fill=0.)
    tsf_resize = transforms.Resize((512, 512))

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        RescaleChannels(),
        tsf_pad,
        tsf_resize,
    ])

    image = transform_image(image)
    image = image.float()
    image = image[None, ...]
    return image, larger_dim, pad_width, pad_height

def convert_format(mask, num_classes=14):
    mask = mask.permute(1, 2, 0)
    mask = torch.argmax(mask, dim=-1)
    mask_orignal = torch.zeros((num_classes-1, mask.shape[0], mask.shape[1]))
    for i in range(1, num_classes):
        mask_orignal[i-1] = (mask == i).float()
    # mask_orignal = mask_orignal[1:,:,:]
    return mask_orignal

def prepare_tensor_for_display(tensor):

    pixel_array = tensor.detach().numpy()

    if len(pixel_array.shape) == 3:
        pixel_array = np.moveaxis(pixel_array, 0, -1)
        if pixel_array.shape[-1] == 1:
            pixel_array = pixel_array[:,:,0]
    elif len(pixel_array.shape) == 2:
        pixel_array = pixel_array
    pixel_array = pixel_array * 255
    pixel_array = pixel_array.astype(np.uint8)
    return pixel_array

def post_processing(pixel_array):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    binary_image = pixel_array.astype(np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return binary_image

def get_offset(pixel_array):
    # ret, binary_image = cv2.threshold(pixel_array, 127, 255, cv2.THRESH_BINARY)
    binary_image = pixel_array

    h, w = binary_image.shape
    for i in range(5, w-5):   # w                                  
        for j in range(5, h-5): # h
            if binary_image[j, i] == 255:
                return i
    return 0

def set_boarder_to_zero(output):
    # Set first 5 and last 5 elements to zero for the Height dimension
    output[:, :, 0:5, :] = 0
    output[:, :, -5:, :] = 0

    # Set first 5 and last 5 elements to zero for the Width dimension
    output[:, :, :, 0:5] = 0
    output[:, :, :, -5:] = 0
    return output

def interplote(data):
    """
    Interpolates a signal segment, fulfilling the missing values.

    Args:
        signal_segment (numpy.ndarray): The input signal segment.

    Returns:
        numpy.ndarray: The interpolated signal segment.

    """
    # Convert to a NumPy array
    data_array = np.array(data)

    # Identify the positions of zero values
    missing_positions = data_array == 0.0

    # Replace zero values with NaN (not a number)
    data_array[missing_positions] = np.nan

    # Use Pandas to interpolate the missing values
    data_series = pd.Series(data_array)
    interpolated_data = data_series.interpolate(method='quadratic')

    # Convert back to a list if needed
    interpolated_list = np.array(interpolated_data.tolist())

    return interpolated_list

def pixel2signal(pixel_array, offset_horizonal, pixel_num_full, larger_dim, sampling_num, pixel_per_grid=39.37/5, image_name=None, i=None):
    """
    Convert a pixel array to a signal segment.

    Args:
        pixel_array (ndarray): The input pixel array.
        offset_horizonal (int): The horizontal offset.
        pixel_num_full (int): The number of pixels in the full length.
        larger_dim (int): The dimension of the larger image.
        sampling_num (int): The number of samples.
        pixel_per_grid (float, optional): The number of pixels per grid. Defaults to 39.37/5.
        thresholding_value (int, optional): The thresholding value. Defaults to 127.
        image_name (str): The name of the image.
        i (int): The index of the image.

    Returns:
        tuple: A tuple containing the signal segment, the start index offset, and the end index offset.
    """
    pixel_array = prepare_tensor_for_display(pixel_array)
    # pixel_array = 255 - pixel_array
    # pixel_array[pixel_array < 15] = 255
    pixel_array = post_processing(pixel_array)

    im = Image.fromarray(pixel_array)
    # im = im.resize((larger_dim, larger_dim))

    # find r peaks
    row_peak_idx = []

    signal_segment, stt_c, stp_c, min_h, max_h = reduce_dimension(np.array(im), offset_horizonal, max_len=pixel_num_full, row_peak_idx=row_peak_idx)
    signal_segment = interplote(signal_segment)
    signal_segment = signal_segment + max(signal_segment)
    signal_segment = signal_segment - np.mean(signal_segment)
    signal_segment = signal_segment / pixel_per_grid * 0.1
    
    stt_c_offset = stt_c - offset_horizonal
    stp_c_offset = stp_c - offset_horizonal
    # some stt_c_offset has -1 issue
    if stt_c_offset < 0:
        stp_c_offset = stp_c_offset - stt_c_offset
        stt_c_offset = 0
    signal_predict = signal_sampling(signal_segment, stt_c_offset, stp_c_offset, pixel_num_full, sampling_num)
    
    return signal_predict, stt_c_offset, stp_c_offset, row_peak_idx


def reduce_dimension(pixel_array, offset, max_len = 2000, row_peak_idx=[]):
    """
    Reduce the dimension of a pixel array by extracting the signal position along the width.

    Args:
        pixel_array (numpy.ndarray): The input pixel array.
        offset (int): The offset value.
        max_len (int, optional): The maximum length. Defaults to 2000.

    Returns:
        numpy.ndarray: The reduced dimension signal array.
        int: The start column index.
        int: The stop column index.
        int: The minimum height value.
        int: The maximum height value.
    """
    binary_image = pixel_array

    sig_list = []
    stt_c, stp_c = -1, -1 
    min_h, max_h = np.inf, -1 
    h, w = binary_image.shape
    for i in range(5, w-5):   # along the width, per column                           
        zero_positions = np.argwhere(binary_image[:, i] == 255)
        if zero_positions.size > 1:
            meansig = np.median(zero_positions)
            sig_list.append(meansig)

            if stt_c == -1:
                stt_c = i
            if stp_c < i:
                stp_c = i
        else:
            if stt_c > -1 and i < stt_c + max_len:
                try:
                    # sig_list.append(sig_list[-1])
                    sig_list.append(0)
                except IndexError:
                    sig_list.append(0)
                    print(f'IndexError: {i}, offset, {offset}, max_len, {max_len}')
                stp_c = i
    # remove the trailing zeros
    while sig_list[-1] == 0:
        sig_list.pop()
        stp_c -= 1
    # sig_list = sig_list[:stp_c - stt_c + 1]
    sig_list = np.array(sig_list)*-1
    return sig_list, stt_c, stp_c, min_h, max_h


def signal_sampling(signal_segment, stt, stp, full_length, sampling_num):
    """
    Samples a signal segment and returns the sampled signal.

    Parameters:
    signal_segment (numpy.ndarray): The signal segment to be sampled.
    stt (int): The starting index of the signal segment.
    stp (int): The ending index of the signal segment.
    full_length (int): The length of the full signal.
    sampling_num (int): The number of samples to be taken.

    Returns:
    numpy.ndarray: The sampled signal.

    """
    # signal_list_full_length = full_length
    # print(f"signal_segment: {len(signal_segment)}, stt: {stt}, stp: {stp}, full_length: {full_length}, sampling_num: {sampling_num}")
    # sys.exit()
    # hotfix of length doesn't match issue
    signal_list_full_length = max(full_length, stp + 1)
    if stp - stt + 1 != len(signal_segment):
        stp = stt + len(signal_segment) - 1
    # fulfil signal_segment to the full signal length
    signal_predict = np.zeros(signal_list_full_length)
    signal_predict[stt:stp+1] = signal_segment
    # print(f"signal_segment: {len(signal_segment)}, stt: {stt}, stp: {stp}, full_length: {full_length}, sampling_num: {sampling_num}")
    # sampling
    if len(signal_predict) > sampling_num:
        idx = np.round(np.linspace(0, len(signal_predict) - 1, sampling_num)).astype(int)
        signal_sampled = signal_predict[idx]
    else:
        # Create an interpolation function
        x = np.arange(len(signal_predict))
        f = interp.interp1d(x, signal_predict)
        # Create a new x array with 5000 points
        xnew = np.linspace(0, len(signal_predict) - 1, sampling_num)
        # Use the interpolation function to compute the values at the new points
        signal_sampled = f(xnew)
    return signal_sampled

def row_ract(ecg, rpos, length=850):
    """
    Extracts rows from an ECG signal based on the R-peaks positions.

    Parameters:
    - ecg (array-like): The ECG signal.
    - rpos (array-like): The positions of the R-peaks in the ECG signal.
    - length (int): The length of the extracted rows. Default is 850.

    Returns:
    - fin_rpos (set): The set of positions of the extracted rows.

    """
    ts = 50
    ts_str = abs(rpos[0])
    ts_end = abs(length - rpos[-1])
    ts = np.min([ts_str, ts_end, ts])

    new_rpos = []
    for idx, r in enumerate(rpos):
        segmet = ecg[r - ts:r + ts]

        new_rpos.append(np.argmax(segmet) + r - ts)
    fin_rpos = set(new_rpos)

    return fin_rpos


def check_img_sqi(image_float):
    h = image_float.shape[0]

    row_sum = cv2.reduce(image_float, 1, cv2.REDUCE_SUM)[2:-2, 0]    # 横向
    col_sum = cv2.reduce(image_float, 0, cv2.REDUCE_SUM)[0, 2:-2]    # 纵向

    process_row_sum = row_sum
    process_col_sum = col_sum

    # find the postion of first non-zero element in y direction
    for idx in range(len(process_row_sum)):
        if process_row_sum[idx] != 0:
            startidx = idx
            break
    # find the postion of last non-zero element in y direction
    for idx in range(len(process_row_sum)):
        if process_row_sum[::-1][idx] != 0:
            endidx = len(process_row_sum) - idx
            break
    
    # calculate the distance between the first and last non-zero element in y direction
    distance = endidx - startidx

    row_peak_fp = find_peaks(process_row_sum, height=0.1*max(process_col_sum))[0]
    row_peak_idx = row_ract(process_row_sum, row_peak_fp, h)
    lead_num = len(row_peak_idx)

    if lead_num > 1:
        img_sqi = False
    else:
        img_sqi = True
    return img_sqi

# classify related

# decode list to label
def decode_labels(label_vector, index_to_label):
    labels = [index_to_label[idx] for idx, label_present in enumerate(label_vector) if label_present == 1]
    return labels

def encode_labels(data, label_to_index):
# 初始化零矩阵，形状为[数据长度, 标签数量]
    target = torch.zeros((len(data), len(label_to_index)), dtype=torch.float32)
    
    # 填充矩阵
    for i, items in enumerate(data):
        for item in items:
            if item in label_to_index:
                target[i, label_to_index[item]] = 1
    
    return target

class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self. transform = transform
        
    def __getitem__(self, index):

        image = self.data[index]
        label = self.label[index]
        
        image = Image.fromarray(image.transpose(1, 2, 0).astype('uint8'))
        if self.transform:
            image = self.transform(image)
            
        return image,torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
def save_checkpoint(model, optimizer,model_folder):
    print('Model Saving...')
    model_state_dict = model.state_dict()
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(model_folder, 'classifiy_model.pth')) # 保存路径

# layout related
def load_label_layout(record):
        
    """
        inout : record
        output : num_col (1,2,4)
                1->1
                2->2
                4->0
    """
    # record /data/0shared_data/ptb__ECGImage/ptbxl_data_new_change_with_log/records500/00000/00001_hr
    base_path = os.path.dirname(record)
    log_path = base_path+'/'+'log.xlsx'
    df = pd.read_excel(log_path) # df是数据
        
    file_name = os.path.basename(record) # 00001_lr 
    num_col_value = df.loc[df['name'] == file_name+'-0', 'num_col'].iloc[0]
        
    if num_col_value == 4:
        return 0
    # 读取log中的文件中的label
    return num_col_value
