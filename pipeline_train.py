# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

import argparse
import os
import sys
import random
import warnings
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

# Tensorflow and Keras
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from utils.loss_functions import dice_coef_loss
from utils.data_generator import data_generator
from models.unet_models import unetModel_basic_4, unetModel_residual
from utils.configurations import Configurations
from utils.utils import filter_tbud_count, eprint, mkdir_if_not_exist

# Set seed value for all of frameworks
seed_value= 2021 
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Get Date for Ouput File Naming
current_time = datetime.now().strftime('%H%M_%m%d%Y')

def get_data(configurations, data_folder):
    
    # Write Directory
    dir_write = os.path.join(configurations.dir_write, '/Run_Train_' + configurations.model_name  + '_' + str(current_time))
    dir_pred = os.path.join(dir_write, 'Pred_imgs')
    dir_model = os.path.join(dir_write, 'Model')
    dir_log = os.path.join(dir_write, 'Log')
    
    if not os.path.exists(dir_write):
        os.makedirs(dir_write)
        os.makedirs(dir_pred)
        os.makedirs(dir_model)
        os.makedirs(dir_log)
    
    IMG_WIDTH = configurations.size_img
    IMG_HEIGHT = configurations.size_img
    IMG_CHANNELS = 3
    TRAIN_PATH = data_folder
    
    # Path of Image Tiles and Masks
    print(data_folder)
    path = os.path.join(TRAIN_PATH, "img") 
    path_mask = os.path.join(TRAIN_PATH, "mask")
    path_bud_info = os.path.join(TRAIN_PATH, "Bud_Info")
    
    eprint(f'[DEBUG][get_data] Getting and Resizing({IMG_WIDTH}x{IMG_HEIGHT}) Train Images and Masks... ')

    # Get and resize train images and masks
    train_cpt = int(sum([len(files) for r, d, files in os.walk(TRAIN_PATH + "img/")]))

    eprint(f'[DEBUG][get_data] Getting and Resizing Train Images and Masks Done!\nPath to img: {path}')
    sys.stdout.flush()

    _, _, files_orj = next(os.walk(path))
    _, _, files_mask = next(os.walk(path_mask))
    files_orj = sorted(files_orj)
    files_mask = sorted(files_mask)

    eprint(f'[DEBUG][get_data] Number of Image Tiles: {len(files_orj)}\t Number of Image Masks: {len(files_mask)}\n')

    train_cpt_filtered = len(files_orj)
    files_orj_filtered = files_orj
    files_mask_filtered = files_mask

    if int(configurations.thold_tbud) > 0:
        train_cpt_filtered = 0
        files_orj_filtered = []
        files_mask_filtered = []

        for i, f in enumerate(files_orj):
            # Apply Bud Threshold 
            if filter_tbud_count(path_bud_info, f, int(configurations.thold_tbud)):
                train_cpt_filtered += 1
                files_orj_filtered.append(files_orj[i])
                files_mask_filtered.append(files_mask[i])
        
    X_train = np.ndarray((train_cpt_filtered, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y_train = np.ndarray((train_cpt_filtered, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)  # dtype=np.bool)

    for i, f in enumerate(files_orj_filtered):
        # # Apply Bud Threshold 
        # if not filter_tbud_count(path_bud_info, f, configurations.thold_tbud):
        #     continue
        img = cv2.imread(os.path.join(path, f))
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        img = img / 255
        X_train[i] = img

    for i, fm in enumerate(files_mask_filtered):
        # # Apply Bud Threshold 
        # if not filter_tbud_count(path_bud_info, fm, configurations.thold_tbud):
        #     continue
        img_mask = cv2.imread(os.path.join(path_mask, fm), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        img_mask = img_mask / 255
        img_mask = np.expand_dims(img_mask, axis=-1)
        Y_train[i] = img_mask
    
    eprint(f'[DEBUG][get_data] After Filter thold_tbud:{configurations.thold_tbud} Number of Image Tiles: {len(X_train)}\t Number of Image Masks: {len(Y_train)}\n')

    eprint(f"[DEBUG][INFO] Data Matrix: {round(X_train.nbytes / (1024 * 1000.0),3)} MB\n")
    pixels = Y_train.flatten().reshape(train_cpt_filtered, IMG_HEIGHT*IMG_WIDTH)
    weights_train = pixels.copy()
    pixels = np.expand_dims(pixels, axis = -1)
    eprint(f"Data Read is Done!")

    return X_train, pixels

def train_model(X, y, configurations):
    
    # Parameters - IMG
    IMG_HEIGHT = int(configurations.size_img)
    IMG_WIDTH = int(configurations.size_img)
    IMG_CHANNELS = 3
    
    # Parameters - Model
    lr_rate = float(configurations.learning_rate)
    model_name = str(configurations.model_name)
    model_type = str(configurations.model_type)
    dir_write = mkdir_if_not_exist(str(configurations.dir_write))
    activation = str(configurations.activation)
    batch_size = int(configurations.batch_size)
    epochs = int(configurations.epoch)
    dropout_ratio = float(configurations.dropout_ratio)
    dropout_level = int(configurations.dropout_level)
    model_string = str(configurations.model_string)
    eprint(f"[INFO][train_model] {model_string}")

    # Free up RAM in case the model definition cells were run multiple times
    K.clear_session()
    # Stop training when a monitoring quantity has stopped improving
    # earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    
    # Initialize the model
    if model_type.lower() == 'resunet':
        model = unetModel_residual(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, dropout_ratio=dropout_ratio, \
            lr_rate=lr_rate, activation=activation, dropout_level=dropout_level)
    else:
        model = unetModel_basic_4(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, dropout_ratio=dropout_ratio, \
            lr_rate=lr_rate, activation=activation, dropout_level=dropout_level)


    # Save the model after every epoch
    checkpointer = ModelCheckpoint(dir_write + "/" + model_string + '_main_modelCheckpoint.h5', verbose=0, monitor='val_loss', \
                                   save_best_only=True, save_weights_only=False, period=1, mode='auto')

    # Log training
    csv_logger = CSVLogger('{}/log_{}.training.csv'.format(dir_write, model_string))
    # Reduce lr_rate on plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=10, verbose=0, mode='max', cooldown=1, min_lr=0.000001)
    # Early stopping with patience
    earlystopping = EarlyStopping(monitor='val_dice_coef', patience=25, mode='max')
    
    # Fit model
    eprint("[INFO][train_model] Model Fit...")
    results = model.fit(X, y, validation_split=0.2, batch_size=batch_size, epochs=epochs,
                    callbacks=[checkpointer, csv_logger, reduce_lr, earlystopping], verbose=1, shuffle=True)#, sample_weight=weights_train)
    eprint("[INFO][train_model] Model Fit Done!")

    # Write model history to the file
    pd.DataFrame(results.history).to_csv(dir_write + "history_" + model_string + ".csv")
    
    return model, results

def save_model(model, configurations):

    # Parameters
    model_string = configurations.model_string
    dir_write = configurations.dir_write

    # Serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(dir_write, model_string + ".json"), "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to H5
    fname_saved = os.path.join(dir_write, model_string + ".h5")
    model.save_weights(fname_saved)
    print(f"Saved model: {fname_saved} to disk")


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help="Data from the Platform", required=True)
    parser.add_argument('--config_file', help="Parameters of the model", required=True)

    args = parser.parse_args()

    # Parameters
    data_folder = args.data_folder
    config_file = args.config_file

    # Configurations
    # SETUP_PATH = 'config/configuration_train.yml'
    configurations = Configurations(config_file)
    print(''.join("%s:\t%s\n" % item for item in vars(configurations).items()))
    
    # Name model with configuration parameters
    # naming_config = '_'.join(list([key + "-" + str(configurations[key]) for key in configurations])[1:]) 
    # configurations['model_name'] = naming_config
    
    os.environ["CUDA_VISIBLE_DEVICES"] = configurations.gpu_no

    # Data
    X_train, y_train= get_data(configurations, data_folder)

    # Training
    model, history = train_model(X=X_train, y=y_train, configurations=configurations)

    # Save Model
    save_model(model, configurations)
    # Testing
    # pred  = models['NN'].predict(models['Encoder'].predict(X_test))
    # auc_score = roc_auc_score(y_test.response, np.squeeze(pred))
    # print(f"\n--\nAUC Score: {auc_score:.3f}\n--\n")

if __name__ == "__main__":
    main()
