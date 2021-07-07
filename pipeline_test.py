
import sys
import os
import cv2
import time
import argparse
import random as rd
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Set seed value for all of frameworks
seed_value= 2021 
os.environ['PYTHONHASHSEED']=str(seed_value)
rd.seed(seed_value)
np.random.seed(seed_value)

# os.chdir('/Users/sonerkoc/Git_repos/tumor-bud-detection')

from utils.loss_functions import dice_coef_loss, dice_coef
from utils.configurations import Configurations
from utils.visualizations import generate_visuals
from utils.utils import eprint, mkdir_if_not_exist

def get_data_test(data_folder, configurations, trained_model):
    
    # Parameters
    IMG_WIDTH = configurations.size_img
    IMG_HEIGHT = configurations.size_img
    IMG_CHANNELS = 3
    TEST_PATH = data_folder
    COUNT = configurations.sample_count
    
    # Path of Image Tiles and Masks
    path = os.path.join(TEST_PATH, "img")
    path_mask = os.path.join(TEST_PATH, "mask")

    total = int(sum([len(files) for r, d, files in os.walk(path)]))
    
    eprint(f'[DEBUG][get_data_test]  Getting and Resizing({IMG_WIDTH}x{IMG_HEIGHT}) Test Images and Masks... ')

    # Get and resize Test images and masks
    test_cpt = int(sum([len(files) for r, d, files in os.walk(path)]))
    
    # X_test = np.ndarray((test_cpt, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    # Y_test = np.ndarray((test_cpt, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)  # dtype=np.bool)

    eprint(f'[DEBUG][get_data_test] Getting and Resizing Test Images and Masks Done!\nPath to img: {path}')
    sys.stdout.flush()

    _, _, files_orj = next(os.walk(path))
    _, _, files_mask = next(os.walk(path_mask))
    files_orj = sorted(files_orj)
    files_mask = sorted(files_mask)

    eprint(f'[DEBUG][get_data_test] Number of Image Tiles: {len(files_orj)}\t Number of Image Masks: {len(files_mask)}')

    # for i, f in enumerate(files_orj[:COUNT]):
    #     img = cv2.imread(os.path.join(path, f))
    #     img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
    #     img = img / 255
    #     X_test[i] = img

    # for i, fm in enumerate(files_mask[:COUNT]):
    #     img_mask = cv2.imread(os.path.join(path_mask, fm), cv2.IMREAD_GRAYSCALE)
    #     img_mask = cv2.resize(img_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
    #     img_mask = img_mask / 255
    #     img_mask = np.expand_dims(img_mask, axis=-1)
    #     Y_test[i] = img_mask
    
    for i, f in enumerate(files_orj[:COUNT]):
        X_test = np.ndarray((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

        img = cv2.imread(os.path.join(path, f))
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        img = img / 255
        X_test[0] = img
        predictions = test_model(X_test, data_folder, trained_model, configurations)

    # eprint(f'[DEBUG][get_data_test] X_test shape: {X_test.shape}\t Y_test shape: {Y_test.shape}')

    # pixels = Y_test.flatten().reshape(test_cpt, IMG_HEIGHT*IMG_WIDTH)
    # pixels = np.expand_dims(pixels, axis = -1)
    # eprint(f"[DEBUG][get_data_test] Data Read is Done!")

    # return X_test, pixels

def test_model(X, data_folder, trained_model, configurations):

    # Parameters
    IMG_HEIGHT = configurations.size_img
    IMG_WIDTH = configurations.size_img
    TEST_PATH = data_folder
    
    # Path of Image Tiles and Masks
    path = os.path.join(TEST_PATH, "img")

    _, _, files_orj = next(os.walk(path))
    files_orj = sorted(files_orj)

    # Load Trained Model
    model = load_model(trained_model, \
        custom_objects={'dice_coef':dice_coef, 'dice_coef_loss':dice_coef_loss})
    
    # Predict
    preds_test = model.predict(X)
    preds_reshaped = np.ndarray((len(preds_test), IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    
    for i in range(len(preds_test)):
        preds_reshaped[i] = preds_test[i].reshape(IMG_HEIGHT, IMG_WIDTH)

    preds_upsampled = []
    for i in range(len(preds_test)):
        preds_upsampled.append(np.expand_dims(cv2.resize(preds_reshaped[i], (IMG_HEIGHT, IMG_WIDTH)), axis=-1))
    print("[INFO] Upsampling is done!(upsampled to ({}, {}) from ({}, {})".format(IMG_HEIGHT, IMG_WIDTH, preds_test[i].shape[0], preds_test[i].shape[1]))

    output_pred = os.path.join(configurations.output_folder, 'Prediction')
    mkdir_if_not_exist(configurations.output_folder)
    mkdir_if_not_exist(output_pred)
    theshold_pred = 0.5

    for k in range(configurations.sample_count):
        img = preds_upsampled[k].copy()

        img[ img > theshold_pred] = 1
        img[ img <= theshold_pred] = 0
        img *= 255
        
        out_name = os.path.join(output_pred, "pred-" + files_orj[k])
        cv2.imwrite(out_name, img)
        
    print('[INFO] Finished Prediction!')

    return output_pred

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help="Data from the Platform", required=True)
    parser.add_argument('--trained_model', help="Trained model", required=True)
    args = parser.parse_args()
    
    # Parameters
    data_folder = args.data_folder
    trained_model = args.trained_model

    # Configurations
    SETUP_PATH = 'configuration_test.yml'
    configurations = Configurations(SETUP_PATH)

    eprint(''.join("%s:\t%s\n" % item for item in vars(configurations).items()))

    # Data
    get_data_test(data_folder, configurations, trained_model)

    # Inference
    # predictions = test_model(X_test, data_folder, trained_model, configurations)

    # for i in range(5):
    generate_visuals(data_folder, os.path.join(configurations.output_folder, 'Prediction/'), thold_area=0)
    # time.sleep(3)

if __name__ == "__main__":
    main()