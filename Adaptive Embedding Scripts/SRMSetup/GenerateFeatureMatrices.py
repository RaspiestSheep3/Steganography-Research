import os
import sys

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

import glob
import numpy as np
from PIL import Image
import sealwatch as sw
from datetime import datetime
from Helpers.HelperFunctions import GetPaths
from concurrent.futures import ProcessPoolExecutor

"""
This code processes the training images for the classifiers and generates the feature matrices used to train those models
"""

SETUP_PATHS = GetPaths()
SRM = SETUP_PATHS["SRM"]

MAX_WORKERS = 10

def ConvertSeconds(seconds):
    minutes, secondsRemaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)} min {round(secondsRemaining)} sec"

def ProcessSingleImage(imagePath):
    try:
        pilImage = Image.open(imagePath).convert('L')
        imageArray = np.array(pilImage, dtype=np.float32)
        
        features = sw.srmq1.extract(imageArray)
        
        if isinstance(features, dict) or hasattr(features, 'values'):
            features = np.concatenate([np.ravel(submodelFeatures) for submodelFeatures in features.values()])
        elif isinstance(features, np.ndarray):
            features = features.flatten()
            
        return features
    except Exception as e:
        print(f"Error processing {imagePath}: {e}")
        return None

def ExtractFeaturesFromFolder(imageFolder, outputFilePath, maxWorkers):
    imagePaths = glob.glob(os.path.join(imageFolder, "*.png"))
    totalImages = len(imagePaths)
    print(f"{datetime.now().strftime("%H:%M:%S")} - Processing folder: {imageFolder} ({totalImages} images) using {maxWorkers} CPU cores")
    
    featureList = []
    startTime = datetime.now()
    
    with ProcessPoolExecutor(max_workers=maxWorkers) as exe:
        results = list(exe.map(ProcessSingleImage, imagePaths))
        
    for result in results:
        if result is not None:
            featureList.append(result)

    featureMatrix = np.array(featureList, dtype=np.float32)
    
    np.save(outputFilePath, featureMatrix)
    
    elapsedTime = (datetime.now() - startTime).total_seconds()
    print(f"{datetime.now().strftime("%H:%M:%S")} - Finished in {ConvertSeconds(elapsedTime)}. Saved matrix w/ shape {featureMatrix.shape} to {outputFilePath}\n")
    return featureMatrix

if __name__ == '__main__':
    featureMatricesFolder = os.path.join(SRM, "Feature Matrices")
    os.makedirs(featureMatricesFolder, exist_ok=True)

    #Covers
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set A Covers"), os.path.join(featureMatricesFolder, "SetACovers.npy"), MAX_WORKERS)
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set C Covers"), os.path.join(featureMatricesFolder, "SetCCovers.npy"), MAX_WORKERS)
    
    #Existing methods
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set A LSBM Stegos"), os.path.join(featureMatricesFolder, "SetALSBMStegos.npy"), MAX_WORKERS)
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set A LSBR Stegos"), os.path.join(featureMatricesFolder, "SetALSBRStegos.npy"), MAX_WORKERS)
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set A PPM Stegos"), os.path.join(featureMatricesFolder, "SetAPPMStegos.npy"), MAX_WORKERS)

    #Composite Methods
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set C 32x32 Stegos"), os.path.join(featureMatricesFolder, "SetC32x32Stegos.npy"), MAX_WORKERS)
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set C 64x64 Stegos"), os.path.join(featureMatricesFolder, "SetC64x64Stegos.npy"), MAX_WORKERS)
    ExtractFeaturesFromFolder(os.path.join(SRM, "Training Data", "Set C 128x128 Stegos"), os.path.join(featureMatricesFolder, "SetC128x128Stegos.npy"), MAX_WORKERS)
