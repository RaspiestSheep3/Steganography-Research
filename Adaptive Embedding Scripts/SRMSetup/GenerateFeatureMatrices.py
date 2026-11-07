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
MODEL_DATA_TRAIN = os.path.join(SETUP_PATHS["Model Data"], "Training")
MODEL_DATA_TEST = os.path.join(SETUP_PATHS["Model Data"], "Testing")

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
    generateTrain = input("Generate Training Data (Y/N) : ").upper()
    generateTest = input("Generate Testing Data (Y/N) : ").upper()

    if(generateTrain == "Y"):
        featureMatricesFolder = os.path.join(SRM, "Feature Matrices - Training")
        os.makedirs(featureMatricesFolder, exist_ok=True)

        #Covers
        
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TRAIN, "cover"), os.path.join(featureMatricesFolder, "Covers.npy"), MAX_WORKERS)
        
        #Existing methods
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TRAIN, "Existing LSBM"), os.path.join(featureMatricesFolder, "ExistingLSBMStegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TRAIN, "Existing LSBR"), os.path.join(featureMatricesFolder, "ExistingLSBRStegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TRAIN, "Existing PPM"), os.path.join(featureMatricesFolder, "ExistingPPMStegos.npy"), MAX_WORKERS)

        #Composite Methods
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TRAIN, "Composite 32x32"), os.path.join(featureMatricesFolder, "Composite32x32Stegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TRAIN, "Composite 64x64"), os.path.join(featureMatricesFolder, "Composite64x64Stegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TRAIN, "Composite 128x128"), os.path.join(featureMatricesFolder, "Composite128x128Stegos.npy"), MAX_WORKERS)

    if(generateTest == "Y"):
        featureMatricesFolder = os.path.join(SRM, "Feature Matrices - Testing")
        os.makedirs(featureMatricesFolder, exist_ok=True)

        #Covers
        
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TEST, "cover"), os.path.join(featureMatricesFolder, "Covers.npy"), MAX_WORKERS)
        
        #Existing methods
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TEST, "Existing LSBM"), os.path.join(featureMatricesFolder, "ExistingLSBMStegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TEST, "Existing LSBR"), os.path.join(featureMatricesFolder, "ExistingLSBRStegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TEST, "Existing PPM"), os.path.join(featureMatricesFolder, "ExistingPPMStegos.npy"), MAX_WORKERS)

        #Composite Methods
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TEST, "Composite 32x32"), os.path.join(featureMatricesFolder, "Composite32x32Stegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TEST, "Composite 64x64"), os.path.join(featureMatricesFolder, "Composite64x64Stegos.npy"), MAX_WORKERS)
        ExtractFeaturesFromFolder(os.path.join(MODEL_DATA_TEST, "Composite 128x128"), os.path.join(featureMatricesFolder, "Composite128x128Stegos.npy"), MAX_WORKERS)