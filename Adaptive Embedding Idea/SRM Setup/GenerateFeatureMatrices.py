import os
import glob
import numpy as np
from PIL import Image
import sealwatch as sw
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

def ConvertSeconds(seconds):
    minutes, secondsRemaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)} min {round(secondsRemaining)} sec"

def ProcessSingleImage(imagePath):
    try:
        pilImage = Image.open(imagePath).convert('L')
        imageArray = np.array(pilImage, dtype=np.float32)
        
        features = sw.srm.extract(imageArray)
        
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
    
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        results = list(executor.map(ProcessSingleImage, imagePaths))
        
    for result in results:
        if result is not None:
            featureList.append(result)

    featureMatrix = np.array(featureList, dtype=np.float32)
    
    np.save(outputFilePath, featureMatrix)
    
    elapsedTime = (datetime.now() - startTime).total_seconds()
    print(f"{datetime.now().strftime("%H:%M:%S")} - Finished in {ConvertSeconds(elapsedTime)}. Saved matrix w/ shape {featureMatrix.shape} to {outputFilePath}\n")
    return featureMatrix

if __name__ == '__main__':
    totalCores = 12
    targetCores = max(1, totalCores - 3)

    coverDir = input("Cover : ")
    lsbrDir = input("LSBR : ")
    lsbmDir = input("LSBM : ")
    ppmDir = input("PPM : ")
    featureMatricesFolder = input("Feature Matrices : ")

    #ExtractFeaturesFromFolder(coverDir, os.path.join(featureMatricesFolder, "CoverFeatures.npy"), targetCores)
    #ExtractFeaturesFromFolder(lsbrDir, os.path.join(featureMatricesFolder, "LSBRFeatures.npy"), targetCores)
    ExtractFeaturesFromFolder(lsbmDir, os.path.join(featureMatricesFolder, "LSBMFeatures.npy"), targetCores)
    ExtractFeaturesFromFolder(ppmDir, os.path.join(featureMatricesFolder, "PPMFeatures.npy"), targetCores)