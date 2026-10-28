import sys
import os

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

import pickle
import numpy as np
import sealwatch as sw
from datetime import datetime
from Helpers.HelperFunctions import SplitIntoBlocks

"""
This code extracts SRM features and uses the trained SRM models to make predictions about images
"""

def ExtractSingleImageFeatures(imgRawArray, useQ1 = True):
    try:
        imageArray = np.array(imgRawArray, dtype=np.float32)
        #imageArray.flatten()
        
        if(useQ1):
            features = sw.srmq1.extract(imageArray)
        else:
            features = sw.srm.extract(imageArray)
            
        if isinstance(features, dict) or hasattr(features, 'values'):
            features = np.concatenate([np.ravel(submodelFeatures) for submodelFeatures in features.values()])
        elif isinstance(features, np.ndarray):
            features = features.flatten()
            
        return features
    except Exception as e:
        print(f"Error {e}")
        return None

def PredictNewImages(trainedEnsemble, imgRawArrays, extractedFeatureList = None):
    
    if(extractedFeatureList == None):
        extractedFeatureList = []
        for imgRawArray in imgRawArrays:
            features = ExtractSingleImageFeatures(imgRawArray)
            if features is not None:
                extractedFeatureList.append(features)
        
    featureMatrix = np.array(extractedFeatureList, dtype=np.float32)
    featureMatrix = np.ascontiguousarray(featureMatrix)
 
    #stego -> +1, cover -> -1   
    predictions = trainedEnsemble.predict(featureMatrix)

    return list(predictions)

def GetEnsembleProbabilities(model, extractedFeatureList):
    featureMatrix = np.array(extractedFeatureList, dtype=np.float32)
    featureMatrix = np.ascontiguousarray(featureMatrix)
    
    #stego -> +1, cover -> -1   
    predictions = model.predict_confidence(featureMatrix)

    return list(predictions)

if __name__ == '__main__':
    modelPath = input("Model Path : ")
    
    with open(modelPath, "rb") as f:
        model = pickle.load(f)
    
    start = datetime.now()
    
    testImages = [
        
        #SplitIntoBlocks(r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Idea\SRM Setup\PPM Stegos\1-0.2.png", 256)[0][0],
        #SplitIntoBlocks(r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Idea\SRM Setup\PPM Stegos\2-0.2.png", 256)[0][0],
        SplitIntoBlocks(r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Bossbase Dataset 1-1000\1.png", 256)[0][0]
    ]
    
    timeTaken = (datetime.now() - start).total_seconds()
    
    print(PredictNewImages(model, testImages))
    print(f"Time taken : {timeTaken}")