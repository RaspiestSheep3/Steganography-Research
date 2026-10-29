import os
import sys

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

import pickle
import numpy as np
import sealwatch as sw
from Helpers.HelperFunctions import GetPaths

"""
This code is used to train 3 SRM models based on the feature matrices extracted by GenerateFeatureMatrices.py
Running it once generates 1 SRM model
When calling this script from the command line, call it from within Adaptive Embedding Scripts as opposed to SRMSetup.
    - This is so GetPaths() works properly
"""

def LoadAndBalanceFeatures(coverPath, stegoPath, samplesPerClass=1000):
    xCover = np.load(coverPath)
    xStego = np.load(stegoPath)
    
    print(f"Original shapes -> Cover: {xCover.shape}, Stego: {xStego.shape}")
    
    if len(xStego) > samplesPerClass:
        xStego = xStego[:samplesPerClass]
        
    if len(xCover) > samplesPerClass:
        xCover = xCover[:samplesPerClass]

    xCover = np.ascontiguousarray(xCover)
    xStego = np.ascontiguousarray(xStego)
    
    return xCover, xStego

SETUP_PATHS = GetPaths()
SRM = SETUP_PATHS["SRM"]

settings = [
    #Classifiers-A
    {
        "Cover NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetACovers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetALSBRStegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-A Models"),
        "Label"             : "Classifiers-A-LSBR"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetACovers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetALSBMStegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-A Models"),
        "Label"             : "Classifiers-A-LSBM"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetACovers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetAPPMStegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-A Models"),
        "Label"             : "Classifiers-A-PPM"
    },
    
    
    #Classifiers-C
    {
        "Cover NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetCCovers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetC32x32Stegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-C Models"),
        "Label"             : "Classifiers-C-32"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetCCovers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetC64x64Stegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-C Models"),
        "Label"             : "Classifiers-C-64"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetCCovers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, "Feature Matrices", "SetC128x128Stegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-C Models"),
        "Label"             : "Classifiers-C-128"
    }
]

for setting in settings:

    xCover, xStego = LoadAndBalanceFeatures(setting["Cover NPY Path"], setting["Stego NPY Path"], samplesPerClass=1000)

    splitIdxCover = int(0.7 * len(xCover))
    splitIdxStego = int(0.7 * len(xStego))

    xCoverTrain, xCoverTest = xCover[:splitIdxCover], xCover[splitIdxCover:]
    xStegoTrain, xStegoTest = xStego[:splitIdxStego], xStego[splitIdxStego:]

    print(f"Training samples per class: {len(xCoverTrain)}")
    print(f"Testing samples per class: {len(xCoverTest)}")

    ensembleTrainer = sw.ensemble_classifier.FldEnsembleTrainer(
        Xc=xCoverTrain, 
        Xs=xStegoTrain, 
        seed=42,
        verbose=0
    )

    print("Training ensemble classifier...")
    trainedEnsemble, trainingRecords = ensembleTrainer.train()

    modelFilePath = os.path.join(setting["Output Folder"], f"TrainedSRMEnsemble-{setting["Label"]}.pkl")
    with open(modelFilePath, "wb") as modelFile:
        pickle.dump(trainedEnsemble, modelFile)
    print(f"Successfully saved trained ensemble model to: {modelFilePath}")

    xTest = np.concatenate((xCoverTest, xStegoTest), axis=0)
    yTest = np.concatenate((
        -np.ones(len(xCoverTest)), 
        +np.ones(len(xStegoTest))
    ), axis=0)

    testAccuracy = trainedEnsemble.score(xTest, yTest)
    errorRate = 1.0 - testAccuracy

    print(f"Test Accuracy: {testAccuracy * 100:.2f}%")
    print(f"P_E / Error Rate: {errorRate * 100:.2f}%")