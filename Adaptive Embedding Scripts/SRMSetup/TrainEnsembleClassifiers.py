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

def LoadAndBalanceFeatures(coverPath, stegoPath):
    xCover = np.load(coverPath)
    xStego = np.load(stegoPath)
    
    print(f"Original shapes -> Cover: {xCover.shape}, Stego: {xStego.shape}")

    xCover = np.ascontiguousarray(xCover)
    xStego = np.ascontiguousarray(xStego)
    
    return xCover, xStego

SETUP_PATHS = GetPaths()
SRM = SETUP_PATHS["SRM"]

training = input("Training mode ? (Y/N) : ").upper() == "Y"
if(training):
    flag = "Training"
else:
    flag = "Testing"

settings = [
    #Classifiers-E
    {
        "Cover NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Covers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "ExistingLSBRStegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-E Models"),
        "Label"             : "Classifiers-E-LSBR"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Covers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "ExistingLSBMStegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-E Models"),
        "Label"             : "Classifiers-E-LSBM"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Covers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "ExistingPPMStegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-E Models"),
        "Label"             : "Classifiers-E-PPM"
    },
    
    
    #Classifiers-C
    {
        "Cover NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Covers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Composite32x32Stegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-C Models"),
        "Label"             : "Classifiers-C-32"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Covers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Composite64x64Stegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-C Models"),
        "Label"             : "Classifiers-C-64"
    },
    {
        "Cover NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Covers.npy"),
        "Stego NPY Path"    : os.path.join(SRM, f"Feature Matrices - {flag}", "Composite128x128Stegos.npy"),
        "Output Folder"     : os.path.join(SRM, "Classifiers-C Models"),
        "Label"             : "Classifiers-C-128"
    }
]

for setting in settings:

    if(training):
        xCoverTrain, xStegoTrain = LoadAndBalanceFeatures(setting["Cover NPY Path"], setting["Stego NPY Path"])

        print(f"Training samples per class: {len(xCoverTrain)}")

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

    else:
        with open(os.path.join(setting["Output Folder"], f"TrainedSRMEnsemble-{setting["Label"]}.pkl"), "rb") as f:
            trainedEnsemble = pickle.load(f)
        
        xCoverTest, xStegoTest = LoadAndBalanceFeatures(setting["Cover NPY Path"], setting["Stego NPY Path"])
        
        xTest = np.concatenate((xCoverTest, xStegoTest), axis=0)
        yTest = np.concatenate((
            -np.ones(len(xCoverTest)), 
            +np.ones(len(xStegoTest))
        ), axis=0)
    
        testAccuracy = trainedEnsemble.score(xTest, yTest)    
        print(f"Test Accuracy for {setting["Label"]}: {testAccuracy * 100:.3f}%")