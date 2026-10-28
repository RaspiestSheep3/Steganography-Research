import os
import pickle
import numpy as np
import sealwatch as sw

"""
This code is used to train 3 SRM models based on the feature matrices extracted by GenerateFeatureMatrices.py
Running it once generates 1 SRM model
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

coverPath = input("Cover : ")
stegoPath = input("Stego : ")
output = input("Output : ")
label = input("Label : ")

xCover, xStego = LoadAndBalanceFeatures(coverPath, stegoPath, samplesPerClass=1000)

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
    verbose=1
)

print("Training ensemble classifier...")
trainedEnsemble, trainingRecords = ensembleTrainer.train()

modelFilePath = os.path.join(output, f"TrainedSRMEnsemble-{label}.pkl")
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