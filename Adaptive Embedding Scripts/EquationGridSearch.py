#Step 1 - Find the maximum Chi2, Zhang, MSE score for a given cover block size across first 1k images
#Step 2 - Do a grid search

#Step 1
import os
import pickle
import numpy as np
from math import inf
from datetime import datetime
from Helpers.HelperFunctions import *
from concurrent.futures import ProcessPoolExecutor
from Helpers.CompositeImplementation import CompositeMethod
from SRMSetup.UseSRMModel import ExtractSingleImageFeatures, GetEnsembleProbabilities
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, MSE

#Step 1
STEP_1_NUM_COVERS = 100

#Step 2
#!Output order = c,z,m
INITIAL_NUM_COVERS = 5
SECONDARY_NUM_COVERS = 50
NUM_COMBINATIONS_FOR_SECONDARY = 5
GRANULARITY_THRESHOLD = 1e-4
BLOCK_SIZES = [32, 64, 128]

BPP = 0.2
MAX_CPU_WORKERS = 10

def ConvertSeconds(seconds):
    minutes, secondsRemaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)} min {round(secondsRemaining)} sec"

def StepOne(blockSizes):
    for blockSize in blockSizes:
        maxChi2 = 0
        maxZhang = 0
        maxMSE = 0
        
        for i in range(1, STEP_1_NUM_COVERS + 1):
            imgPath = os.path.join(coverPath, f"{i}.png")
            blocks = SplitIntoBlocks(imgPath, blockSize)
            for row in blocks:
                for block in row:
                    
                    #MSE is odd because we have to have a second block
                    #My solution is to use all 3 methods at max bpp and take the max MSE of those
                    #We already have these from the SRM training
                    lsbrPath = os.path.join(srmFolder, "LSBR Stegos", f"{i}-1.0.png")
                    lsbmPath = os.path.join(srmFolder, "LSBM Stegos", f"{i}-1.0.png")
                    ppmPath = os.path.join(srmFolder, "PPM Stegos", f"{i}-1.0.png")
                    
                    lsbrBlock = SplitIntoBlocks(lsbrPath, blockSize)[blocks.index(row)][row.index(block)]
                    lsbmBlock = SplitIntoBlocks(lsbmPath, blockSize)[blocks.index(row)][row.index(block)]
                    ppmBlock = SplitIntoBlocks(ppmPath, blockSize)[blocks.index(row)][row.index(block)]
                    
                    maxChi2 = max(maxChi2, abs(ChiSquareAttack(lsbrBlock) - ChiSquareAttack(block)), abs(ChiSquareAttack(lsbmBlock) - ChiSquareAttack(block)), abs(ChiSquareAttack(ppmBlock) - ChiSquareAttack(block)))
                    maxZhang = max(maxZhang, abs(ZhangLSBMatching(lsbrBlock) - ZhangLSBMatching(block)), abs(ZhangLSBMatching(lsbmBlock) - ZhangLSBMatching(block)), abs(ZhangLSBMatching(ppmBlock) - ZhangLSBMatching(block)))
                    
                    maxMSE = max(maxMSE, MSE(block, lsbrBlock), MSE(block, lsbmBlock), MSE(block, ppmBlock))
        
            if(i % 10 == 0):
                print(f"{datetime.now().strftime("%H:%M:%S")} - {i}% complete for blocksize {blockSize}")
        
        print("=" * 30)
        print(f"{datetime.now().strftime("%H:%M:%S")} - Results for {blockSize} : \n    Max Chi2 : {maxChi2}\n    Max Zhang : {maxZhang}\n    Max MSE : {maxMSE}")
        print("=" * 30)    

"""
==============================
14:47:03 - Results for 32 :
    Max Zhang : 1822
    Max MSE : 0.5703125
==============================

==============================
14:47:20 - Results for 64 :
    Max Zhang : 4826
    Max MSE : 0.546875
==============================

==============================
14:47:29 - Results for 128 :
    Max Zhang : 12095
    Max MSE : 0.53082275390625
==============================
"""

#Step 2
#I claim the sum of the weights = 1

def ProcessSingleImage(args):
    try:
        w1Processed, w2Processed, w3Processed, deviationCoefficients, coverPath, i, bytesPerSection, secret, blockSize = args
        
        imgPath = os.path.join(coverPath, f"{i}.png")
        
        #No failures due to infinite threshold
        compositeStego, _, _ = CompositeMethod(
            secret, 
            imgPath, 
            bytesPerSection, 
            256 // blockSize, 
            inf, 
            True, 
            deviationCoefficients
        )
        
        #We pass the stego through the models, and get the average of their classifications
        #We then calculate the alpha for this weight distribution
        #We can measure the average MSE for this combo, and select the combo with the lowest average MSE

        """coverBlock = SplitIntoBlocks(imgPath, 256)[0][0]
        coverChi2 = ChiSquareAttack(coverBlock)
        coverZhang = ZhangLSBMatching(coverBlock)
        
        compositeChi2 = ChiSquareAttack(compositeStego)
        compositeZhang = ZhangLSBMatching(compositeStego)
        
        mse = MSE(coverBlock, compositeStego)
        
        alpha = w1Processed * abs(compositeChi2 - coverChi2) + w2Processed * abs(compositeZhang - coverZhang) + w3Processed * mse

        feature = ExtractSingleImageFeatures(compositeStego)
        
        return alpha, feature"""
        
        #!TEST FLAG
        coverBlocks = SplitIntoBlocks(imgPath, 256)
        compositeBlocks = SplitIntoBlocks(None, 256, compositeStego)
        
        chi2Deltas = []
        zhangDeltas = []
        mseVals = []

        for rIdx, row in enumerate(coverBlocks):
            for cIdx, coverBlock in enumerate(row):
                compBlock = compositeBlocks[rIdx][cIdx]
                
                coverChi2 = ChiSquareAttack(coverBlock)
                compChi2 = ChiSquareAttack(compBlock)
                chi2Deltas.append(abs(compChi2 - coverChi2))
                
                coverZhang = ZhangLSBMatching(coverBlock)
                compZhang = ZhangLSBMatching(compBlock)
                zhangDeltas.append(abs(compZhang - coverZhang))
                
                mseVals.append(MSE(coverBlock, compBlock))
                
        avgChi2Delta = sum(chi2Deltas) / len(chi2Deltas)
        avgZhangDelta = sum(zhangDeltas) / len(zhangDeltas)
        avgMse = sum(mseVals) / len(mseVals)
        
        alpha = w1Processed * avgChi2Delta + w2Processed * avgZhangDelta + w3Processed * avgMse

        feature = ExtractSingleImageFeatures(compositeStego)
        
        return alpha, feature
    except Exception as e:
        print(F"Error : {e}")
        assert 1==2

def StepTwoProcessing(w1, w2, w3, w1Processed, w2Processed, w3Processed, blockSize, lsbrModel, lsbmModel, ppmModel, secret, MSEs, numCovers, coverPath):
    #Formatting the coefficients into the way CompositeMethod expects
    deviationCoefficients = {
        ("Chi Square Attack", 256 // blockSize) : w1Processed,
        ("Zhang", 256 // blockSize) : w2Processed,
        ("MSE", 256 // blockSize) : w3Processed
    }
    
    alphas = []
    features = []   
    bytesPerSection = int(blockSize * blockSize * BPP / 8) #1bpp for the given block size
    
    inputs = list([(  
        w1Processed,
        w2Processed,
        w3Processed,
        deviationCoefficients,
        coverPath,
        i,
        bytesPerSection,
        secret,
        blockSize
    ) for i in range(1, numCovers + 1)])
    
    results = []    
    with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as exe:
        results = exe.map(ProcessSingleImage, inputs)
    
    for result in results:
        alphas.append(result[0])
        features.append(result[1])        

    lsbrPredictions = GetEnsembleProbabilities(lsbrModel, features)
    lsbmPredictions = GetEnsembleProbabilities(lsbmModel, features)
    ppmPredictions = GetEnsembleProbabilities(ppmModel, features)
    
    #MSE calculation
    comboMSEs = []
    
    for i in range(numCovers):
        avgPrediction = (lsbrPredictions[i] + lsbmPredictions[i] + ppmPredictions[i]) / 3
        
        #Normalising avgPrediction from [-1, 1] to [0,1]
        avgPrediction /= 2
        avgPrediction += 0.5 
        #print(f"\nAvg Prediction : {avgPrediction}")
        
        mse = (alphas[i] - avgPrediction) ** 2
        comboMSEs.append(mse)
    
    MSEs[(w1, w2, w3)] = sum(comboMSEs) / len(comboMSEs)
    
    return MSEs, min(alphas), max(alphas)

def CalculateTotalIterations(granularity, w1Start, w1End, w2Start, w2End):
    primaryIters = 0
    for w1 in range(w1Start, w1End):
        for w2 in range(w2Start, w2End):
            w3 = int(round(1 / granularity)) - w1 - w2
            if w3 >= 0 and w2 >= 0 and w1 >= 0:
                primaryIters += 1
                
    secondaryIters = int(SECONDARY_NUM_COVERS / INITIAL_NUM_COVERS) * NUM_COMBINATIONS_FOR_SECONDARY
    currentLevelIters = primaryIters + secondaryIters
    
    newGranularity = granularity / 10
    if newGranularity > GRANULARITY_THRESHOLD:
        return currentLevelIters + CalculateTotalIterations(newGranularity, 0, 11, 0, 10)
    return currentLevelIters

def StepTwo(processNumber = 1, blockSizes = [], granularity = 0.1, w1Start = 0, w1End = 11, w2Start = 0, w2End = 10, srmFolder = "", lipsum = "", coverPath = "", outputFolder = "", count = None, index = None, timesTaken = None):
    maxValues = {
        #Done by blocks per side
        8 : {
            "Chi2" : 0.9974362477030662,
            "Zhang" : 1822,
            "MSE" : 0.5703125
        },
        4 : {
            "Chi2" : 0.988187934182378,
            "Zhang" : 4826,
            "MSE" : 0.546875
        },
        2 : {
            "Chi2" : 0.952416529303949,
            "Zhang" : 12095,
            "MSE" : 0.53082275390625
        },
    }

    with open(os.path.join(srmFolder, "SRM Models", "LSBR.pkl"), "rb") as f:
        lsbrModel = pickle.load(f)
    with open(os.path.join(srmFolder, "SRM Models", "LSBM.pkl"), "rb") as f:
        lsbmModel = pickle.load(f)
    with open(os.path.join(srmFolder, "SRM Models", "PPM.pkl"), "rb") as f:
        ppmModel = pickle.load(f)

    if(count == None):
        # Gemini : Updated count calculation using the recursive helper function
        singleBlockTotal = CalculateTotalIterations(granularity, w1Start, w1End, w2Start, w2End)
        count = len(blockSizes) * singleBlockTotal
        
        index = 1
        timesTaken = []

    #Operating at 1bpp
    with open(lipsum, "r") as f:
        secret = f.read(int(256 * 256 * BPP / 8)) #1bpp for a 256*256 image

    print(f"{datetime.now().strftime("%H:%M:%S")} - Starting Loop (Granularity: {granularity})")
    for blockSize in blockSizes:
        
        MSEs = dict()
        
        for w1 in range(w1Start, w1End):
            for w2 in range(w2Start, w2End):
                startTime = datetime.now()
                
                w3 = int(1 / granularity) - w1 - w2
                
                if not(w3 >= 0 and w2 >= 0 and w1 >= 0):
                    continue
                
                #Normalising to between 0 and 1, then dividing by max
                w1Processed = np.float64(w1) / (int(1 / granularity) * maxValues[256 // blockSize]["Chi2"])
                w2Processed = np.float64(w2) / (int(1 / granularity) * maxValues[256 // blockSize]["Zhang"])
                w3Processed = np.float64(w3) / (int(1 / granularity) * maxValues[256 // blockSize]["MSE"])
                
                MSEs, minAlpha, maxAlpha = StepTwoProcessing(
                    w1,
                    w2,
                    w3,
                    w1Processed,
                    w2Processed,
                    w3Processed,
                    blockSize,
                    lsbrModel,
                    lsbmModel,
                    ppmModel,
                    secret,
                    MSEs,
                    INITIAL_NUM_COVERS,
                    coverPath
                )

                timesTaken.append((datetime.now() - startTime).total_seconds())
        
                print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Process {processNumber} - Completed {index} / {count} w/ Granularity {granularity} (ETA = {ConvertSeconds(sum(timesTaken[-20:]) / len(timesTaken[-20:]) * (count - index))}) | Min alpha : {minAlpha}, Max alpha : {maxAlpha}", end="", flush=True)
                index += 1
        
        index -= 1
        print(f"\r\033[K{datetime.now().strftime("%H:%M:%S")} - Process {processNumber} - Beginning Second Pass For Granularity {granularity}", end="", flush=True)
        
        sortedMSEs = sorted(list(MSEs.values()))
        newMSEs = dict()
        for i in range(NUM_COMBINATIONS_FOR_SECONDARY):
            (w1, w2, w3) = list(MSEs.keys())[list(MSEs.values()).index(sortedMSEs[i])]
            
            #Normalising to between 0 and 1, then dividing by max
            w1Processed = w1 / (int(1 / granularity) * maxValues[256 // blockSize]["Chi2"])
            w2Processed = w2 / (int(1 / granularity) * maxValues[256 // blockSize]["Zhang"])
            w3Processed = w3 / (int(1 / granularity) * maxValues[256 // blockSize]["MSE"])
            
            newMSEs, minAlpha, maxAlpha = StepTwoProcessing(
                w1,
                w2,
                w3,
                w1Processed,
                w2Processed,
                w3Processed,
                blockSize,
                lsbrModel,
                lsbmModel,
                ppmModel,
                secret,
                newMSEs,
                SECONDARY_NUM_COVERS,
                coverPath
            )
            
            index += int(SECONDARY_NUM_COVERS / INITIAL_NUM_COVERS)
            print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Process {processNumber} - Completed {index} / {count} w/ Granularity {granularity} (ETA = {ConvertSeconds(sum(timesTaken[-20:]) / len(timesTaken[-20:]) * (count - index))}) | Min alpha : {minAlpha}, Max alpha : {maxAlpha}", end="", flush=True)
        
        smallestMSE = min(list(newMSEs.values()))
        optimalComboUnscaled = list(newMSEs.keys())[list(newMSEs.values()).index(smallestMSE)]
        
        optimalCombo = list(optimalComboUnscaled)
        for i in range(len(optimalCombo)):
            optimalCombo[i] = optimalCombo[i] * granularity
        
        print()
        print("=" * 30)
        print(f"For Block Size = {blockSize} w/ Granularity = {granularity}:\n    Combo : {optimalCombo}\n    MSE : {smallestMSE}")
        print("=" * 30)

        with open(os.path.join(outputFolder, f"{processNumber}.log"), "a") as f:
            f.write(f"For Block Size = {blockSize} w/ Granularity = {granularity}:\n    Combo : {optimalCombo}\n    MSE : {smallestMSE}\n")

        
        halfGranularity = granularity / 2
        newGranularity = granularity / 10
        
        optW1Value = optimalComboUnscaled[0] * granularity
        optW2Value = optimalComboUnscaled[1] * granularity
        
        w1Start2 = round((optW1Value - halfGranularity) / newGranularity)
        w1End2 = round((optW1Value + halfGranularity) / newGranularity) + 1
        w2Start2 = round((optW2Value - halfGranularity) / newGranularity)
        w2End2 = round((optW2Value + halfGranularity) / newGranularity) + 1
        
        if newGranularity > GRANULARITY_THRESHOLD:
            index = StepTwo(
                processNumber=processNumber,
                blockSizes=[blockSize],
                granularity=newGranularity,
                w1Start=w1Start2,
                w1End=w1End2,
                w2Start=w2Start2,
                w2End=w2End2,
                srmFolder=srmFolder,
                lipsum=lipsum,
                coverPath=coverPath,
                outputFolder=outputFolder,
                count=count,
                index=index,
                timesTaken=timesTaken
            )
    
    return index
        

if __name__ == '__main__':
    setupPaths = GetPaths()
    coverPath = setupPaths["Bossbase Set A"]
    srmFolder = setupPaths["SRM"]
    outputFolder = setupPaths["Equation Grid Search Logs"]
    lipsum = setupPaths["Lipsum"]
    
    #StepOne(BLOCK_SIZES)
    
    StepTwo(
        blockSizes=BLOCK_SIZES,
        granularity=0.1,
        srmFolder=srmFolder,
        lipsum=lipsum,
        coverPath=coverPath,
        outputFolder=outputFolder
    )