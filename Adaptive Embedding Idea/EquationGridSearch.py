#Step 1 - Find the maximum Chi2, Zhang, MSE score for a given cover block size across first 1k images
#Step 2 - Do a grid search

#Step 1
import os
import pickle
from math import inf
from datetime import datetime
from Helpers.HelperFunctions import *
from concurrent.futures import ProcessPoolExecutor
from Helpers.TestingScriptFuncVersion import CompositeMethod
from SRMSetup.UseSRMModel import PredictNewImages, ExtractSingleImageFeatures
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, MSE

STEP_1_NUM_COVERS = 100
INITIAL_NUM_COVERS = 10
SECONDARY_NUM_COVERS = 100
NUM_COMBINATIONS_FOR_SECONDARY = 10
GRANULARITY_THRESHOLD = 1e-8

if __name__ == '__main__':
    coverPath = input("Cover : ").strip()
    srmFolder = input("SRM : ").strip()
    outputFolder = input("Output : ").strip()
    lipsum = input("Lipsum : ").strip()

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
                    maxChi2 = max(maxChi2, ChiSquareAttack(block))
                    maxZhang = max(maxZhang, ZhangLSBMatching(block))
                    
                    #MSE is odd because we have to have a second block
                    #My solution is to use all 3 methods at max bpp and take the max MSE of those
                    #We already have these from the SRM training
                    lsbrPath = os.path.join(srmFolder, "LSBR Stegos", f"{i}-1.0.png")
                    lsbmPath = os.path.join(srmFolder, "LSBM Stegos", f"{i}-1.0.png")
                    ppmPath = os.path.join(srmFolder, "PPM Stegos", f"{i}-1.0.png")
                    
                    lsbrBlock = SplitIntoBlocks(lsbrPath, blockSize)[blocks.index(row)][row.index(block)]
                    lsbmBlock = SplitIntoBlocks(lsbmPath, blockSize)[blocks.index(row)][row.index(block)]
                    ppmBlock = SplitIntoBlocks(ppmPath, blockSize)[blocks.index(row)][row.index(block)]
                    
                    maxMSE = max(maxMSE, MSE(block, lsbrBlock), MSE(block, lsbmBlock), MSE(block, ppmBlock))
        
            if(i % 10 == 0):
                print(f"{datetime.now().strftime("%H:%M:%S")} - {i}% complete for blocksize {blockSize}")
        
        print("=" * 30)
        print(f"{datetime.now().strftime("%H:%M:%S")} - Results for {blockSize} : \n    Max Chi2 : {maxChi2}\n    Max Zhang : {maxZhang}\n    Max MSE : {maxMSE}")
        print("=" * 30)    

#Results for 16 : 
#    Max Chi2 : 256.0
#    Max Zhang : 720
#    Max MSE : 0.6328125

#Results for 32 :
#    Max Chi2 : 1024.0
#    Max Zhang : 2004
#    Max MSE : 0.5703125

#Results for 64 :
#    Max Chi2 : 4096.0
#    Max Zhang : 4097
#    Max MSE : 0.546875

#Results for 128 :
#    Max Chi2 : 10345.668017755757
#    Max Zhang : 7343
#    Max MSE : 0.53082275390625

#Step 2
#I claim the sum of the weights = 1

def StepTwoProcessing(w1, w2, w3, w1Processed, w2Processed, w3Processed, blockSize, lsbrModel, lsbmModel, ppmModel, secret, MSEs, numCovers, coverPath):
    #Formatting the coefficients into the way CompositeMethod expects
    deviationCoefficients = {
        ("Chi Square Attack", 256 // blockSize) : w1Processed,
        ("Zhang", 256 // blockSize) : w2Processed,
        "MSE" : w3Processed
    }
    
    alphas = []
    compositeStegos = []
    for i in range(1, numCovers + 1):
        imgPath = os.path.join(coverPath, f"{i}.png")
        
        bytesPerSection = int(blockSize * blockSize / 8) #1bpp for the given block size
        
        #For this there are no failures (threshold is infinite)
        compositeStego, _, _ = CompositeMethod(
            secret, 
            imgPath, 
            bytesPerSection, 
            256 // blockSize, 
            inf, 
            True, 
            deviationCoefficients
        )
        compositeStegos.append(compositeStego)
        
        #We pass the stego through the models, and get the average of their classifications
        #We then calculate the alpha for this weight distribution
        #We can measure the average MSE for this combo, and select the combo with the lowest average MSE

        coverBlock = SplitIntoBlocks(imgPath, 256)[0][0]
        coverChi2 = ChiSquareAttack(coverBlock)
        coverZhang = ZhangLSBMatching(coverBlock)
        
        compositeChi2 = ChiSquareAttack(compositeStego)
        compositeZhang = ZhangLSBMatching(compositeStego)
        
        mse = MSE(coverBlock, compositeStego)
        
        alpha = w1Processed * abs(compositeChi2 - coverChi2) + w2Processed * abs(compositeZhang - coverZhang) + w3Processed * mse
        alphas.append(alpha)
        
        #print(f"{datetime.now().strftime("%H:%M:%S")} - TEMP flag - {i}")

    #Generating the features once to save on speed
    features = []
    for compositeStego in compositeStegos:
        features.append(ExtractSingleImageFeatures(compositeStego))

    lsbrPredictions = PredictNewImages(lsbrModel, None, features)
    lsbmPredictions = PredictNewImages(lsbmModel, None, features)
    ppmPredictions = PredictNewImages(ppmModel, None, features)
    
    #MSE calculation
    comboMSEs = []
    
    for i in range(numCovers):
        avgPrediction = (lsbrPredictions[i] + lsbmPredictions[i] + ppmPredictions[i]) / 3
        
        #Normalising avgPrediction from [-1, 1] to [0,1]
        avgPrediction /= 2
        avgPrediction += 0.5 
        
        mse = (alphas[i] - avgPrediction) ** 2
        comboMSEs.append(mse)
    
    MSEs[(w1, w2, w3)] = sum(comboMSEs) / len(comboMSEs)
    
    return MSEs

def StepTwo(processNumber = 1, blockSizes = [], granularity = 0.1, w1Start = 1, w1End = 10, w2Start = 1, w2End = 9, srmFolder = "", lipsum = "", coverPath = "", outputFolder = ""):
    maxValues = {
        #Done by blocks per side
        16 : {
            "Chi2" : 256,
            "Zhang" : 720,
            "MSE" : 0.6328125
        },
        8 : {
            "Chi2" : 1024,
            "Zhang" : 2004,
            "MSE" : 0.5703125
        },
        4 : {
            "Chi2" : 4096,
            "Zhang" : 4097,
            "MSE" : 0.546875
        },
        2 : {
            "Chi2" : 10345.668017755757,
            "Zhang" : 7343,
            "MSE" : 0.53082275390625
        },
    }

    with open(os.path.join(srmFolder, "SRM Models", "LSBR.pkl"), "rb") as f:
        lsbrModel = pickle.load(f)
    with open(os.path.join(srmFolder, "SRM Models", "LSBM.pkl"), "rb") as f:
        lsbmModel = pickle.load(f)
    with open(os.path.join(srmFolder, "SRM Models", "PPM.pkl"), "rb") as f:
        ppmModel = pickle.load(f)

    primaryIters = 0
    for w1 in range(w1Start, w1End):
        for w2 in range(w2Start, w2End):
            w3 = int(1 / granularity) - w1 - w2
            if w3 >= 0:
                primaryIters += 1

    secondaryIters = int(SECONDARY_NUM_COVERS / INITIAL_NUM_COVERS) * NUM_COMBINATIONS_FOR_SECONDARY

    count = len(blockSizes) * (primaryIters + secondaryIters)
    
    
    index = 1
    timesTaken = []

    #Operating at 1bpp
    with open(lipsum, "r") as f:
        secret = f.read(int(256 * 256 / 8)) #1bpp for a 256*256 image

    print(f"{datetime.now().strftime("%H:%M:%S")} - Starting Loop (Granularity: {granularity})")
    for blockSize in blockSizes:
        
        MSEs = dict()
        
        for w1 in range(w1Start, w1End):
            for w2 in range(w2Start, w2End):
                startTime = datetime.now()
                
                w3 = int(1 / granularity) - w1 - w2
                
                if w3 < 0:
                    continue
                
                #Normalising to between 0 and 1, then dividing by max
                w1Processed = w1 / (int(1 / granularity) * maxValues[256 // blockSize]["Chi2"])
                w2Processed = w2 / (int(1 / granularity) * maxValues[256 // blockSize]["Zhang"])
                w3Processed = w3 / (int(1 / granularity) * maxValues[256 // blockSize]["MSE"])
                
                MSEs = StepTwoProcessing(
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
        
                print(f"{datetime.now().strftime("%H:%M:%S")} - Process {processNumber} - Completed {index} / {count} (ETA = {ConvertSeconds(sum(timesTaken[-20:]) / len(timesTaken[-20:]) * (count - index))})")
                index += 1
        
        index -= 1
        print(f"{datetime.now().strftime("%H:%M:%S")} - Process {processNumber} - Beginning Second Pass For Granularity {granularity}")
        
        sortedMSEs = sorted(list(MSEs.values()))
        newMSEs = dict()
        for i in range(NUM_COMBINATIONS_FOR_SECONDARY):
            (w1, w2, w3) = list(MSEs.keys())[list(MSEs.values()).index(sortedMSEs[i])]
            
            #Normalising to between 0 and 1, then dividing by max
            w1Processed = w1 / (int(1 / granularity) * maxValues[256 // blockSize]["Chi2"])
            w2Processed = w2 / (int(1 / granularity) * maxValues[256 // blockSize]["Zhang"])
            w3Processed = w3 / (int(1 / granularity) * maxValues[256 // blockSize]["MSE"])
            
            newMSEs = StepTwoProcessing(
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
            print(f"{datetime.now().strftime("%H:%M:%S")} - Process {processNumber} - Completed {index} / {count} (ETA = {ConvertSeconds(sum(timesTaken[-20:]) / len(timesTaken[-20:]) * (count - index))})")
        
        smallestMSE = min(list(newMSEs.values()))
        optimalComboUnscaled = list(newMSEs.keys())[list(newMSEs.values()).index(smallestMSE)]
        
        optimalCombo = list(optimalComboUnscaled)
        for i in range(len(optimalCombo)):
            optimalCombo[i] = optimalCombo[i] * granularity
        
        print("=" * 30)
        print(f"For Block Size = {blockSize} w/ Granularity = {granularity}:\n    Combo : {optimalCombo}\n    MSE : {smallestMSE}")
        print("=" * 30)

        with open(os.path.join(outputFolder, f"{processNumber}.log"), "a") as f:
            f.write(f"For Block Size = {blockSize} w/ Granularity = {granularity}:\n    Combo : {optimalCombo}\n    MSE : {smallestMSE}\n")

        
        halfGranularity = granularity / 2
        newGranularity = granularity / 10
        
        optW1Value = optimalComboUnscaled[0] * granularity
        optW2Value = optimalComboUnscaled[1] * granularity
        
        w1Start = round((optW1Value - halfGranularity) / newGranularity)
        w1End = round((optW1Value + halfGranularity) / newGranularity) + 1
        w2Start = round((optW2Value - halfGranularity) / newGranularity)
        w2End = round((optW2Value + halfGranularity) / newGranularity) + 1
        
        if newGranularity >= GRANULARITY_THRESHOLD:
            StepTwo(
                processNumber=processNumber,
                blockSizes=[blockSize],
                granularity=newGranularity,
                w1Start=w1Start,
                w1End=w1End,
                w2Start=w2Start,
                w2End=w2End,
                srmFolder=srmFolder,
                lipsum=lipsum,
                coverPath=coverPath,
                outputFolder=outputFolder
            )
        

if __name__ == '__main__':
    processConfigs = [
        {"Process Number": 1, "w1Start": 1, "w1End": 5, "Block Size": 16},
        {"Process Number": 2, "w1Start": 1, "w1End": 5, "Block Size": 32},
        {"Process Number": 3, "w1Start": 1, "w1End": 5, "Block Size": 64},
        {"Process Number": 4, "w1Start": 1, "w1End": 5, "Block Size": 128},
        {"Process Number": 5, "w1Start": 5, "w1End": 10, "Block Size": 16},
        {"Process Number": 6, "w1Start": 5, "w1End": 10, "Block Size": 32},
        {"Process Number": 7, "w1Start": 5, "w1End": 10, "Block Size": 64},
        {"Process Number": 8, "w1Start": 5, "w1End": 10, "Block Size": 128},
    ]

    print(f"{datetime.now().strftime('%H:%M:%S')} - Starting Grid Search")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                StepTwo,
                processNumber=config["Process Number"],
                blockSizes=[config["Block Size"]],
                granularity=0.1,
                w1Start=config["w1Start"],
                w1End=config["w1End"],
                srmFolder=srmFolder,
                lipsum=lipsum,
                coverPath=coverPath,
                outputFolder=outputFolder
            )
            for config in processConfigs
        ]
        
        for future in futures:
            future.result()

    print(f"{datetime.now().strftime('%H:%M:%S')} - All Systems Complete")