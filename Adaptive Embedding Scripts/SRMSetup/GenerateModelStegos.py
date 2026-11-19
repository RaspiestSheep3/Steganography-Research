import os
import sys

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

import shutil
import numpy as np
import conseal as cl
from math import inf
from PIL import Image
from datetime import datetime
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
from Helpers.CompositeImplementation import *
from concurrent.futures import ProcessPoolExecutor

"""
This code generates the images used to train the classifiers and SRNets.
The data is split as such:
    - BOWS2 - Training data
    - BossBase 2001-5500 - TESTING data
    - BossBase 5501-9000 - VALIDATION data
"""

SETUP_PATHS = GetPaths()
LIPSUM = SETUP_PATHS["Structured Secret"]
MODEL_TRAINING_DATA = SETUP_PATHS["Model Data"]
BOWS2 = SETUP_PATHS["BOWS2"]
BOSSBASE = SETUP_PATHS["Bossbase"]
TRAIN_PATH = os.path.join(MODEL_TRAINING_DATA, "Training")
TEST_PATH = os.path.join(MODEL_TRAINING_DATA, "Testing")
VAL_PATH = os.path.join(MODEL_TRAINING_DATA, "Validation")

MAX_WORKERS = 10

timesTaken = []
timeCount = 34 * 8 #1 count = 500 paths
timeIndex = 0

def SaveImage(block, path):
    stego = Image.new("L", (256,256))          
    
    for i in range(256):
        for j in range(256):
            stego.putpixel((i, j), block[j][i])
    
    stego.save(path)

def ProcessExistingStegos(args):
    coverPathArg, setLSBMStegosPath, setLSBRStegosPath, setPPMStegosPath, secret, i, offset, count = args
    
    coverPath = os.path.join(coverPathArg, f"{i}.png")
    block = SplitIntoBlocks(coverPath, 256)[0][0]
    
    if(i <= count // 5):
        bpp = 0.2
    elif(i <= (2*count) // 5):
        bpp = 0.4
    elif(i <= (3*count) // 5):
        bpp = 0.6
    elif(i <= (4*count) // 5):
        bpp = 0.8
    else:
        bpp = 1.0  
    
    secretBytesLength = int(IMAGE_SIZE ** 2  * bpp / 8)
    trimmedSecret = secret[:secretBytesLength]
    
    lsbmBlock = LSBMatching(
        QuickDeepcopy(block),
        trimmedSecret
    )
    
    lsbrBlock = LSBReplacement(
        QuickDeepcopy(block),
        trimmedSecret
    )
    
    ppmBlock = PixelPairMatching(
        QuickDeepcopy(block),
        trimmedSecret
    )
    
    SaveImage(lsbmBlock, os.path.join(setLSBMStegosPath, f"{i - offset}.png"))
    SaveImage(lsbrBlock, os.path.join(setLSBRStegosPath, f"{i - offset}.png"))
    SaveImage(ppmBlock, os.path.join(setPPMStegosPath, f"{i - offset}.png"))

def ProcessCompositeStegos(args):
    coverPathArg, size32Path, size64Path, size128Path, secret, i, offset, count = args
    
    coverPath = os.path.join(coverPathArg, f"{i}.png")
    
    if(i < count // 5):
        bpp = 0.2
    elif(i <= (2*count) // 5):
        bpp = 0.4
    elif(i <= (3*count) // 5):
        bpp = 0.6
    elif(i <= (4*count) // 5):
        bpp = 0.8
    else:
        bpp = 1.0  
    
    secretBytesLength = int(IMAGE_SIZE ** 2  * bpp / 8)
    trimmedSecret = secret[:secretBytesLength]
    
    blocks32, _, _ = CompositeMethod(
        trimmedSecret,
        coverPath,
        round(len(trimmedSecret) / ((IMAGE_SIZE // 32) ** 2)),
        (IMAGE_SIZE // 32),
        inf
    )
    
    blocks64, _, _ = CompositeMethod(
        trimmedSecret,
        coverPath,
        round(len(trimmedSecret) / ((IMAGE_SIZE // 64) ** 2)),
        (IMAGE_SIZE // 64),
        inf
    )
    
    blocks128, _, _ = CompositeMethod(
        trimmedSecret,
        coverPath,
        round(len(trimmedSecret) / ((IMAGE_SIZE // 128) ** 2)),
        (IMAGE_SIZE // 128),
        inf
    )
    
    SaveImage(blocks32, os.path.join(size32Path, f"{i - offset}.png"))
    SaveImage(blocks64, os.path.join(size64Path, f"{i - offset}.png"))
    SaveImage(blocks128, os.path.join(size128Path, f"{i - offset}.png"))

def ProcessAdaptiveStegos(args):
    coverPathArg, hugoPath, uniwardPath, i, offset, count = args
    
    coverPath = os.path.join(coverPathArg, f"{i}.png")
    
    if(i < count // 5):
        bpp = 0.2
    elif(i <= (2*count) // 5):
        bpp = 0.4
    elif(i <= (3*count) // 5):
        bpp = 0.6
    elif(i <= (4*count) // 5):
        bpp = 0.8
    else:
        bpp = 1.0  
    
    #Adapted from https://github.com/uibk-uncover/conseal
    coverHUGO = np.array(Image.open(coverPath))
    coverUniward = np.array(Image.open(coverPath)) #Keeping different objects just in case
    
    stegoHUGO = cl.hugo.simulate_single_channel(
        x0 = coverHUGO,
        alpha=bpp,
        seed = i
    )
    
    stegoUniward = cl.suniward.simulate_single_channel(
        x0 = coverUniward,
        alpha=bpp,
        seed = i
    )
    
    Image.fromarray(stegoHUGO).save(os.path.join(hugoPath, f"{i - offset}.png"))
    Image.fromarray(stegoUniward).save(os.path.join(uniwardPath, f"{i - offset}.png"))
    
def ProcessSet(dataset, folderPath, start, count, generateExistingStegos, generateCompositeStegos, generateAdaptiveStegos):
    global timesTaken, timeIndex
    
    #Starting with the covres
    #This is a fast operation - no point multithreading
    os.makedirs(os.path.join(folderPath, "cover"), exist_ok=True)
    for i in range(start + 1, start + count + 1):
        shutil.copy(
            os.path.join(dataset, f"{i}.png"),
            os.path.join(folderPath, "cover", f"{i}.png")
        )
    
    #Dealing with the stego stuff
    for path in [
        os.path.join(folderPath, "Composite 32x32"),
        os.path.join(folderPath, "Composite 64x64"),
        os.path.join(folderPath, "Composite 128x128"),
        os.path.join(folderPath, "Existing LSBM"),
        os.path.join(folderPath, "Existing LSBR"),
        os.path.join(folderPath, "Existing PPM"),
        os.path.join(folderPath, "Adaptive HUGO"),
        os.path.join(folderPath, "Adaptive S-UNIWARD")
    ]:
        os.makedirs(path, exist_ok=True)
    
    if(generateCompositeStegos):
        compositeTaskArgs = [
            (
                dataset,
                os.path.join(folderPath, "Composite 32x32"),
                os.path.join(folderPath, "Composite 64x64"),
                os.path.join(folderPath, "Composite 128x128"),
                secret,
                i,
                start,
                count
            ) for i in range(start + 1, start + count + 1)
        ]
        
        timeStart = datetime.now()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            exe.map(ProcessCompositeStegos, compositeTaskArgs)
        
        timeTaken = (datetime.now()-timeStart).total_seconds()
        timesTaken.append(timeTaken)
    
    timeIndex += (count // 500) * 3
    
    #print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {timeCount} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (timeCount - timeIndex))})", end="", flush=True)
    
    if(generateExistingStegos):
        existingTaskArgs = [
            (
                dataset,
                os.path.join(folderPath, "Existing LSBM"),
                os.path.join(folderPath, "Existing LSBR"),
                os.path.join(folderPath, "Existing PPM"),
                secret,
                i,
                start,
                count
            ) for i in range(start + 1, start + count + 1)
        ]
        
        timeStart = datetime.now()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            exe.map(ProcessExistingStegos, existingTaskArgs)

        timeTaken = (datetime.now()- timeStart).total_seconds()
        timesTaken.append(timeTaken)
    timeIndex += (count // 500) * 3
    
    if(generateAdaptiveStegos):
        adaptiveTaskArgs = [
            (
                dataset,
                os.path.join(folderPath, "Adaptive HUGO"),
                os.path.join(folderPath, "Adaptive S-UNIWARD"),
                i,
                start,
                count
            ) for i in range(start + 1, start + count + 1)
        ]
        
        timeStart = datetime.now()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            exe.map(ProcessAdaptiveStegos, adaptiveTaskArgs)

        timeTaken = (datetime.now()- timeStart).total_seconds()
        timesTaken.append(timeTaken)
    timeIndex += (count // 500) * 2
    
    print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {timeCount} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (timeCount - timeIndex))})", end="", flush=True)

if __name__ == "__main__":
    generateExistingStegos = input("Generate Existing Stegos (Y/N) : ").upper() == "Y"
    generateCompositeStegos = input("Generate Composite Stegos (Y/N) : ").upper() == "Y"
    generateAdaptiveStegos = input("Generate Adaptive Stegos (Y/N) : ").upper() == "Y"
    
    with open(LIPSUM, "r") as f:
        secret = f.read()

    #1. Start with the training data
    ProcessSet(
        BOWS2,
        TRAIN_PATH,
        0,
        10000,
        generateExistingStegos=generateExistingStegos,
        generateCompositeStegos=generateCompositeStegos,
        generateAdaptiveStegos=generateAdaptiveStegos
    )
    
    print("\nTraining done")
      
    #2. TESTING data
    ProcessSet(
        BOSSBASE,
        TEST_PATH,
        2000,
        3500,
        generateExistingStegos=generateExistingStegos,
        generateCompositeStegos=generateCompositeStegos,
        generateAdaptiveStegos=generateAdaptiveStegos
    )
    
    print("\nTesting done")
    
    #3. Validation data
    ProcessSet(
        BOSSBASE,
        VAL_PATH,
        5500,
        3500,
        generateExistingStegos=generateExistingStegos,
        generateCompositeStegos=generateCompositeStegos,
        generateAdaptiveStegos=generateAdaptiveStegos
    )
    
    print("\nValidation Done")