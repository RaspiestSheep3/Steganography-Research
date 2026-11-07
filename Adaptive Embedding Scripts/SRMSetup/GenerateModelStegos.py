import os
import sys

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

import shutil
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
LIPSUM = SETUP_PATHS["Lipsum"]
MODEL_TRAINING_DATA = SETUP_PATHS["Model Data"]
BOWS2 = SETUP_PATHS["BOWS2"]
BOSSBASE = SETUP_PATHS["Bossbase"]
TRAIN_PATH = os.path.join(MODEL_TRAINING_DATA, "Training")
TEST_PATH = os.path.join(MODEL_TRAINING_DATA, "Testing")
VAL_PATH = os.path.join(MODEL_TRAINING_DATA, "Validation")

MAX_WORKERS = 10

timesTaken = []
timeCount = 34 * 6 #1 count = 500 paths
timeIndex = 0


#I'm not pushing these paths to github, so its good to make them in code

def ConvertSeconds(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return f"{int(hour)}h {int(min)} min {int(sec)} sec"

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
    
    SaveImage(lsbmBlock, os.path.join(setLSBMStegosPath, f"{i}.png"))
    SaveImage(lsbrBlock, os.path.join(setLSBRStegosPath, f"{i}.png"))
    SaveImage(ppmBlock, os.path.join(setPPMStegosPath, f"{i}.png"))

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
    
    SaveImage(blocks32, os.path.join(size32Path, f"{i}.png"))
    SaveImage(blocks64, os.path.join(size64Path, f"{i}.png"))
    SaveImage(blocks128, os.path.join(size128Path, f"{i}.png"))

def ProcessSet(dataset, folderPath, start, count):
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
    ]:
        os.makedirs(path, exist_ok=True)
    
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
    
    print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {timeCount} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (timeCount - timeIndex))})", end="", flush=True)
    
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
    
    print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {timeCount} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (timeCount - timeIndex))})", end="", flush=True)

if __name__ == "__main__":
    with open(LIPSUM, "r") as f:
        secret = f.read()

    #1. Start with the training data
    """ProcessSet(
        BOWS2,
        TRAIN_PATH,
        0,
        10000
    )
    
    print("\nTraining done")
    
    """
      
    #2. TESTING data
    ProcessSet(
        BOSSBASE,
        TEST_PATH,
        2000,
        3500
    )
    
    print("\nTesting done")
    
    #3. Validation data
    ProcessSet(
        BOSSBASE,
        VAL_PATH,
        5500,
        3500
    )
    
    print("\nValidation Done")