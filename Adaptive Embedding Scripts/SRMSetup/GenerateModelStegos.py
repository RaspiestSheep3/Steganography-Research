import os
import sys

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

import shutil
from math import inf
from PIL import Image
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
from Helpers.CompositeImplementation import *
from concurrent.futures import ProcessPoolExecutor

"""
This code generates the images used to train the classifiers and SRNets.
It processes both Set A and Set C.
When calling this script from the command line, call it from within Adaptive Embedding Scripts as opposed to SRMSetup.
    - This is so GetPaths() works properly
"""

SETUP_PATHS = GetPaths()
SET_A = SETUP_PATHS["Bossbase Set A"]
SET_C = SETUP_PATHS["Bossbase Set C"]
LIPSUM = SETUP_PATHS["Lipsum"]
SRM = SETUP_PATHS["SRM"]

MAX_WORKERS = 10

setACoversPath = os.path.join(SRM, "Training Data", "Set A Covers")
setALSBMStegosPath = os.path.join(SRM, "Training Data", "Set A LSBM Stegos")
setALSBRStegosPath = os.path.join(SRM, "Training Data", "Set A LSBR Stegos")
setAPPMStegosPath = os.path.join(SRM, "Training Data", "Set A PPM Stegos")
setCCoversPath = os.path.join(SRM, "Training Data", "Set C Covers")
setC32Path = os.path.join(SRM, "Training Data", "Set C 32x32 Stegos")
setC64Path = os.path.join(SRM, "Training Data", "Set C 64x64 Stegos")
setC128Path = os.path.join(SRM, "Training Data", "Set C 128x128 Stegos")

#I'm not pushing these paths to github, so its good to make them in code
for path in [
    setACoversPath,
    setALSBMStegosPath,
    setALSBRStegosPath,
    setAPPMStegosPath,
    setCCoversPath,
    setC32Path,
    setC64Path,
    setC128Path
]:
    os.makedirs(path, exist_ok=True)

def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return f"{int(hour)}h {int(min)} min {round(sec)} sec"

def SaveImage(blocks, path):
    stego = Image.new("L", (256,256))
    
    counter = 0
    for blockRow in blocks:
        for block in blockRow:
            xOffset = (counter % 1) * (256 // 1)
            yOffset = (counter // 1) * (256 // 1)
            
            for i in range(256 // 1):
                for j in range(256 // 1):
                    stego.putpixel((xOffset + i, yOffset + j), block[j][i])
            
            counter += 1
    
    stego.save(path)

def ProcessExistingStegos(args):
    coverPathArg, setLSBMStegosPath, setLSBRStegosPath, setPPMStegosPath, secret, i = args
    
    coverPath = os.path.join(coverPathArg, f"{i}.png")
    block = SplitIntoBlocks(coverPath, 256)[0][0]
    
    if(i <= 600):
        bpp = 0.2
    elif(i <= 700):
        bpp = 0.4
    elif(i <= 800):
        bpp = 0.6
    elif(i <= 900):
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
    
    SaveImage([[lsbmBlock]], os.path.join(setLSBMStegosPath, f"{i-500}.png"))
    SaveImage([[lsbrBlock]], os.path.join(setLSBRStegosPath, f"{i-500}.png"))
    SaveImage([[ppmBlock]], os.path.join(setPPMStegosPath, f"{i-500}.png"))

def ProcessCompositeStegos(args):
    coverPathArg, size32Path, size64Path, size128Path, secret, i = args
    
    coverPath = os.path.join(coverPathArg, f"{i+2000}.png")
    
    if(i <= 600):
        bpp = 0.2
    elif(i <= 700):
        bpp = 0.4
    elif(i <= 800):
        bpp = 0.6
    elif(i <= 900):
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
    
    SaveImage([[blocks32]], os.path.join(size32Path, f"{i-500}.png"))
    SaveImage([[blocks64]], os.path.join(size64Path, f"{i-500}.png"))
    SaveImage([[blocks128]], os.path.join(size128Path, f"{i-500}.png"))

if __name__ == "__main__":
    with open(LIPSUM, "r") as f:
        secret = f.read()

    #First we must generate the covers from Set A and C
    for i in range(1, 501):
        #This is quite a fast operation so no need to multithread it
        shutil.copy(os.path.join(SET_A, f"{i}.png"), os.path.join(setACoversPath, f"{i}.png"))
        shutil.copy(os.path.join(SET_C, f"{i+2000}.png"), os.path.join(setCCoversPath, f"{i}.png"))

    print("Covers done, moving to main")

    #Next we generate the stegos from set A
    taskArgs = [
        (
            SET_A,
            setALSBMStegosPath,
            setALSBRStegosPath,
            setAPPMStegosPath,
            secret,
            i
        ) for i in range(501, 1001)
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(ProcessExistingStegos, taskArgs)

    print("Finished existing stegos, moving to composite")

    taskArgs = [
        (
            SET_C,
            setC32Path,
            setC64Path,
            setC128Path,
            secret,
            i
        ) for i in range(501, 1001)
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(ProcessCompositeStegos, taskArgs)