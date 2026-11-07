import os
import sys

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

from math import inf
from PIL import Image
from datetime import datetime
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
from Helpers.CompositeImplementation import *
from concurrent.futures import ProcessPoolExecutor

"""
This code generates the images used to validate the SRNet models.
When calling this script from the command line, call it from within Adaptive Embedding Scripts as opposed to SRMSetup.
    - This is so GetPaths() works properly
This script is very similar to GenerateModelStegos.py, but I'm keeping it seperate because it targets different stuff, and I want to be able to run them seperately easily.
"""

SETUP_PATHS = GetPaths()
SET_D = SETUP_PATHS["Bossbase Set D"]
SET_E = SETUP_PATHS["Bossbase Set E"]
SET_F = SETUP_PATHS["BOWS2 Set F"]
LIPSUM = SETUP_PATHS["Lipsum"]
SRNET_TRAINING_DATA = SETUP_PATHS["SRNet Training Data"]
SRNET_TESTING_DATA = SETUP_PATHS["SRNet Testing Data"]
SRNET_VALIDATION_DATA = SETUP_PATHS["SRNet Validation Data"]

MAX_WORKERS = 10

def ConvertSeconds(seconds):
    minutes, secondsRemaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)} min {int(secondsRemaining)} sec"

def SaveImage(block, path):
    stego = Image.new("L", (256,256))
    
    for i in range(256):
        for j in range(256):
            stego.putpixel((j, i), block[i][j])
    
    stego.save(path)

def CarryOutLSBR(args):
    inPath, secret, outPath = args
    
    if(os.path.exists(outPath)): #This means if something happens it shouldnt overrwite itself
        return
    
    block = SplitIntoBlocks(inPath, 256)[0][0]
    newBlock = LSBReplacement(block, secret)
    SaveImage(newBlock, outPath)

def CarryOutLSBM(args):
    inPath, secret, outPath = args
    
    if(os.path.exists(outPath)):
        return
    
    block = SplitIntoBlocks(inPath, 256)[0][0]
    newBlock = LSBMatching(block, secret)
    SaveImage(newBlock, outPath)

def CarryOutPPM(args):
    inPath, secret, outPath = args
    
    if(os.path.exists(outPath)):
        return
    
    block = SplitIntoBlocks(inPath, 256)[0][0]
    newBlock = PixelPairMatching(block, secret)
    SaveImage(newBlock, outPath)

def CarryOutComposite(args):
    inPath, secret, outPath, bytesPerSection, blockSize = args
    
    if(os.path.exists(outPath)):
        return
    
    block, _, _ = CompositeMethod(
        secret, 
        inPath,
        bytesPerSection,
        IMAGE_SIZE // blockSize,
        inf
    )

    SaveImage(block, outPath)

if __name__ == "__main__":
    sizeOfEachStegoTrainingSet = len(os.listdir(SET_F)) // 2
    sizeOfEachStegoTestingSet = 1500 #=validation set, so not declaring a second variable
    
    totalImageBytes = (256 ** 2) // 8
    with open(LIPSUM, "r") as f:
        secretTotal = f.read()
        
    taskArgs : list[tuple[str, str, str]] = []
    
    timesTaken = []
    count = 48  #1 count = 1000 images 
    timeIndex = 0
    #Step 1 - the training set
    os.makedirs(os.path.join(SRNET_TRAINING_DATA, "LSBR Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TRAINING_DATA, "LSBM Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TRAINING_DATA, "PPM Stegos"), exist_ok=True)
    
    os.makedirs(os.path.join(SRNET_TRAINING_DATA, "Composite 32x32 Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TRAINING_DATA, "Composite 64x64 Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TRAINING_DATA, "Composite 128x128 Stegos"), exist_ok=True)
    
    
    os.makedirs(os.path.join(SRNET_TESTING_DATA, "LSBR Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TESTING_DATA, "LSBM Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TESTING_DATA, "PPM Stegos"), exist_ok=True)
    
    os.makedirs(os.path.join(SRNET_TESTING_DATA, "Composite 32x32 Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TESTING_DATA, "Composite 64x64 Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_TESTING_DATA, "Composite 128x128 Stegos"), exist_ok=True)
    
    
    os.makedirs(os.path.join(SRNET_VALIDATION_DATA, "LSBR Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_VALIDATION_DATA, "LSBM Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_VALIDATION_DATA, "PPM Stegos"), exist_ok=True)
    
    os.makedirs(os.path.join(SRNET_VALIDATION_DATA, "Composite 32x32 Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_VALIDATION_DATA, "Composite 64x64 Stegos"), exist_ok=True)
    os.makedirs(os.path.join(SRNET_VALIDATION_DATA, "Composite 128x128 Stegos"), exist_ok=True)
    
    #Same 20-20-20-20-20 BPP distribution
    
    print(f"{datetime.now().strftime('%H:%M:%S')} - Starting processing")
    
    #Step 1.1 - LSBR
    index = sizeOfEachStegoTrainingSet
    
    taskArgs1 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TRAINING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTrainingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
                    os.path.join(SET_F, f"{i}.png"),
                    secretTotal[:int(totalImageBytes * 0.4)],
                    os.path.join(SRNET_TRAINING_DATA, "LSBR Stegos", f"{i+1-index}.png")
                ) for i in range(index + sizeOfEachStegoTrainingSet // 5 + 1, index + (2*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TRAINING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTrainingSet) // 5 + 1, index + (3*sizeOfEachStegoTrainingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TRAINING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTrainingSet) // 5 + 1, index + (4*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TRAINING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTrainingSet) // 5 + 1, index + sizeOfEachStegoTrainingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start = datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutLSBR, taskArgs)
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 5
    if(timeTaken > 30): #Having a slight buffer because if the files exist i dont want to update times taken
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    #Step 1.2 - LSBM
    taskArgs1 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TRAINING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTrainingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
                    os.path.join(SET_F, f"{i}.png"),
                    secretTotal[:int(totalImageBytes * 0.4)],
                    os.path.join(SRNET_TRAINING_DATA, "LSBM Stegos", f"{i+1-index}.png")
                ) for i in range(index + sizeOfEachStegoTrainingSet // 5 + 1, index + (2*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TRAINING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTrainingSet) // 5 + 1, index + (3*sizeOfEachStegoTrainingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TRAINING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTrainingSet) // 5 + 1, index + (4*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TRAINING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTrainingSet) // 5 + 1, index + sizeOfEachStegoTrainingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start = datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutLSBM, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 5
    if(timeTaken > 30):
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    #Step 1.3 - PPM
    taskArgs1 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TRAINING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTrainingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
                    os.path.join(SET_F, f"{i}.png"),
                    secretTotal[:int(totalImageBytes * 0.4)],
                    os.path.join(SRNET_TRAINING_DATA, "PPM Stegos", f"{i+1-index}.png")
                ) for i in range(index + sizeOfEachStegoTrainingSet // 5 + 1, index + (2*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TRAINING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTrainingSet) // 5 + 1, index + (3*sizeOfEachStegoTrainingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TRAINING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTrainingSet) // 5 + 1, index + (4*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TRAINING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTrainingSet) // 5 + 1, index + sizeOfEachStegoTrainingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutPPM, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    print("\nExisting training set complete")

    #Step 1.4 - 32x32 Composite
    blockSize = 32
    
    taskArgs1 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTrainingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTrainingSet // 5 + 1, index + (2*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTrainingSet) // 5 + 1, index + (3*sizeOfEachStegoTrainingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTrainingSet) // 5 + 1, index + (4*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTrainingSet) // 5 + 1, index + sizeOfEachStegoTrainingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
        
    #Step 1.5 - 64x64 Composite
    blockSize = 64
    
    taskArgs1 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTrainingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTrainingSet // 5 + 1, index + (2*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTrainingSet) // 5 + 1, index + (3*sizeOfEachStegoTrainingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTrainingSet) // 5 + 1, index + (4*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTrainingSet) // 5 + 1, index + sizeOfEachStegoTrainingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
        
    #Step 1.6 - 128x128 Composite
    blockSize = 128
    
    taskArgs1 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTrainingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTrainingSet // 5 + 1, index + (2*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTrainingSet) // 5 + 1, index + (3*sizeOfEachStegoTrainingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTrainingSet) // 5 + 1, index + (4*sizeOfEachStegoTrainingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_F, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TRAINING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTrainingSet) // 5 + 1, index + sizeOfEachStegoTrainingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    print("\nComposite Training data complete")
    
    #Step 2 - Testing data
    index = 7501
    
    #Step 2.1 - LSBR
    taskArgs1 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TESTING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
                    os.path.join(SET_E, f"{i}.png"),
                    secretTotal[:int(totalImageBytes * 0.4)],
                    os.path.join(SRNET_TESTING_DATA, "LSBR Stegos", f"{i+1-index}.png")
                ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TESTING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TESTING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TESTING_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start = datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutLSBR, taskArgs)
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): #Having a slight buffer because if the files exist i dont want to update times taken
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    #Step 2.2 - LSBM
    taskArgs1 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TESTING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TESTING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TESTING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TESTING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TESTING_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start = datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutLSBM, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30):
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    #Step 2.3 - PPM
    taskArgs1 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TESTING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TESTING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TESTING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TESTING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TESTING_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutPPM, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    print("\nExisting Testing set complete")

    #Step 2.4 - 32x32 Composite
    blockSize = 32
    
    taskArgs1 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
        
    #Step 2.5 - 64x64 Composite
    blockSize = 64
    
    taskArgs1 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
        
    #Step 2.6 - 128x128 Composite
    blockSize = 128
    
    taskArgs1 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_E, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_TESTING_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    print("\nComposite Testing data complete")
    
    #Step 3 - Validation data
    index = 4501
    
    #Step 3.1 - LSBR
    taskArgs1 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBR Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start = datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutLSBR, taskArgs)
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): #Having a slight buffer because if the files exist i dont want to update times taken
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    #Step 3.2 - LSBM
    taskArgs1 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_VALIDATION_DATA, "LSBM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start = datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutLSBM, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30):
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    #Step 3.3 - PPM
    taskArgs1 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_VALIDATION_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_VALIDATION_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_VALIDATION_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_VALIDATION_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_VALIDATION_DATA, "PPM Stegos", f"{i+1-index}.png")
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5
    
    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutPPM, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    print("\nExisting Validation set complete")

    #Step 3.4 - 32x32 Composite
    blockSize = 32
    
    taskArgs1 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
        
    #Step 3.5 - 64x64 Composite
    blockSize = 64
    
    taskArgs1 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
        
    #Step 3.6 - 128x128 Composite
    blockSize = 128
    
    taskArgs1 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.2)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.2 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index, index + sizeOfEachStegoTestingSet // 5 + 1)
    ]
    taskArgs2 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.4)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.4 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + sizeOfEachStegoTestingSet // 5 + 1, index + (2*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs3 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.6)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.6 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (2*sizeOfEachStegoTestingSet) // 5 + 1, index + (3*sizeOfEachStegoTestingSet) // 5 + 1)
    ]
    taskArgs4 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 0.8)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 0.8 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (3*sizeOfEachStegoTestingSet) // 5 + 1, index + (4*sizeOfEachStegoTestingSet) // 5 + 1)
    ] 
    taskArgs5 = [
        (
            os.path.join(SET_D, f"{i}.png"),
            secretTotal[:int(totalImageBytes * 1.0)],
            os.path.join(SRNET_VALIDATION_DATA, f"Composite {blockSize}x{blockSize} Stegos", f"{i+1-index}.png"),
            int(totalImageBytes * 1.0 / ((IMAGE_SIZE // blockSize) ** 2)),
            blockSize
        ) for i in range(index + (4*sizeOfEachStegoTestingSet) // 5 + 1, index + sizeOfEachStegoTestingSet)
    ]
    
    taskArgs = taskArgs1 + taskArgs2 + taskArgs3 + taskArgs4 + taskArgs5

    start=datetime.now()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(CarryOutComposite, taskArgs)
    
    timeTaken = (datetime.now()-start).total_seconds()
    timeIndex += 1.5
    if(timeTaken > 30): 
        timesTaken.append(timeTaken)
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {timeIndex} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - timeIndex))})", end="", flush=True)
    else:
        print("Time taken too short - skipped")
    
    print("\nComposite Validation data complete")