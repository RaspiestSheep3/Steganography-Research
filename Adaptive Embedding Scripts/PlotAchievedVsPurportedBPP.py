import numpy as np
from math import inf, floor
from datetime import datetime
import matplotlib.pyplot as plt
from Helpers.HelperFunctions import *
from concurrent.futures import ProcessPoolExecutor
from Helpers.CompositeImplementation import CompositeMethod

"""
This code plots the achieved BPP against purported BPP for the composite method at different thresholds.
This is done because the existence of index, failure and PPM blocks mean the purported BPP cannot always be achieved.
All data is saved as NPY files.
"""

SETUP_PATHS = GetPaths()
COVERS = SETUP_PATHS["Bossbase Set B"]
LIPSUM = SETUP_PATHS["Lipsum"]
SAVE_PATH = SETUP_PATHS["Achieved VS Purported BPP Save Folder"]
GRAPHS = SETUP_PATHS["Graphs"]

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, inf]
IMAGE_SIZE = 256
BLOCK_SIZES = [32, 64, 128]
MAX_WORKERS = 10
STEP = 5
COLOURS = ["red", "orange", "green", "blue", "purple", "pink", "lightblue", "brown"]

totalImageBytes = (IMAGE_SIZE ** 2) / 8

def ProcessImg(args):
    secret, path, bytesPerSection, blockSize, threshold = args
    _, counts, _ = CompositeMethod(
        secret, 
        path,
        bytesPerSection,
        IMAGE_SIZE // blockSize,
        threshold,
        reform=False #No point reforming, not doing it is faster
    )
    
    LSBRs = counts[1]
    LSBMs = counts[2]
    PPMs = counts[3]
    
    achievedBits = 0
    #We can treat LSBR and LSBM as embedding exactly how much we give it=
    achievedBits += (LSBRs + LSBMs) * bytesPerSection * 8 #Multiply by 8 for byte-bit conversion
    
    #PPM caps at 1/2 of the size of the block
    achievedBits += PPMs * min(bytesPerSection, int((blockSize ** 2) * 0.5)) * 8
    
    achievedBPP = achievedBits / (totalImageBytes * 8)
    
    return achievedBPP

def ConvertSeconds(seconds):
    minutes, secondsRemaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)} min {round(secondsRemaining)} sec"

if __name__ == "__main__":
    mode = input("Mode - (G)enerate or (P)lot : ").upper()
    
    if(mode == "G"):
        with open(LIPSUM, "r") as f:
            secretTotal = f.read()
            
        paths = os.listdir(COVERS)

        count = len(BLOCK_SIZES) * len(THRESHOLDS) * len(list(idx for idx in range(STEP, 100 + STEP, STEP)))
        index = 1
        timesTaken = []

        print(f"{datetime.now().strftime('%H:%M:%S')} - Starting Loop")

        for blockSize in BLOCK_SIZES:
            for threshold in THRESHOLDS:
                
                combinationResults = []
                
                for idx in range(STEP, 100 + STEP, STEP):
                    start = datetime.now()
                    if(idx > 100):
                        continue
                    
                    purportedBPP = idx / 100 
                    secretBytesAmount = floor(totalImageBytes * purportedBPP)
                    secret = secretTotal[:secretBytesAmount]
                    bytesPerSection = floor(secretBytesAmount / ((IMAGE_SIZE // blockSize) ** 2))
                    
                    taskArgs = [
                        (
                            secret,
                            os.path.join(COVERS, path),
                            bytesPerSection,
                            blockSize,
                            threshold
                        ) for path in paths
                    ]
                    
                    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
                        results = list(exe.map(ProcessImg, taskArgs))

                    combinationResults.append(np.float64(sum(results)) / len(results))

                    timesTaken.append((datetime.now() - start).total_seconds())
                    print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {index} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - index))})", end="", flush=True)
                    index += 1
                
                np.save(os.path.join(SAVE_PATH, f"{blockSize}-{threshold}.npy"), np.array(combinationResults, dtype=np.float64))
    
    elif(mode == "P"):
        files = os.listdir(SAVE_PATH)
        
        x = list(np.float64(idx)/100 for idx in range(0, 100 + STEP, STEP))
        
        for blockSize in BLOCK_SIZES:
            colourCount = 0
            plt.figure(figsize=(10, 5))
            
            for threshold in THRESHOLDS:
                result = list(np.load(os.path.join(SAVE_PATH, f"{blockSize}-{threshold}.npy")))
                result.insert(0, 0) #The graph would logically go through 0,0
                
                plt.plot(x, result, color=COLOURS[colourCount], lw=2, label=f"Threshold = {threshold if threshold != inf else "None"}")
                colourCount += 1
            
            #Plotting the baseline
            plt.plot([0.0, 1.0], [0.0, 1.0], color=COLOURS[colourCount], lw=2, linestyle='--', label= "Baseline")
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel("Purported BPP", fontsize=12)
            plt.ylabel("Average Achieved BPP", fontsize=12)
            
            plt.title(f"Achieved VS Purported BPP for Composite Method (Block Size = {blockSize}x{blockSize})", fontsize=14)
            plt.legend(loc="upper left", fontsize=12)
            plt.grid(alpha=0.3)
            
            os.makedirs(os.path.join(GRAPHS, "Achieved VS Purported BPPs"), exist_ok=True)
            plt.savefig(os.path.join(GRAPHS, "Achieved VS Purported BPPs", f"AchievedVSPurportedBPPs-BlockSize={blockSize}.png"), dpi=600)
            plt.close() 
            
        print("Generated all graphs")