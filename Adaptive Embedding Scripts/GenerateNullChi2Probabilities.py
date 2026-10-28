import os
import numpy as np
import matplotlib.pyplot as plt
from Helpers.HelperFunctions import *
from concurrent.futures import ProcessPoolExecutor
from Helpers.SteganalysisMethods import ChiSquareAttack

"""
This code shows the average tail probabilites for the cover images across different block sizes for the Chi-Square Test.
This is used for calibration.
"""

SETUP_PATHS = GetPaths()
COVERS = SETUP_PATHS["Bossbase Set A"]
SAVE_PATH = SETUP_PATHS["Chi2 Null Probabilities"]
GRAPHS = SETUP_PATHS["Graphs"]

BLOCK_SIZES = [32, 64, 128]
MAX_WORKERS = 10
COLOURS = ["red", "green", "blue"]

def ProcessPath(path):
    out = {blockSize : [] for blockSize in BLOCK_SIZES}
    
    path = os.path.join(COVERS, path)
    for blockSize in BLOCK_SIZES:
            blocks = SplitIntoBlocks(path, blockSize)
            
            for row in blocks:
                for block in row:
                    out[blockSize].append(ChiSquareAttack(block))

    return out

if __name__ == "__main__":
    if(os.listdir(SAVE_PATH) in (None, [])):
        #Generating the data
    
        paths = os.listdir(COVERS)
        results = {blockSize : [] for blockSize in BLOCK_SIZES}

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            functionOut = list(exe.map(ProcessPath, paths))

        print("Completed collecting data")

        for output in functionOut:
            for blockSize in BLOCK_SIZES:
                results[blockSize] += output[blockSize]
        
        for blockSize in BLOCK_SIZES:
            result = np.array(results[blockSize])
            
            np.save(os.path.join(SAVE_PATH, f"{blockSize}.npy"), result)
            
        print("Saved")
    
    print("Generating graphs")
    for idx, blockSize in enumerate(BLOCK_SIZES):
        result = np.load(os.path.join(SAVE_PATH, f"{blockSize}.npy"))
        
        mean = np.mean(result)
        sd = np.std(result)
        median = np.median(result)
        
        print(f"Results for {blockSize}:")
        print(f"    Mean :      {mean:.3f}")
        print(f"    SD :        {sd:.3f}")
        print(f"    Median :    {median:.3f}")
        print("\n")
        
        plt.figure(figsize=(10, 5))
        plt.hist(result, bins=50, color=COLOURS[idx % len(COLOURS)], edgecolor="black", alpha=0.8)
        
        plt.title(f"Null Distribution Histogram for Cover Chi-Square Tail Probability For Blocksize {blockSize}x{blockSize}", fontsize=12)
        plt.xlabel("Chi-Squared Tail Probability", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.xlim([0.0, 1.0])
        plt.grid(alpha=0.3)
        
        os.makedirs(os.path.join(GRAPHS, "Null Chi2 Probabilities"), exist_ok=True)
        plt.savefig(os.path.join(GRAPHS, "Null Chi2 Probabilities", f"{blockSize}.png"), dpi=600)
    
    print("Graphs complete")