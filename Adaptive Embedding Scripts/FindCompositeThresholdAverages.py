import numpy as np
from math import inf, floor
from Helpers.HelperFunctions import *
from concurrent.futures import ProcessPoolExecutor
from Helpers.CompositeImplementation import CompositeMethod

SETUP_PATHS = GetPaths()
SET_A = SETUP_PATHS["Bossbase Set A"]
LIPSUM = SETUP_PATHS["Lipsum"]

IMAGE_SIZE = 256
BLOCK_SIZES = [32, 64, 128]
MAX_WORKERS = 10
BPP = 1.0

def ProcessImg(args):
    path, secret, bytesPerSection, blocksPerSide = args
    
    _, _, t = CompositeMethod(
        secret,
        path,
        bytesPerSection,
        blocksPerSide,
        inf,
        reform=False
    )
    
    return t
   
if __name__ == "__main__":
    totalImageBytes = (IMAGE_SIZE ** 2) / 8
    with open(LIPSUM, "r") as f:
        secretTotal = f.read()
    
    for blockSize in BLOCK_SIZES:
        
        secretBytesAmount = floor(totalImageBytes * BPP)
        secret = secretTotal[:secretBytesAmount]
        bytesPerSection = floor(secretBytesAmount / ((IMAGE_SIZE // blockSize) ** 2))
        
        taskArgs = [
            (
                os.path.join(SET_A, f"{i}.png"),
                secret,
                bytesPerSection,
                IMAGE_SIZE // blockSize
            ) for i in range(1, 1001)
        ]
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            results = list(exe.map(ProcessImg, taskArgs))
            
        bigList = []
        for result in results:
            bigList += result
            
        results = np.array(bigList, dtype=np.float64)
        
        print(f"For Block Size = {blockSize} @ Purported 1.0 BPP:")
        print(f"    Mean    : {np.mean(results):.3f}")
        print(f"    Median  : {np.median(results):.3f}")
        print(f"    SD      : {np.std(results):.3f}")
        print("\n")