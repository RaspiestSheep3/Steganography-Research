import os
import numpy as np
from scipy import stats
from math import ceil, inf
from datetime import datetime
import matplotlib.pyplot as plt
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
from Helpers.SteganalysisMethods import *
from sklearn.metrics import roc_curve, auc
from Helpers.TestingScriptFuncVersion import CompositeMethod

BOSSBASE_FOLDER = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Bossbase Dataset 1-10"
LIPSUM = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Idea\Lipsum.txt"
STEP = 5
IMAGE_SIZE = (256, 256)
BLOCK_SIZE = 64
EXPERIMENTAL_BLOCK_SIZE = 64
THRESHOLD = inf
EDGE_MARGIN = 1
KL_NUM_BINS = 100
KL_EPSILON = 1e-10
CARRY_OUT_STATISTICAL_TESTS = False

paths = os.listdir(BOSSBASE_FOLDER)
blocksPerSideArg = int(IMAGE_SIZE[0] / BLOCK_SIZE)

globalTrueLabels = []
globalScores = []

def ApplyKernel(kernel: list[list[int]], base: list[list[int]]) -> float:
    sumValue = 0
    for i in range(len(kernel)):
        for j in range(len(kernel[i])):
            sumValue += kernel[i][j] * base[i][j]
    return sumValue

def CalculateVariance(base: list[list[float]]) -> float:
    totalCount = len(base) * len(base[0])
    meanValue = sum(sum(row) for row in base) / totalCount
    
    varianceSum = 0
    for row in base:
        for item in row:
            varianceSum += (item - meanValue) ** 2
            
    return varianceSum / totalCount

coverBlocks = []
for path in paths:
    block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
    coverBlocks.append(block)
    
embedData = []
with open(LIPSUM, "r") as f:
    embedData = [*f.read()]

for stepPercent in range(STEP, 101, STEP):
    if stepPercent > 100:
        break
    
    # Finding amount of bytes to embed
    totalImageBytes = (IMAGE_SIZE[0] * IMAGE_SIZE[1]) / 8
    secretBytesAmount = int(totalImageBytes * (stepPercent / 100))

    # Finding the bytes per section to attempt
    bytesPerSection = ceil(secretBytesAmount / ((IMAGE_SIZE[0] / BLOCK_SIZE) ** 2))
    index = 0
    
    klScores = []
    pValues = []
    
    for path in paths:
        stegoBlock, _, _ = CompositeMethod(
            embedData[:secretBytesAmount], 
            os.path.join(BOSSBASE_FOLDER, path), 
            acceptableMappingThreshold=THRESHOLD, 
            bytesPerSection=bytesPerSection, 
            blocksPerSide=blocksPerSideArg,
        )

        height = len(stegoBlock)
        width = len(stegoBlock[0])

        # 3x3 Laplacian 8 neighbour high pass filter
        filterKernel = [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]
        
        residualMatrix = [[0.0 for _ in range(width)] for _ in range(height)]
        
        for i in range(height):
            for j in range(width):
                passBase = [[0 for _ in range(3)] for _ in range(3)]
                
                for iOffset in range(-1, 2):
                    for jOffset in range(-1, 2):
                        targetI = min(max(i + iOffset, 0), height - 1)
                        targetJ = min(max(j + jOffset, 0), width - 1)
                        passBase[iOffset + 1][jOffset + 1] = stegoBlock[targetI][targetJ]
                
                residualMatrix[i][j] = ApplyKernel(filterKernel, passBase)
        
        # Variance calculation
        varianceMatrix = [[0.0 for _ in range(width)] for _ in range(height)]
                
        for i in range(height):
            for j in range(width):
                passBase = [[0.0 for _ in range(5)] for _ in range(5)]
                
                for iOffset in range(-2, 3):
                    for jOffset in range(-2, 3):
                        targetI = min(max(i + iOffset, 0), height - 1)
                        targetJ = min(max(j + jOffset, 0), width - 1)

                        passBase[iOffset + 2][jOffset + 2] = residualMatrix[targetI][targetJ]
                
                varianceMatrix[i][j] = CalculateVariance(passBase)

        #ROC setup
        for row in range(height):
            for col in range(width):
                isOnRowBoundary = (row % EXPERIMENTAL_BLOCK_SIZE < EDGE_MARGIN) or (row % EXPERIMENTAL_BLOCK_SIZE >= EXPERIMENTAL_BLOCK_SIZE - EDGE_MARGIN)
                isOnColBoundary = (col % EXPERIMENTAL_BLOCK_SIZE < EDGE_MARGIN) or (col % EXPERIMENTAL_BLOCK_SIZE >= EXPERIMENTAL_BLOCK_SIZE - EDGE_MARGIN)
                
                # Ground truth label: 1 if boundary, 0 if interior
                isBoundary = 1 if (isOnRowBoundary or isOnColBoundary) else 0
                
                globalTrueLabels.append(isBoundary)
                globalScores.append(varianceMatrix[row][col])

        if(CARRY_OUT_STATISTICAL_TESTS):
            #Sorting into boundary + interior
            vBoundary = []
            vInterior = []
            
            for row in range(height):
                for col in range(width):
                    isOnRowBoundary = (row % EXPERIMENTAL_BLOCK_SIZE < EDGE_MARGIN) or (row % EXPERIMENTAL_BLOCK_SIZE >= EXPERIMENTAL_BLOCK_SIZE - EDGE_MARGIN)
                    isOnColBoundary = (col % EXPERIMENTAL_BLOCK_SIZE < EDGE_MARGIN) or (col % EXPERIMENTAL_BLOCK_SIZE >= EXPERIMENTAL_BLOCK_SIZE - EDGE_MARGIN)
                    
                    if isOnRowBoundary or isOnColBoundary:
                        vBoundary.append(varianceMatrix[row][col])
                    else:
                        vInterior.append(varianceMatrix[row][col])

            vBoundary = np.array(vBoundary)
            vInterior = np.array(vInterior)
        
            #Statistical testing
            #Welchs t Test
            tStat, tPValue = stats.ttest_ind(vBoundary, vInterior, equal_var=False)
            uStat, uPValue = stats.mannwhitneyu(vBoundary, vInterior, alternative='two-sided')
            
            #KL divergence
            minVal = min(np.min(vBoundary), np.min(vInterior))
            maxVal = max(np.max(vBoundary), np.max(vInterior))
            bins = np.linspace(minVal, maxVal, KL_NUM_BINS + 1)
            
            pBoundary, _ = np.histogram(vBoundary, bins=bins, density=True)
            pInterior, _ = np.histogram(vInterior, bins=bins, density=True)
            
            pBoundary = (pBoundary + KL_EPSILON) / np.sum(pBoundary + KL_EPSILON)
            pInterior = (pInterior + KL_EPSILON) / np.sum(pInterior + KL_EPSILON)
            
            klDivergence = np.sum(pBoundary * np.log2(pBoundary / pInterior))

            pValues.append(uPValue)
            klScores.append(klDivergence)

        index += 1

    print(f"{datetime.now().strftime('%H:%M:%S')} - {stepPercent}% Complete")
    
    if(CARRY_OUT_STATISTICAL_TESTS):
        print(f"AVG KL : {np.mean(klScores)} | AVG P VALUE : {np.mean(pValues)}")
    
print("Generating ROC Curve")
fpr, tpr, thresholds = roc_curve(globalTrueLabels, globalScores)
rocAuc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Boundary Detector (AUC = {rocAuc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.05, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title(f'ROC Curve for Block Boundary Steganalysis Detector (Block Size = {BLOCK_SIZE}, Test Size = {EXPERIMENTAL_BLOCK_SIZE})', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

plt.savefig(f'BoundarySteganalysis_ROC_{BLOCK_SIZE}-Testing={EXPERIMENTAL_BLOCK_SIZE}.png', dpi=600)

print(f"Final AUC Score: {rocAuc:.4f}")