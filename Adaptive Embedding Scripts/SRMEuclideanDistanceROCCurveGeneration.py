import os
import numpy as np
from copy import deepcopy
from math import inf, ceil
from datetime import datetime
import matplotlib.pyplot as plt
from sealwatch.srm import srmq1
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import euclidean
from concurrent.futures import ProcessPoolExecutor
from Helpers.CompositeImplementation import CompositeMethod

"""
This code is used to generate an ROC curve analysing the efficacy of the composite method compared to pure methods.
In order to do this, for each cover image in Set B, for a given bits per pixel, a stego is generated using the existing methods and the composite method.
Their features are then extracted using SRM, and the euclidean vector distance between their SRMs and the mean SRM for the cover images (see GenerateAverageCoverSRM.py) are extracted.
The actual classification (0 for cover, 1 for stego) is used for the true labels, and the euclidean distance for the score labels
A higher AUC means this steganalysis method is more effective against that embedding method  
"""

SETUP_PATHS = GetPaths()
BOSSBASE_FOLDER = SETUP_PATHS["Bossbase Set B"]
LIPSUM = SETUP_PATHS["Lipsum"]
MEAN_COVER_NPY = SETUP_PATHS["Mean Cover SRM NPY"]
GRAPHS = SETUP_PATHS["Graphs"]

STEP = 5
IMAGE_SIZE = (256, 256)
THRESHOLD = inf
BLOCK_SIZES = [32, 64, 128]
COLOURS = ["purple", "orange", "pink"]
MAX_CPU_WORKERS = 10
BPPs = [0.1, 0.5, 1.0]

paths = os.listdir(BOSSBASE_FOLDER)

def FlattenSRM(srmDict):
    vectors = []
    for key in sorted(srmDict.keys()):
        vectors.append(np.array(srmDict[key]).flatten())
    return np.concatenate(vectors)

def ConvertSeconds(seconds):
    minutes, secondsRemaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)} min {round(secondsRemaining)} sec"

def ProcessSet(args):
    path, embedData, secretBytesAmount, secretBytesAmountComp, blockSizes, threshold, bossbaseFolder, meanVector = args
    
    coverBlock = SplitIntoBlocks(os.path.join(bossbaseFolder, path), 256)[0][0]
            
    lsbrBlock = LSBReplacement(deepcopy(coverBlock), embedData[:secretBytesAmount])
    lsbmBlock = LSBMatching(deepcopy(coverBlock), embedData[:secretBytesAmount])
    ppmBlock = PixelPairMatching(deepcopy(coverBlock), embedData[:secretBytesAmount])
    
    coverSRM = srmq1.extract(np.array(coverBlock, dtype=np.uint8))
    lsbrSRM = srmq1.extract(np.array(lsbrBlock, dtype=np.uint8))
    lsbmSRM = srmq1.extract(np.array(lsbmBlock, dtype=np.uint8))
    ppmSRM = srmq1.extract(np.array(ppmBlock, dtype=np.uint8))
    
    coverVector = FlattenSRM(coverSRM)
    lsbrVector = FlattenSRM(lsbrSRM)
    lsbmVector = FlattenSRM(lsbmSRM)
    ppmVector = FlattenSRM(ppmSRM)
    
    compDists = {}
    for blockSize in blockSizes:
        blocksPerSideArg = int(IMAGE_SIZE[0] / blockSize)
        bytesPerSectionComp = ceil(secretBytesAmountComp / (blocksPerSideArg ** 2))
        
        compBlock, _, _ = CompositeMethod(
            embedData[:secretBytesAmountComp], 
            os.path.join(bossbaseFolder, path), 
            acceptableMappingThreshold=threshold, 
            bytesPerSection=bytesPerSectionComp, 
            blocksPerSide=blocksPerSideArg,
        )
        compSRM = srmq1.extract(np.array(compBlock, dtype=np.uint8))
        compVector = FlattenSRM(compSRM)
        compDists[blockSize] = euclidean(meanVector, compVector)

    return {
        'coverDist': euclidean(meanVector, coverVector),
        'lsbrDist': euclidean(meanVector, lsbrVector),
        'lsbmDist': euclidean(meanVector, lsbmVector),
        'ppmDist': euclidean(meanVector, ppmVector),
        'compDists': compDists
    }

if __name__ == "__main__":
    print("Imports complete")
    
    meanVector = np.load(MEAN_COVER_NPY)
    
    embedData = []
    with open(LIPSUM, "r") as f:
        embedData = [*f.read()]
    
    index = 1
    count = len(BPPs)
    timesTaken = []

    print("Starting loop")
    for bpp in BPPs:
        rocLSBRTrue = []
        rocLSBRScores = []
        rocLSBMTrue = []
        rocLSBMScores = []
        rocPPMTrue = []
        rocPPMScores =[]
        
        rocCompTrueDict = {blockSize: [] for blockSize in BLOCK_SIZES}
        rocCompScoresDict = {blockSize: [] for blockSize in BLOCK_SIZES}

        # Finding amount of bytes to embed
        totalImageBytes = (IMAGE_SIZE[0] * IMAGE_SIZE[1] * bpp) / 8
        secretBytesAmount = int(totalImageBytes)
        secretBytesAmountComp = int(totalImageBytes)
        
        taskArgs = [
            (path, embedData, secretBytesAmount, secretBytesAmountComp, BLOCK_SIZES, THRESHOLD, BOSSBASE_FOLDER, meanVector)
            for path in paths
        ]
        
        start = datetime.now()
        
        with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as exe: 
            resultsBatch = list(exe.map(ProcessSet, taskArgs))
            
        for res in resultsBatch:
            # Add cover scores (Label 0)
            rocLSBRTrue.append(0); rocLSBRScores.append(res['coverDist'])
            rocLSBMTrue.append(0); rocLSBMScores.append(res['coverDist'])
            rocPPMTrue.append(0); rocPPMScores.append(res['coverDist'])
            
            for blockSize in BLOCK_SIZES:
                rocCompTrueDict[blockSize].append(0)
                rocCompScoresDict[blockSize].append(res['coverDist'])
                
            # Add stego scores (Label 1)
            rocLSBRTrue.append(1); rocLSBRScores.append(res['lsbrDist'])
            rocLSBMTrue.append(1); rocLSBMScores.append(res['lsbmDist'])
            rocPPMTrue.append(1); rocPPMScores.append(res['ppmDist'])
            
            for blockSize in BLOCK_SIZES:
                rocCompTrueDict[blockSize].append(1)
                rocCompScoresDict[blockSize].append(res['compDists'][blockSize])
        
        timesTaken.append((datetime.now() - start).total_seconds())
        print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {index} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - index))})", end="", flush=True)
        index += 1

        fprLSBR, tprLSBR, _ = roc_curve(rocLSBRTrue, rocLSBRScores)
        rocAucLSBR = auc(fprLSBR, tprLSBR)
        fprLSBM, tprLSBM, _ = roc_curve(rocLSBMTrue, rocLSBMScores)
        rocAucLSBM = auc(fprLSBM, tprLSBM)
        fprPPM, tprPPM, _ = roc_curve(rocPPMTrue, rocPPMScores)
        rocAucPPM = auc(fprPPM, tprPPM)

        plt.figure(figsize=(10, 5))
        plt.plot(fprLSBR, tprLSBR, color='red', lw=2, label=f'LSBR (AUC = {rocAucLSBR:.3f})')
        plt.plot(fprLSBM, tprLSBM, color='green', lw=2, label=f'LSBM (AUC = {rocAucLSBM:.3f})')
        plt.plot(fprPPM, tprPPM, color='blue', lw=2, label=f'PPM (AUC = {rocAucPPM:.3f})')

        for idx, blockSize in enumerate(BLOCK_SIZES):
            colour = COLOURS[idx % len(COLOURS)]
            fprComp, tprComp, _ = roc_curve(rocCompTrueDict[blockSize], rocCompScoresDict[blockSize])
            rocAucComp = auc(fprComp, tprComp)
            plt.plot(fprComp, tprComp, color=colour, lw=2, label=f'Composite ({blockSize}x{blockSize}, AUC = {rocAucComp:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.05, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'ROC Curves for SRM Euclidean Distance ({bpp} BPP)', fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(GRAPHS, f'SRMEuclideanDistanceROCCurve-{bpp}bpp.png'), dpi=600)
        plt.close() 