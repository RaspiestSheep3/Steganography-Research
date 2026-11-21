import pickle
import sqlite3
import numpy as np
from math import inf
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from Helpers.HelperFunctions import *
from sklearn.metrics import roc_auc_score
from Helpers.EmbeddingAlgorithms import *
from SRNet.UseSRNetModel import SteganalysisModel
from concurrent.futures import ProcessPoolExecutor
from Helpers.CompositeImplementation import CompositeMethod
from SRMSetup.UseSRMModel import PredictNewImages, ExtractSingleImageFeatures

"""
- This code compares the composite method against the existing methods by comparing AUCs.
- It then generates a heatmap with the comparison.

- Is a +ve ΔAUC good or bad and why?
    - A +ve ΔAUC is good
    - ΔAUC is found by [existing mean AUC] - [composite AUC]
    - If [composite AUC] > [existing mean AUC] => ΔAUC < 0
    - In this case, the composite method is easier to detect => this is bad
"""

SETUP_PATHS = GetPaths()
NPY_FILES = SETUP_PATHS["NPY"]
BOSSBASE_B = SETUP_PATHS["Bossbase Set B"]
DB = SETUP_PATHS["Database"]
LIPSUM = SETUP_PATHS["Structured Secret"]
SRM = SETUP_PATHS["SRM"]
GRAPHS = SETUP_PATHS["Graphs"]

THRESHOLDS = {
    32  : (0.3, 0.4, 0.6, inf),
    64  : (0.4, 0.5, 0.6, inf),
    128 : (0.2, 0.3, 0.4, inf)
}
PURPORTED_BPPS = [
    0.1, 
    0.5, 
    1.0
]
NP_PURPORTED_DATA_STEP_SIZE = 2
NULL_ACHIEVED_BPP_MAX = 0.01
MAX_WORKERS = 10
CMAP_THEME = "icefire"
HEATMAP_FIGSIZE = (18, 10)
PARETO_FIGSIZE = (24, 18)
COLOUR_BAR_RELATIVE_SIZE = 0.03
PARETO_COLOURS = [
    "red",
    "blue",
    "green",
    "purple"
]
PARETO_MARKERS = [
    "o",
    "x",
    "+",
    "v"
]

completedCount = []

def LoadModels():
    srmModels = dict()
    srnetModels = dict()
    
    srmModels["LSBR"]   = OpenSRMModel(os.path.join(SRM, "Classifiers-E Models", "TrainedSRMEnsemble-Classifiers-E-LSBR.pkl"))
    srmModels["LSBM"]   = OpenSRMModel(os.path.join(SRM, "Classifiers-E Models", "TrainedSRMEnsemble-Classifiers-E-LSBM.pkl"))
    srmModels["PPM"]    = OpenSRMModel(os.path.join(SRM, "Classifiers-E Models", "TrainedSRMEnsemble-Classifiers-E-PPM.pkl"))
    srmModels[32]       = OpenSRMModel(os.path.join(SRM, "Classifiers-C Models", "TrainedSRMEnsemble-Classifiers-C-32.pkl"))
    srmModels[64]       = OpenSRMModel(os.path.join(SRM, "Classifiers-C Models", "TrainedSRMEnsemble-Classifiers-C-64.pkl"))
    srmModels[128]      = OpenSRMModel(os.path.join(SRM, "Classifiers-C Models", "TrainedSRMEnsemble-Classifiers-C-128.pkl"))
    
    srnetModels["LSBR"] = SteganalysisModel(os.path.join(SRM, "SRNets-E Models", "LSBR Model.pt"))
    srnetModels["LSBM"] = SteganalysisModel(os.path.join(SRM, "SRNets-E Models", "LSBM Model.pt"))
    srnetModels["PPM"]  = SteganalysisModel(os.path.join(SRM, "SRNets-E Models", "PPM Model.pt"))
    srnetModels[32]     = SteganalysisModel(os.path.join(SRM, "SRNets-C Models", "32x32 Model.pt"))
    srnetModels[64]     = SteganalysisModel(os.path.join(SRM, "SRNets-C Models", "64x64 Model.pt"))
    srnetModels[128]    = SteganalysisModel(os.path.join(SRM, "SRNets-C Models", "128x128 Model.pt"))

    return srmModels, srnetModels

def CalculateAUCConfidenceInterval(trueData, scoreData, boostrapNo = 1000, interval = 95):
    trueDataNP = np.array(trueData)
    scoreDataNP = np.array(scoreData)
    
    bootstrappedScores = []
    rng = np.random.default_rng()
    
    for _ in range(boostrapNo):
        indices = rng.integers(0, len(trueData), len(trueData))
        
        if len(np.unique(trueDataNP[indices])) < 2:
            continue
    
        score = roc_auc_score(
            trueDataNP[indices], scoreDataNP[indices]
        )
        bootstrappedScores.append(score)
    
    lowerPercentile = (100-interval) / 2
    upperPercentile = 100 - lowerPercentile
    
    ciLower = np.percentile(bootstrappedScores, lowerPercentile)
    ciUpper = np.percentile(bootstrappedScores, upperPercentile)
    meanAUC = np.mean(bootstrappedScores)
    
    return meanAUC, ciLower, ciUpper

def CoverSRM(coverFile):
    imgObject = Image.open(coverFile)
    npArray = np.asarray(imgObject)
    
    return ExtractSingleImageFeatures(npArray)

def OpenSRMModel(modelPath):
    with open(modelPath, "rb") as f:
        return pickle.load(f)

def ProcessExistingMethod(achievedBPP, method, srmModels, srnetModels, coverSRMs, existingFN : function):
    with open(LIPSUM, "r") as f:
        secret = f.read(int(achievedBPP * 256 * 256 / 8))
    
    srmROCTrue = []
    srmROCScores= []
    srnetROCTrue = []
    srnetROCScores = []
    
    for index, basePath in enumerate(os.listdir(BOSSBASE_B)):
        path = os.path.join(BOSSBASE_B, basePath)
        
        lsbrBlock = existingFN(
            SplitIntoBlocks(path, 256)[0][0],
            secret,
        )
        
        srmModel = srmModels[method]
        srmPrediction = PredictNewImages(
            srmModel,
            [lsbrBlock],
        )[0]
        
        imgObject = Image.fromarray(
            np.asarray(lsbrBlock, dtype=np.uint8)
        )
        
        srnetModel = srnetModels[method]
        srnetPredictionDict = srnetModel.PredictImage(
            imgObject=imgObject
        )
        srnetPrediction = srnetPredictionDict["confidence"] * (-1 if srnetPredictionDict["prediction"] == "Cover" else 1)
        
        #ROC
        srmROCScores.append(srmPrediction)
        srmROCTrue.append(1)
        
        srnetROCScores.append(srnetPrediction)
        srnetROCTrue.append(1)
        
        #Doing the cover files
        srmModel = srmModels[method]
        srmPrediction = PredictNewImages(
            srmModel,
            extractedFeatureList=[coverSRMs[index]]
        )[0]
        
        imgObject = Image.fromarray(
            np.asarray(SplitIntoBlocks(path, 256)[0][0], dtype=np.uint8)
        )
        
        srnetModel = srnetModels[method]
        srnetPredictionDict = srnetModel.PredictImage(
            imgObject=imgObject
        )
        
        srnetPrediction = srnetPredictionDict["confidence"] * (-1 if srnetPredictionDict["prediction"] == "Cover" else 1)
    
        srmROCScores.append(srmPrediction)
        srmROCTrue.append(0)
                
        srnetROCScores.append(srnetPrediction)
        srnetROCTrue.append(0)
    
    srmAUC, srmCILower, srmCIUpper = CalculateAUCConfidenceInterval(srmROCTrue, srmROCScores)
    srnetAUC, srnetCILower, srnetCIUpper = CalculateAUCConfidenceInterval(srnetROCTrue, srnetROCScores)
    return srmAUC, srmCILower, srmCIUpper, srnetAUC, srnetCILower, srnetCIUpper

def Iteration(args):
    blockSize, threshold, purportedBPP, coverSRMs = args
    
    srmModels, srnetModels = LoadModels()
    
    conn = sqlite3.connect(DB, timeout=30.0)
    cursor = conn.cursor()
    
    #Calculating the combination index
    #Hundreds unit = block size (1-3)
    #Tens unit = threshold (1-4)
    #Ones unit = purported bpp (1-3)
    
    combinationIndex = 100 * (list(THRESHOLDS.keys()).index(blockSize) + 1)
    combinationIndex += 10 * (list(THRESHOLDS[blockSize]).index(threshold) + 1)
    combinationIndex += PURPORTED_BPPS.index(purportedBPP) + 1
    
    #print(f"Test flag - {combinationIndex}")
    
    #Finding the achieved BPP for the composite method
    npData = np.load(os.path.join(
        NPY_FILES, 
        "Achieved BPP VS Purported BPP", 
        f"{blockSize}-{threshold}-BPP.npy"   
    ))
    
    achievedBPP = npData[int(purportedBPP * 100) // NP_PURPORTED_DATA_STEP_SIZE - 1]

    cursor.execute("""
        SELECT * FROM CompositeAUCs WHERE combinationIndex = ?
    """,
    (combinationIndex, ))
    
    rows = cursor.fetchall()
    
    if(not (rows == None or rows == [])):
        #Entry already exists - continuing
        #completedCount.append(0)
        #print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - {len(completedCount)} complete (Latest ended early due to already existing)", end = "", flush = True)
        conn.close()
        
        return

    if(achievedBPP <= NULL_ACHIEVED_BPP_MAX):
        cursor.execute("""
            INSERT INTO CompositeAUCs(
                combinationIndex,
                blockSize,
                purportedBPP,
                threshold,
                srmAUC,
                srmCILower,
                srmCIUpper,
                srnetAUC,
                srnetCILower,
                srnetCIUpper,
                ignoreEntry
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)  
        """,
        (
            combinationIndex,
            blockSize,
            purportedBPP,
            threshold,
            0,
            0,
            0,
            0,
            0,
            0,
            1
        ))
        
        methods = ["LSBR", "LSBM", "PPM"]
        for method in methods:
            cursor.execute("""
                INSERT INTO ExistingAUCs(
                    combinationIndex,
                    method,
                    srmAUC,
                    srmCILower,
                    srmCIUpper,
                    srnetAUC,
                    srnetCILower,
                    srnetCIUpper,
                    ignoreEntry
                )
                VALUES (?, ?, ?, ?, ?, ?, ? ,? ,?)  
                """,
                (
                    combinationIndex + 0.1 * (methods.index(method) + 1),
                    method,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1
                )
            )
        
        conn.commit()
    
        #completedCount.append(0)
        #print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - {len(completedCount)} complete (Latest ended early due to low BPP)", end = "", flush = True)
        conn.close()
        
        return

    #Generating and processing composite stegos
    with open(LIPSUM, "r") as f:
        secret = f.read(int(purportedBPP * 256 * 256 / 8))
    
    bytesPerSection = int(int(purportedBPP * 256 * 256 / 8) / ((256 // blockSize) ** 2))
    
    srmROCTrue = []
    srmROCScores= []
    srnetROCTrue = []
    srnetROCScores = []

    for index, basePath in enumerate(sorted(os.listdir(BOSSBASE_B))):
        path = os.path.join(BOSSBASE_B, basePath)
                
        compositeImageBlocks, _, _ = CompositeMethod(
            secret,
            path,
            bytesPerSection,
            256 // blockSize,
            threshold
        )
        
        srmModel = srmModels[blockSize]
        srmPrediction = PredictNewImages(
            srmModel,
            [compositeImageBlocks],
        )[0]
        
        imgObject = Image.fromarray(
            np.asarray(compositeImageBlocks, dtype=np.uint8)
        )
        
        srnetModel = srnetModels[blockSize]
        srnetPredictionDict = srnetModel.PredictImage(
            imgObject=imgObject
        )
        
        srnetPrediction = srnetPredictionDict["confidence"] * (-1 if srnetPredictionDict["prediction"] == "Cover" else 1)
        
        #ROC
        srmROCScores.append(srmPrediction)
        srmROCTrue.append(1)
        
        srnetROCScores.append(srnetPrediction)
        srnetROCTrue.append(1)
        
        #Doing the cover files
        srmModel = srmModels[blockSize]
        srmPrediction = PredictNewImages(
            srmModel,
            extractedFeatureList=[coverSRMs[index]]
        )[0]
        
        imgObject = Image.fromarray(
            np.asarray(SplitIntoBlocks(path, 256)[0][0], dtype=np.uint8)
        )
        
        srnetModel = srnetModels[blockSize]
        srnetPredictionDict = srnetModel.PredictImage(
            imgObject=imgObject
        )
        
        srnetPrediction = srnetPredictionDict["confidence"] * (-1 if srnetPredictionDict["prediction"] == "Cover" else 1)
    
        srmROCScores.append(srmPrediction)
        srmROCTrue.append(0)
                
        srnetROCScores.append(srnetPrediction)
        srnetROCTrue.append(0)
    
    srmAUC, srmCILower, srmCIUpper = CalculateAUCConfidenceInterval(srmROCTrue, srmROCScores)
    srnetAUC, srnetCILower, srnetCIUpper = CalculateAUCConfidenceInterval(srnetROCTrue, srnetROCScores)
        
    cursor.execute("""
        INSERT INTO CompositeAUCs(
            combinationIndex,
            blockSize,
            purportedBPP,
            threshold,
            srmAUC,
            srmCILower,
            srmCIUpper,
            srnetAUC,
            srnetCILower,
            srnetCIUpper,
            ignoreEntry
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)  
    """,
    (
        combinationIndex,
        blockSize,
        purportedBPP,
        threshold,
        srmAUC,
        srmCILower,
        srmCIUpper,
        srnetAUC,
        srnetCILower,
        srnetCIUpper,
        0
    ))
    
    conn.commit()

    #Generating and processing existing stegos

    #LSBR
    srmAUC, srmCILower, srmCIUpper, srnetAUC, srnetCILower, srnetCIUpper = ProcessExistingMethod(
        achievedBPP,
        "LSBR",
        srmModels, 
        srnetModels,
        coverSRMs,
        LSBReplacement
    )
    
    cursor.execute("""
        INSERT INTO ExistingAUCs(
            combinationIndex,
            method,
            srmAUC,
            srmCILower,
            srmCIUpper,
            srnetAUC,
            srnetCILower,
            srnetCIUpper,
            ignoreEntry
        )
        VALUES (?, ?, ?, ?, ?, ?, ? ,? ,?)  
        """,
        (
            combinationIndex + 0.1,
            "LSBR",
            srmAUC,
            srmCILower,
            srmCIUpper,
            srnetAUC,
            srnetCILower,
            srnetCIUpper,
            0
        )
    )
    conn.commit()
    
    #LSBM
    srmAUC, srmCILower, srmCIUpper, srnetAUC, srnetCILower, srnetCIUpper = ProcessExistingMethod(
        achievedBPP,
        "LSBM",
        srmModels, 
        srnetModels,
        coverSRMs,
        LSBMatching
    )
    
    cursor.execute("""
        INSERT INTO ExistingAUCs(
            combinationIndex,
            method,
            srmAUC,
            srmCILower,
            srmCIUpper,
            srnetAUC,
            srnetCILower,
            srnetCIUpper,
            ignoreEntry
        )
        VALUES (?, ?, ?, ?, ?, ?, ? ,? ,?)  
        """,
        (
            combinationIndex + 0.2,
            "LSBM",
            srmAUC,
            srmCILower,
            srmCIUpper,
            srnetAUC,
            srnetCILower,
            srnetCIUpper,
            0
        )
    )
    conn.commit()
    
    #PPM
    if(achievedBPP > 0.5): #Max BPP = 0.5
        cursor.execute("""
            INSERT INTO ExistingAUCs(
                combinationIndex,
                method,
                srmAUC,
                srmCILower,
                srmCIUpper,
                srnetAUC,
                srnetCILower,
                srnetCIUpper,
                ignoreEntry
            )
            VALUES (?, ?, ?, ?, ?, ?, ? ,? ,?)   
            """,
            (
                combinationIndex + 0.3,
                "PPM",
                0,
                0,
                0,
                0,
                0,
                0,
                1
            )
        )
        
        conn.commit()

    else:
        srmAUC, srmCILower, srmCIUpper, srnetAUC, srnetCILower, srnetCIUpper = ProcessExistingMethod(
            achievedBPP,
            "PPM",
            srmModels, 
            srnetModels,
            coverSRMs,
            PixelPairMatching
        )
            
        cursor.execute("""
            INSERT INTO ExistingAUCs(
                combinationIndex,
                method,
                srmAUC,
                srmCILower,
                srmCIUpper,
                srnetAUC,
                srnetCILower,
                srnetCIUpper,
                ignoreEntry
            )
            VALUES (?, ?, ?, ?, ?, ?, ? ,? ,?)  
            """,
            (
                combinationIndex + 0.3,
                "PPM",
                srmAUC,
                srmCILower,
                srmCIUpper,
                srnetAUC,
                srnetCILower,
                srnetCIUpper,
                0
            )
        )
        
    conn.commit()
    
    #completedCount.append(0)
    
    #print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - {len(completedCount)} complete", end = "", flush = True)
    conn.close()

def OrganiseRowData(rawRowData, isComposite):
    out = dict()
    for row in rawRowData:
        #if(row[-1] == 1):
        #    continue
        
        if(isComposite):
            subDict = {
                "srmAUC"            : row[4],
                "srmCILower"        : row[5],
                "srmCIUpper"        : row[6],
                "srnetAUC"          : row[7],
                "srnetCILower"      : row[8],
                "srnetCIUpper"      : row[9],
                "ignore"            : row[-1]
            }
        
        else:
            subDict = {
                "srmAUC"            : row[2],
                "srmCILower"        : row[3],
                "srmCIUpper"        : row[4],
                "srnetAUC"          : row[5],
                "srnetCILower"      : row[6],
                "srnetCIUpper"      : row[7],
                "ignore"            : row[-1]
            }
            
        out[row[0]] = subDict

    return out

def PlotHeatmaps(organisedCompositeData, organisedExistingData):
    #X axis - BPP
    #Y axis - Threshold
    #Values - difference against avg auc for existing method
    
    fig, axes = plt.subplots(
        nrows = 2, 
        ncols = 4, 
        figsize = HEATMAP_FIGSIZE, 
        gridspec_kw = {
            "width_ratios" : [1, 1, 1, COLOUR_BAR_RELATIVE_SIZE]
        }
    )

    for i, blockSize in enumerate(list(THRESHOLDS.keys())):
        srmHeatmapData = [
            [], #0.1 bpp
            [], #0.5 bpp
            []  #1.0 bpp
        ]
        
        srnetHeatmapData = [
            [], #0.1 bpp
            [], #0.5 bpp
            []  #1.0 bpp
        ]
        
        xLabels = []
        yLabels = []
        
        for j, bpp in enumerate(PURPORTED_BPPS):
            yLabels.append(bpp)
            
            for k, threshold in enumerate(THRESHOLDS[blockSize]):
                if(j == 0):
                    xLabels.append(str(threshold) if threshold != inf else "None")
                
                idx = 100 * (i + 1) + 10 * (k + 1) + (j + 1)

                organisedEntry = organisedCompositeData[idx]

                if(organisedEntry["ignore"] == 1):
                    srmHeatmapData[j].append(float("nan"))
                    srnetHeatmapData[j].append(float("nan"))
                    continue
                
                lsbrEntry = organisedExistingData[idx + 0.1]
                lsbmEntry = organisedExistingData[idx + 0.2]
                ppmEntry = organisedExistingData[idx + 0.3]
                
                meanSRMAUC = (lsbrEntry["srmAUC"] + lsbmEntry["srmAUC"] + (ppmEntry["srmAUC"] if ppmEntry["ignore"] != 1 else 0))
                meanSRMAUC /= (3 if ppmEntry["ignore"] != 1 else 2)
                
                meanSRNetAUC = (lsbrEntry["srnetAUC"] + lsbmEntry["srnetAUC"] + (ppmEntry["srnetAUC"] if ppmEntry["ignore"] != 1 else 0))
                meanSRNetAUC /= (3 if ppmEntry["ignore"] != 1 else 2)
                
                srmHeatmapData[j].append(meanSRMAUC - organisedEntry["srmAUC"])
                srnetHeatmapData[j].append(meanSRNetAUC - organisedEntry["srnetAUC"])

        srmNP = np.array(srmHeatmapData, dtype=float)
        srmMask = np.isnan(srmNP) 
        
        axSRM = axes[0, i]
        sns.heatmap(
            srmNP,
            mask=srmMask,
            xticklabels=xLabels,
            yticklabels=yLabels,
            vmin = -0.06,
            vmax = 0.10,
            center=0,
            linewidths=0.5,
            cmap=CMAP_THEME,
            linecolor="black",
            annot=True,
            fmt=".3f",
            ax=axSRM,
            cbar = (
                i == 2
            ),
            cbar_ax=(
                axes[0,3] if i == 2 else None
            )
        )
        
        axSRM.set_facecolor("darkgrey")
        
        axSRM.axvline(len(srmHeatmapData[0]), color="black", linewidth=2.0)
        axSRM.axhline(len(srmHeatmapData), color="black", linewidth=2.0)

        axSRM.set_xlabel("Security Threshold $α_T$", fontsize=12)
        axSRM.set_ylabel("BPP", fontsize=12)
        axSRM.set_title(f"SRM ΔAUCs for Block Size = {blockSize}x{blockSize} For Set B")
        #"Heatmap of for $α_T$ Against BPP"

        #plt.savefig(os.path.join(GRAPHS, "Comparision of Composite Efficacy", f"SRMHeatmap_{blockSize}.png"), dpi=600)
        #plt.close()
        
        srnetNP = np.array(srnetHeatmapData, dtype=float)
        srnetMask = np.isnan(srnetNP) 
        
        axSRNet = axes[1, i]
        sns.heatmap(
            srnetNP,
            mask=srnetMask,
            xticklabels=xLabels,
            yticklabels=yLabels,
            vmin = -0.06,
            vmax = 0.10,
            center=0,
            linewidths=0.5,
            cmap=CMAP_THEME,
            linecolor="black",
            annot=True,
            fmt = ".3f",
            ax = axSRNet,
            cbar = (
                i == 2
            ),
            cbar_ax=(
                axes[1,3] if i == 2 else None
            )
        )
        
        axSRNet.set_facecolor("darkgrey")
        
        axSRNet.axvline(len(srnetHeatmapData[0]), color="black", linewidth=2.0)
        axSRNet.axhline(len(srnetHeatmapData), color="black", linewidth=2.0)

        axSRNet.set_xlabel("Security Threshold $α_T$", fontsize=12)
        axSRNet.set_ylabel("BPP", fontsize=12)
        axSRNet.set_title(f"SRNet ΔAUCs for Block Size = {blockSize}x{blockSize} For Set B")

    plt.tight_layout()
    os.makedirs(
        os.path.join(GRAPHS, "Comparision of Composite Efficacy"),
        exist_ok=True
    )
    
    plt.savefig(os.path.join(GRAPHS, "Comparision of Composite Efficacy", "CombinedSRMAndSRNetHeatmap.png"), dpi=600)
    plt.close()

def GetParetoFrontier(xs, ys, maximiseX=True, maximiseY=True):
    points = sorted(zip(xs, ys), key=lambda p: p[0], reverse=not maximiseX)

    efficientPoints = []
    bestY = -float("inf") if maximiseY else float("inf")

    for x, y in points:
        if (maximiseY and y > bestY) or (not maximiseY and y < bestY):
            efficientPoints.append((x, y))
            bestY = y

    if not efficientPoints:
        return [], []
    px, py = zip(*efficientPoints)
    return list(px), list(py)

def PlotPareto(organisedCompositeData, organisedExistingData):
    
    #x axis - achieved BPP
    #y axis - delta AUC
    
    fig, axes = plt.subplots(
        nrows = 3, 
        ncols = 3, 
        figsize = PARETO_FIGSIZE, 
    )
    
    for i, blockSize in enumerate(list(THRESHOLDS.keys())):
        axSRM = axes[0, i]
        axSRNet = axes[1, i]
        axSum = axes[2, i]
        
        axSRM.grid(alpha=0.3)
        axSRNet.grid(alpha=0.3)
        axSum.grid(alpha=0.3)
        
        srmAllX, srmAllY = [], []
        srnetAllX, srnetAllY = [], []
        sumAllX, sumAllY = [], []
        
        for j, bpp in enumerate(PURPORTED_BPPS):
            
            for k, threshold in enumerate(THRESHOLDS[blockSize]):
                idx = 100 * (i + 1) + 10 * (k + 1) + (j + 1)
                
                organisedEntry = organisedCompositeData[idx]
                if(organisedEntry["ignore"] == 1):
                    continue
                
                #First we need to find the achieved bpp for the x axis
                npyFile = np.load(os.path.join(
                    NPY_FILES,
                    "Achieved BPP VS Purported BPP",
                    f"{blockSize}-{threshold if threshold != inf else "inf"}-BPP.npy"     
                ))
                
                step = 100 // len(npyFile) 
                achievedBPP = npyFile[int(bpp * 100) // step - 1]
                
                #Next we find the deltaAUC
                lsbrEntry = organisedExistingData[idx + 0.1]
                lsbmEntry = organisedExistingData[idx + 0.2]
                ppmEntry = organisedExistingData[idx + 0.3]
                
                meanSRMAUC = (lsbrEntry["srmAUC"] + lsbmEntry["srmAUC"] + (ppmEntry["srmAUC"] if ppmEntry["ignore"] != 1 else 0))
                meanSRMAUC /= (3 if ppmEntry["ignore"] != 1 else 2)
                
                meanSRNetAUC = (lsbrEntry["srnetAUC"] + lsbmEntry["srnetAUC"] + (ppmEntry["srnetAUC"] if ppmEntry["ignore"] != 1 else 0))
                meanSRNetAUC /= (3 if ppmEntry["ignore"] != 1 else 2)
                
                srmAllX.append(achievedBPP)
                srmAllY.append(meanSRMAUC - organisedEntry["srmAUC"])
                
                srnetAllX.append(achievedBPP)
                srnetAllY.append(meanSRNetAUC - organisedEntry["srnetAUC"])
                
                sumAllX.append(achievedBPP)
                sumAllY.append(meanSRMAUC - organisedEntry["srmAUC"] + meanSRNetAUC - organisedEntry["srnetAUC"])
              
                axSRM.scatter(
                    [achievedBPP],
                    [meanSRMAUC - organisedEntry["srmAUC"]],
                    c = PARETO_COLOURS[j],
                    marker = PARETO_MARKERS[k], 
                    #label = f"Purported BPP = {str(bpp)}"  
                )
                
                axSRNet.scatter(
                    [achievedBPP],
                    [meanSRNetAUC - organisedEntry["srnetAUC"]],
                    c = PARETO_COLOURS[j],
                    marker = PARETO_MARKERS[k], 
                    #label = f"Purported BPP = {str(bpp)}"    
                )
                
                axSum.scatter(
                    [achievedBPP],
                    [meanSRMAUC - organisedEntry["srmAUC"] + meanSRNetAUC - organisedEntry["srnetAUC"]],
                    c = PARETO_COLOURS[j],
                    marker = PARETO_MARKERS[k]
                )
        
        axSRM.set_xlim([0.0, 1.0])
        axSRNet.set_xlim([0.0, 1.0])
        axSum.set_xlim([0.0, 1.0])
        
        srm_px, srm_py = GetParetoFrontier(
            srmAllX, srmAllY, maximiseX=False, maximiseY=True
        )
        if srm_px:
            axSRM.plot(
                srm_px,
                srm_py,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label="Pareto Frontier",
            )

        srnet_px, srnet_py = GetParetoFrontier(
            srnetAllX, srnetAllY, maximiseX=False, maximiseY=True
        )
        if srnet_px:
            axSRNet.plot(
                srnet_px,
                srnet_py,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label="Pareto Frontier",
            )
        
        sum_px, sum_py = GetParetoFrontier(
            sumAllX, sumAllY, maximiseX=False, maximiseY=True
        )
        if sum_px:
            axSum.plot(
                sum_px,
                sum_py,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label="Pareto Frontier",
            )
        
        #Plotting the graphs
        bppHandles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PARETO_COLOURS[j],
                markersize=8,
                label=f"BPP = {bpp}",
            )
            for j, bpp in enumerate(PURPORTED_BPPS)
        ]

        thresholdHandles = [
            Line2D(
                [0],
                [0],
                marker=PARETO_MARKERS[k],
                color="black",
                linestyle="None",
                markersize=8,
                label=f"$\\alpha_T$ = {t if t != inf else 'None'}",
            )
            for k, t in enumerate(THRESHOLDS[blockSize])
        ]

        combinedHandles = thresholdHandles + bppHandles
        axSRM.legend(
            handles=combinedHandles, loc="best", fontsize=7, ncol=2, framealpha=0.7
        )
        axSRNet.legend(
            handles=combinedHandles, loc="best", fontsize=7, ncol=2, framealpha=0.7
        )
        
        axSum.legend(
            handles=combinedHandles, loc="best", fontsize=7, ncol=2, framealpha=0.7
        )
        
        axSRM.set_xlabel("Achieved BPP")
        axSRM.set_ylabel("ΔAUC")
        
        axSRNet.set_xlabel("Achieved BPP")
        axSRNet.set_ylabel("ΔAUC")
        
        axSum.set_xlabel("Achieved BPP")
        axSum.set_ylabel("ΔAUC")
        
        axSRM.set_title(f"SRM Pareto Frontier for Block Size = {blockSize}x{blockSize} For Set B")
        axSRNet.set_title(f"SRNet Pareto Frontier for Block Size = {blockSize}x{blockSize} For Set B")
        axSum.set_title(f"Sum Pareto Frontier for Block Size = {blockSize}x{blockSize} For Set B")

    plt.tight_layout()
    os.makedirs(
        os.path.join(GRAPHS, "Comparision of Composite Efficacy"),
        exist_ok=True
    )
    
    plt.savefig(os.path.join(GRAPHS, "Comparision of Composite Efficacy", "CombinedSRMAndSRNetPareto.png"), dpi=600)
    plt.close()
        
if __name__ == "__main__":
    mode = input("(G)enerate or (P)lot : ").upper()
    
    if(mode == "G"):
        print(f"{datetime.now().strftime('%H:%M:%S')} - Starting")
        
        os.makedirs(
            os.path.join(NPY_FILES, "Composite Efficacy Comparison"),
            exist_ok=True
        )
        
        #Defining the database
        conn = sqlite3.connect(DB)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS CompositeAUCs (
                combinationIndex INTEGER PRIMARY KEY,
                blockSize INT,
                purportedBPP FLOAT,
                threshold FLOAT,
                srmAUC FLOAT,
                srmCILower FLOAT,
                srmCIUpper FLOAT,
                srnetAUC FLOAT,
                srnetCILower FLOAT,
                srnetCIUpper FLOAT,
                ignoreEntry INT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ExistingAUCs (
                combinationIndex FLOAT NOT NULL,
                method TINYTEXT,
                srmAUC FLOAT,
                srmCILower FLOAT,
                srmCIUpper FLOAT,
                srnetAUC FLOAT,
                srnetCILower FLOAT,
                srnetCIUpper FLOAT,
                ignoreEntry INT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

        print(f"{datetime.now().strftime('%H:%M:%S')} - Setup Complete")

        paths = [
            os.path.join(BOSSBASE_B, path) for path in sorted(os.listdir(BOSSBASE_B))
        ]
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            coverSRMs = list(exe.map(CoverSRM, paths))

        print(f"{datetime.now().strftime('%H:%M:%S')} - Cover SRMs Extracted - len = {len(coverSRMs)}")

        arguments = []
        for blockSize in THRESHOLDS.keys():
            for threshold in THRESHOLDS[blockSize]:
                for purportedBPP in PURPORTED_BPPS:
                    arguments.append((
                        blockSize,
                        threshold,
                        purportedBPP,
                        coverSRMs
                    ))  
        
        print(f"{datetime.now().strftime('%H:%M:%S')} - Arguments Generated")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            list(exe.map(Iteration, arguments))

        print(f"{datetime.now().strftime('%H:%M:%S')} - Processing Complete")
    
    else:
        conn = sqlite3.connect(DB)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM CompositeAUCs
        """)
        
        compositeRows = OrganiseRowData(cursor.fetchall(), True)
        
        cursor.execute("""
            SELECT * FROM ExistingAUCs               
        """)
        
        existingRows = OrganiseRowData(cursor.fetchall(), False)
        
        PlotHeatmaps(compositeRows, existingRows)
        
        PlotPareto(compositeRows, existingRows)
        