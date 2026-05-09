import os
import sqlite3
import pandas as pd
import seaborn as sns
from math import ceil
from datetime import datetime
import matplotlib.pyplot as plt
from Helpers.HelperFunctions import *
from matplotlib.colors import LogNorm
from Helpers.SteganalysisMethods import *
from Helpers.EmbeddingAlgorithms import *
from Helpers.TestingScriptFuncVersion import CompositeMethod
from multiprocessing import Process

print(f"{datetime.now().strftime("%H:%M:%S")} - Imports Complete")

imageSize = (256,256)
BOSSBASE_FOLDER = GetPaths()["Bossbase Path"]
DATABASE_NAME = GetPaths()["Database"]
paths = os.listdir(BOSSBASE_FOLDER)
DEVIATION_COEFFICENTS = {
    "Chi Square Attack" : -1/580.76,
    "PSNR" : 40,
    "Zhang" : 1/10856.838
} 

def GenerateBlockSizeHeatmapData(isIndexBlockMethod : bool, thresholdArg : float):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CompositeHeatmapBlockSize
    (
        EmbedPercentage FLOAT NOT NULL,
        BlockSize INT NOT NULL,
        Security FLOAT NOT NULL,
        IsIndexBlockMethod INTEGER NOT NULL
    )""")
    conn.commit()

    coverBlocks = []
    for path in paths:
        block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
        coverBlocks.append(block)

    embedData = []
    with open("Lipsum.txt", "r") as f:
        embedData = [*f.read()]
        
        
    #Finding amount of bytes to embed

    for i in range(25,101,25):
        totalImageBytes = (imageSize[0] * imageSize[1]) / 8
        secretBytesAmount = int(totalImageBytes * (i/100))

        #!TEMP - CAHNGE THIS FOR DIFFERENT SIZES
        for j in range(3,8):
            blockSize = (2 ** j, 2**j)

            cursor.execute("SELECT * FROM CompositeHeatmapBlockSize WHERE EmbedPercentage = ? AND BlockSize = ? AND IsIndexBlockMethod = ?", (i, blockSize[0], int(isIndexBlockMethod))) 
            rows = cursor.fetchone()
            if not(rows == None or rows == []):
                continue
            
            #Finding the bytes per section to attempt
            bytesPerSection = ceil(secretBytesAmount / ((imageSize[0] / blockSize[0]) ** 2))
            
            print(f"Secret Bytes Amount : {secretBytesAmount} | Bytes Per Section : {bytesPerSection}")
            
            index = 0

            totalThresholdSum = 0
            for path in paths:
                
                if(index % 10 == 0):
                    print(f"{datetime.now().strftime("%H:%M:%S")} - Index : {index}")
                
                stegoBlock, _, _ = CompositeMethod(embedData[:secretBytesAmount], os.path.join(BOSSBASE_FOLDER, path), acceptableMappingThreshold=thresholdArg, bytesPerSection=bytesPerSection, blocksPerSide=int(256 / blockSize[0]), indexBlockMethod=isIndexBlockMethod)
                
                consideredBlock = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
                coverMappings = {
                    "Chi Square Attack" : ChiSquareAttack(consideredBlock),
                    "Zhang" : ZhangLSBMatching(consideredBlock)
                }
                threshold = (
                    (ChiSquareAttack(stegoBlock) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"]
                    + (ZhangLSBMatching(stegoBlock) - coverMappings["Zhang"]) * DEVIATION_COEFFICENTS["Zhang"]
                    + DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoBlock)
                )
                
                totalThresholdSum += threshold
                index += 1
            
            totalThresholdSum /= index
            cursor.execute("""INSERT INTO CompositeHeatmapBlockSize
            (EmbedPercentage, BlockSize, Security, IsIndexBlockMethod)
            VALUES (?, ?, ?, ?)""",(i, blockSize[0], totalThresholdSum, int(isIndexBlockMethod)))
            conn.commit()
            print(f"{datetime.now().strftime("%H:%M:%S")} - {i}|{j} complete")
            
    conn.close()
    
def HeatmapDisplay():
    conn = sqlite3.connect(DATABASE_NAME)

    #Block Size Data
    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapBlockSize WHERE BlockSize != 256 AND IsIndexBlockMethod = 1 ", conn) 
    heatmapData = df.pivot( index="BlockSize", columns="EmbedPercentage", values="Security" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt=""
    )
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Block Size")
    plt.title(f"Heatmap Of α For Block Size Against Embed % For First 1000 Images Of BossBase Using Index Block Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Block Size VS Embedding Rate Composite Alpha Heatmap 1000 Index Method.png"), dpi=600)
    
    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapBlockSize WHERE BlockSize != 256 AND IsIndexBlockMethod = 0 ", conn) 
    heatmapData = df.pivot( index="BlockSize", columns="EmbedPercentage", values="Security" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt=""
    )
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Block Size")
    plt.title(f"Heatmap Of α For Block Size Against Embed % For First 1000 Images Of BossBase Using Block Marking Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Block Size VS Embedding Rate Composite Alpha Heatmap 1000 Mark Method.png"), dpi=600)

    #Threshold Data
    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapThreshold WHERE IsIndexBlockMethod = 1", conn) 
    heatmapData = df.pivot( index="Threshold", columns="EmbedPercentage", values="Security" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt=""
    )
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Security Threshold [α_t]")
    plt.title(f"Heatmap Of α For Threshold (α_t) Against Embed % For First 1000 Images Of BossBase Using Index Block Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Block Size VS Threshold Composite Alpha Heatmap 1000 Index Method.png"), dpi=600)

    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapThreshold WHERE IsIndexBlockMethod = 0", conn) 
    heatmapData = df.pivot( index="Threshold", columns="EmbedPercentage", values="Security" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt=""
    )
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Security Threshold [α_t]")
    plt.title(f"Heatmap Of α For Threshold (α_t) Against Embed % For First 1000 Images Of BossBase Using Block Marking Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Block Size VS Threshold Composite Alpha Heatmap 1000 Mark Method.png"), dpi=600)
    
    #*Failure data
    
    #Block Size Data
    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapBlockSizeFailures WHERE BlockSize != 256 AND IsIndexBlockMethod = 1 ", conn) 
    heatmapData = df.pivot( index="BlockSize", columns="EmbedPercentage", values="Failures" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt=""
    )
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Block Size")
    plt.title(f"Heatmap Of No. Of Failures For Block Size Against Embed % For First 1000 Images Of BossBase Using Index Block Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Block Size VS Embedding Rate Composite Failure Heatmap 1000 Index Method.png"), dpi=600)
    
    
    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapBlockSizeFailures WHERE BlockSize != 256 AND IsIndexBlockMethod = 0 ", conn) 
    heatmapData = df.pivot( index="BlockSize", columns="EmbedPercentage", values="Failures" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt=""
    )
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Block Size")
    plt.title(f"Heatmap Of No. Of Failures For Block Size Against Embed % For First 1000 Images Of BossBase Using Block Marking Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Block Size VS Embedding Rate Composite Failure Heatmap 1000 Mark Method.png"), dpi=600)
 
    #Threshold Data
    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapThresholdFailures WHERE IsIndexBlockMethod = 1", conn) 
    heatmapData = df.pivot( index="Threshold", columns="EmbedPercentage", values="Failures" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    
    #!LOG STUFF
    heatmapData = heatmapData.replace(0, 1e-6)
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt="",
        norm=LogNorm(
            vmin=heatmapData[heatmapData > 0].min().min(),  # smallest non-zero
            vmax=heatmapData.max().max()
        )
    )
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Security Threshold [α_t]")
    plt.title(f"Heatmap Of Failures For Threshold (α_t) Against Embed % For First 1000 Images Of BossBase Using Index Block Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Threshold VS Embedding Rate Composite Failure Heatmap 1000 Index Method.png"), dpi=600)

    df = pd.read_sql_query("SELECT * FROM CompositeHeatmapThresholdFailures WHERE IsIndexBlockMethod = 0", conn) 
    heatmapData = df.pivot( index="Threshold", columns="EmbedPercentage", values="Failures" ) 
    annotLabels = heatmapData.map(lambda x: f"{x:.3g}")
    plt.figure(figsize=(12, 5))
    #!LOG STUFF
    heatmapData = heatmapData.replace(0, 1e-6)
    sns.heatmap(
        heatmapData,
        cmap="YlOrRd",
        annot=annotLabels,
        fmt="",
        norm=LogNorm(
            vmin=heatmapData[heatmapData > 0].min().min(),  # smallest non-zero
            vmax=heatmapData.max().max()
        )
    )
    
    plt.xlabel("Embedding Percentage (%)")
    plt.ylabel("Security Threshold [α_t]")
    plt.title(f"Heatmap Of Failures For Threshold (α_t) Against Embed % For First 1000 Images Of BossBase Using Block Marking Method")
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join("Updated Folder", f"Threshold vs Embedding Rate Composite Failures Heatmap 1000 Mark Method.png"), dpi=600)

def GenerateThresholdHeatmapData(isIndexBlockMethod : bool, blockSize : tuple[int, int] = (64, 64)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CompositeHeatmapThreshold
    (
        EmbedPercentage FLOAT NOT NULL,
        Threshold FLOAT NOT NULL,
        Security FLOAT NOT NULL,
        IsIndexBlockMethod INTEGER NOT NULL
    )""")
    conn.commit()

    coverBlocks = []
    for path in paths:
        block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
        coverBlocks.append(block)

    embedData = []
    with open("Lipsum.txt", "r") as f:
        embedData = [*f.read()]
        
        
    #Finding amount of bytes to embed

    for k in range(25,225,25):
        i = k / 2
        totalImageBytes = (imageSize[0] * imageSize[1]) / 8
        secretBytesAmount = int(totalImageBytes * (i/100))

        for l in range(1, 11):
            j = l / 2

            #Seeing if its already in the SQL - If it is, we skip to the next one
            cursor.execute("SELECT * FROM CompositeHeatmapThreshold WHERE EmbedPercentage = ? AND Threshold = ? AND isIndexBlockMethod = ?", (i, j, int(isIndexBlockMethod))) 
            rows = cursor.fetchone()
            if not(rows == None or rows == []):
                continue

            #Finding the bytes per section to attempt
            bytesPerSection = ceil(secretBytesAmount / ((imageSize[0] / blockSize[0]) ** 2))
            
            print(f"Secret Bytes Amount : {secretBytesAmount} | Bytes Per Section : {bytesPerSection}")
            
            index = 0

            totalThresholdSum = 0
            for path in paths:
                
                if(index % 10 == 0):
                    print(f"{datetime.now().strftime("%H:%M:%S")} - Index : {index}")
                
                stegoBlock, _, _ = CompositeMethod(embedData[:secretBytesAmount], os.path.join(BOSSBASE_FOLDER, path), acceptableMappingThreshold=j, bytesPerSection=bytesPerSection, blocksPerSide=int(256 / blockSize[0]), indexBlockMethod=isIndexBlockMethod)
                
                consideredBlock = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
                coverMappings = {
                    "Chi Square Attack" : ChiSquareAttack(consideredBlock),
                    "Zhang" : ZhangLSBMatching(consideredBlock)
                }
                threshold = (
                    (ChiSquareAttack(stegoBlock) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"]
                    + (ZhangLSBMatching(stegoBlock) - coverMappings["Zhang"]) * DEVIATION_COEFFICENTS["Zhang"]
                    + DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoBlock)
                )
                
                totalThresholdSum += threshold
                index += 1
            
            totalThresholdSum /= index
            cursor.execute("""INSERT INTO CompositeHeatmapThreshold
            (EmbedPercentage, Threshold, Security, IsIndexBlockMethod)
            VALUES (?, ?, ?, ?)""",(i, j, totalThresholdSum, int(isIndexBlockMethod)))
            conn.commit()
            print(f"{datetime.now().strftime("%H:%M:%S")} - {i}|{j} complete")
            
    conn.close()

def GenerateThresholdFailureHeatmapData(isIndexBlockMethod : bool, blockSize : tuple[int, int] = (64, 64)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CompositeHeatmapThresholdFailures
    (
        EmbedPercentage FLOAT NOT NULL,
        Threshold FLOAT NOT NULL,
        Failures FLOAT NOT NULL,
        IsIndexBlockMethod INTEGER NOT NULL
    )""")
    conn.commit()

    coverBlocks = []
    for path in paths:
        block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
        coverBlocks.append(block)

    embedData = []
    with open("Lipsum.txt", "r") as f:
        embedData = [*f.read()]
        
        
    #Finding amount of bytes to embed

    for k in range(25,225,25):
        i = k / 2
        totalImageBytes = (imageSize[0] * imageSize[1]) / 8
        secretBytesAmount = int(totalImageBytes * (i/100))

        for l in range(1, 11):
            j = l / 2

            #Seeing if its already in the SQL - If it is, we skip to the next one
            cursor.execute("SELECT * FROM CompositeHeatmapThresholdFailures WHERE EmbedPercentage = ? AND Threshold = ? AND isIndexBlockMethod = ?", (i, j, int(isIndexBlockMethod))) 
            rows = cursor.fetchone()
            if not(rows == None or rows == []):
                continue

            #Finding the bytes per section to attempt
            bytesPerSection = ceil(secretBytesAmount / ((imageSize[0] / blockSize[0]) ** 2))
            
            print(f"Secret Bytes Amount : {secretBytesAmount} | Bytes Per Section : {bytesPerSection}")
            
            index = 0

            totalFailureSum = 0
            for path in paths:
                
                if(index % 10 == 0):
                    print(f"{datetime.now().strftime("%H:%M:%S")} - Index : {index}")
                
                _, blockCountData, _ = CompositeMethod(embedData[:secretBytesAmount], os.path.join(BOSSBASE_FOLDER, path), acceptableMappingThreshold=j, bytesPerSection=bytesPerSection, blocksPerSide=int(256 / blockSize[0]), indexBlockMethod=isIndexBlockMethod)
                
                totalFailureSum += blockCountData[0]
                index += 1
            
            totalFailureSum /= index
            cursor.execute("""INSERT INTO CompositeHeatmapThresholdFailures
            (EmbedPercentage, Threshold, Failures, IsIndexBlockMethod)
            VALUES (?, ?, ?, ?)""",(i, j, totalFailureSum, int(isIndexBlockMethod)))
            conn.commit()
            print(f"{datetime.now().strftime("%H:%M:%S")} - {i}|{j} complete")
            
    conn.close()

def GenerateBlockSizeFailureHeatmapData(isIndexBlockMethod : bool, thresholdArg : float):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CompositeHeatmapBlockSizeFailures
    (
        EmbedPercentage FLOAT NOT NULL,
        BlockSize INT NOT NULL,
        Failures FLOAT NOT NULL,
        IsIndexBlockMethod INTEGER NOT NULL
    )""")
    conn.commit()

    coverBlocks = []
    for path in paths:
        block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
        coverBlocks.append(block)

    embedData = []
    with open("Lipsum.txt", "r") as f:
        embedData = [*f.read()]
        
        
    #Finding amount of bytes to embed

    for i in range(25,101,25):
        totalImageBytes = (imageSize[0] * imageSize[1]) / 8
        secretBytesAmount = int(totalImageBytes * (i/100))

        #!TEMP - CAHNGE THIS FOR DIFFERENT SIZES
        for j in range(3,8):
            blockSize = (2 ** j, 2**j)

            cursor.execute("SELECT * FROM CompositeHeatmapBlockSizeFailures WHERE EmbedPercentage = ? AND BlockSize = ? AND IsIndexBlockMethod = ?", (i, blockSize[0], int(isIndexBlockMethod))) 
            rows = cursor.fetchone()
            if not(rows == None or rows == []):
                continue
            
            #Finding the bytes per section to attempt
            bytesPerSection = ceil(secretBytesAmount / ((imageSize[0] / blockSize[0]) ** 2))
            
            print(f"Secret Bytes Amount : {secretBytesAmount} | Bytes Per Section : {bytesPerSection}")
            
            index = 0

            totalFailureSum = 0
            for path in paths:
                
                if(index % 10 == 0):
                    print(f"{datetime.now().strftime("%H:%M:%S")} - Index : {index}")
                
                _, blockCountData, _ = CompositeMethod(embedData[:secretBytesAmount], os.path.join(BOSSBASE_FOLDER, path), acceptableMappingThreshold=thresholdArg, bytesPerSection=bytesPerSection, blocksPerSide=int(256 / blockSize[0]), indexBlockMethod=isIndexBlockMethod)
                
                
                totalFailureSum += blockCountData[0]
                index += 1
            
            totalFailureSum /= index
            cursor.execute("""INSERT INTO CompositeHeatmapBlockSizeFailures
            (EmbedPercentage, BlockSize, Failures, IsIndexBlockMethod)
            VALUES (?, ?, ?, ?)""",(i, blockSize[0], totalFailureSum, int(isIndexBlockMethod)))
            conn.commit()
            print(f"{datetime.now().strftime("%H:%M:%S")} - {i}|{j} complete")
            
    conn.close()
    
HeatmapDisplay()
"""if __name__ == "__main__":
    p1 = Process(target=GenerateBlockSizeHeatmapData,
                kwargs={"isIndexBlockMethod": False, "thresholdArg": 3})

    p2 = Process(target=GenerateBlockSizeHeatmapData,
                kwargs={"isIndexBlockMethod": True, "thresholdArg": 3})

    p3 = Process(target=GenerateBlockSizeFailureHeatmapData,
                kwargs={"isIndexBlockMethod": False, "thresholdArg": 3})

    p4 = Process(target=GenerateBlockSizeFailureHeatmapData,
                 kwargs={"isIndexBlockMethod": True, "thresholdArg": 3})

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    
    print("Set 1 complete")
    
    p1 = Process(target=GenerateThresholdHeatmapData,
                kwargs={"isIndexBlockMethod": False, "blockSize" : (64,64)})

    p2 = Process(target=GenerateThresholdHeatmapData,
                kwargs={"isIndexBlockMethod": True, "blockSize" : (64,64)})

    p3 = Process(target=GenerateThresholdFailureHeatmapData,
                kwargs={"isIndexBlockMethod": False, "blockSize" : (64,64)})

    p4 = Process(target=GenerateThresholdFailureHeatmapData,
                 kwargs={"isIndexBlockMethod": True, "blockSize" : (64,64)})

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()"""

"""GenerateBlockSizeHeatmapData(False, 3)
print("1 done")
GenerateBlockSizeHeatmapData(True, 3)
print("2 done")
GenerateThresholdHeatmapData(False, (64,64))

print("3 complete")

GenerateThresholdHeatmapData(True, (64,64))

print("4 Complete")
GenerateThresholdFailureHeatmapData(True, (64,64))

print("5 complete")

GenerateThresholdFailureHeatmapData(False, (64,64))

print("6 complete")

GenerateBlockSizeFailureHeatmapData(True, 3)

print("7 complete")

GenerateBlockSizeFailureHeatmapData(False, 3)"""