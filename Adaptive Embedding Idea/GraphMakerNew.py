from Helpers.SteganalysisMethods import *
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
from Helpers.TestingScriptFuncVersion import CompositeMethod
import matplotlib.pyplot as plt
from math import ceil
from datetime import datetime
from copy import deepcopy
import sqlite3
import os

print(f"{datetime.now().strftime("%H:%M:%S")} - Imports complete for GraphMakerSQLVersion")

BOSSBASE_FOLDER = GetPaths()["Bossbase Path"]
DATABASE_NAME = GetPaths()["Database"]
paths = os.listdir(BOSSBASE_FOLDER)

def SQLStoreExistingMethodsPSNRvEmbedRate(embedPath = "Lipsum.txt", imageSize = (256,256)):
    """
    **Parameters** : str="Lipsum.txt" (filepath with the secret to be embedded), (int,int) = (256,256) (Size of processed images)
    **Returns** : None
    
    - Looks at all 3 chosen embed methods and stores the average PSNR against embed rate for each method for first 1k images of BossBase
    - Stores the data in the SQL database
    """
    
    print(f"{datetime.now().strftime("%H:%M:%S")} - Starting setup")
    
    embedPercentage = []
    LSBData = []
    LSBMatchingData = []
    PPMData = []
    LSBChi2Data = []
    LSBZhangData = []
    LSBMatchingChi2Data = []
    LSBMatchingZhangData = []
    PPMChi2Data = []
    PPMZhangData = []
    
    embedData = []
    with open(embedPath, "r") as f:
        embedData = [*f.read()]
    
    coverBlocks = []
    for path in paths:
        block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
        coverBlocks.append(block)
    
    print(f"{datetime.now().strftime("%H:%M:%S")} - Starting increment loop")
    
    start = datetime.now()
    for i in range(0,101,2):
        if(i > 100):
            break
        
        embedPercentage.append(i)
        
        #Finding amount of bytes to embed
        totalImageBytes = (imageSize[0] * imageSize[1]) / 8
        secretBytesAmount = int(totalImageBytes * (i/100))
        
        #print(f"{i} : {secretBytesAmount}, {len(embedData[:secretBytesAmount])}")
        #print(f"L : {len(coverBlocks[0])}, {len(coverBlocks[0][0])}")
        
        LSBDataPerI = []
        LSBMatchingDataPerI = []
        PPMDataPerI = []
        LSBChi2DataPerI = []
        LSBZhangDataPerI = []
        LSBMatchingChi2DataPerI = []
        LSBMatchingZhangDataPerI = []
        PPMChi2DataPerI = []
        PPMZhangDataPerI = []
        
        for coverBlock in coverBlocks:
            secret = embedData[:secretBytesAmount]
            LSBBlock = StandardLSB(deepcopy(coverBlock), secret)
            LSBMatchingBlock = LSBMatching(deepcopy(coverBlock), secret)
            PPMBlock = PixelPairMatching(deepcopy(coverBlock), secret)

            LSBDataPerI.append(PSNR(coverBlock, LSBBlock))
            LSBMatchingDataPerI.append(PSNR(coverBlock, LSBMatchingBlock))
            PPMDataPerI.append(PSNR(coverBlock,PPMBlock))
            
            LSBChi2DataPerI.append(ChiSquareAttack(LSBBlock))
            LSBZhangDataPerI.append(ZhangLSBMatching(LSBBlock))
            LSBMatchingChi2DataPerI.append(ChiSquareAttack(LSBMatchingBlock))
            LSBMatchingZhangDataPerI.append(ZhangLSBMatching(LSBMatchingBlock))
            PPMChi2DataPerI.append(ChiSquareAttack(PPMBlock))
            PPMZhangDataPerI.append(ZhangLSBMatching(PPMBlock))
                    
        LSBData.append(sum(LSBDataPerI)/len(LSBDataPerI))
        LSBMatchingData.append(sum(LSBMatchingDataPerI)/len(LSBMatchingDataPerI))
        PPMData.append(sum(PPMDataPerI)/len(PPMDataPerI))
        LSBChi2Data.append(sum(LSBChi2DataPerI) / len(LSBChi2DataPerI))
        LSBZhangData.append(sum(LSBZhangDataPerI) / len(LSBZhangDataPerI))
        LSBMatchingChi2Data.append(sum(LSBMatchingChi2DataPerI) / len(LSBMatchingChi2DataPerI))
        LSBMatchingZhangData.append(sum(LSBMatchingZhangDataPerI) / len(LSBMatchingZhangDataPerI))
        PPMChi2Data.append(sum(PPMChi2DataPerI) / len(PPMChi2DataPerI))
        PPMZhangData.append(sum(PPMZhangDataPerI) / len(PPMZhangDataPerI))
        
        print(f"{datetime.now().strftime("%H:%M:%S")} - {i}% done")
    
    end = datetime.now()
    print(f"Time taken : {end - start}")
    
    #print(f"LSB : {LSBData}")
    #print(f"LSBM : {LSBMatchingData}")
    #print(f"PPM : {PPMData}")
        
    #Storing the info in SQL
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ExistingMethodsvsEmbedRate
    (
        EmbedType TEXT NOT NULL,
        EmbedPercentage FLOAT NOT NULL,
        PSNR FLOAT NOT NULL,
        Chi2 FLOAT NOT NULL,
        Zhang FLOAT NOT NULL
    )""")
    conn.commit()
    
    for i in range(len(embedPercentage)):
        cursor.execute("""
        INSERT INTO ExistingMethodsvsEmbedRate 
        (EmbedType, EmbedPercentage, PSNR, Chi2, Zhang)
        VALUES (?,?,?,?,?)""", 
        ("LSB", embedPercentage[i], LSBData[i], LSBChi2Data[i], LSBZhangData[i]))
        cursor.execute("""
        INSERT INTO ExistingMethodsvsEmbedRate 
        (EmbedType, EmbedPercentage, PSNR, Chi2, Zhang)
        VALUES (?,?,?,?,?)""", 
        ("Matching", embedPercentage[i], LSBMatchingData[i], LSBMatchingChi2Data[i], LSBMatchingZhangData[i]))
        cursor.execute("""
        INSERT INTO ExistingMethodsvsEmbedRate 
        (EmbedType, EmbedPercentage, PSNR, Chi2, Zhang)
        VALUES (?,?,?,?,?)""", 
        ("PPM", embedPercentage[i], PPMData[i], PPMChi2Data[i], PPMZhangData[i]))
    
    conn.commit()
    conn.close()

def SQLStoreCompositeMethodPSNRvEmbedRate(imageSize : tuple = (256,256), blockSize : tuple = (64,64), threshold=3):
    """
    
    """
    
    coverBlocks = []
    for path in paths:
        block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
        coverBlocks.append(block)
    
    embedData = []
    with open("Lipsum.txt", "r") as f:
        embedData = [*f.read()]
    
    embedPercentage = []
    psnrData = []
    failureData = []
    lsbData = []
    matchingData = []
    ppmData = []
    Chi2Data = []
    ZhangData = []
    
    print(f"{datetime.now().strftime("%H:%M:%S")} - Loop Start")
    start = datetime.now()
    for i in range(2,101,2):
        if(i > 100):
            break
        
        psnrDataPerI = []
        failureDataPerI = []     
        lsbDataPerI = []
        matchingDataPerI = []
        ppmDataPerI = []   
        Chi2DataPerI = []
        ZhangDataPerI = []

        embedPercentage.append(i)
        
        #Finding amount of bytes to embed
        totalImageBytes = (imageSize[0] * imageSize[1]) / 8
        secretBytesAmount = int(totalImageBytes * (i/100))

        #Finding the bytes per section to attempt
        bytesPerSection = ceil(secretBytesAmount / ((imageSize[0] / blockSize[0]) ** 2))
        index = 0
        for path in paths:
            #!THIS IS CURRENTLY SET FOR 32 * 32 - REMEBMER TO CHANGE FOR 64 * 64
            stegoBlock, (failureCount, lsbCount, matchingCount, ppmCount), _ = CompositeMethod(embedData[:secretBytesAmount], os.path.join(BOSSBASE_FOLDER, path), acceptableMappingThreshold=threshold, bytesPerSection=bytesPerSection, blocksPerSide=8)

            psnr = PSNR(coverBlocks[index], stegoBlock)
            if(psnr != float("inf")):
                psnrDataPerI.append(psnr)
            failureDataPerI.append(failureCount)
            lsbDataPerI.append(lsbCount)
            matchingDataPerI.append(matchingCount)
            ppmDataPerI.append(ppmCount)
            Chi2DataPerI.append(ChiSquareAttack(stegoBlock))
            ZhangDataPerI.append(ZhangLSBMatching(stegoBlock))
            index += 1
        
        psnrData.append(sum(psnrDataPerI)/len(psnrDataPerI))
        failureData.append(sum(failureDataPerI)/len(failureDataPerI))
        lsbData.append(sum(lsbDataPerI)/len(lsbDataPerI))
        matchingData.append(sum(matchingDataPerI)/len(matchingDataPerI))
        ppmData.append(sum(ppmDataPerI)/len(ppmDataPerI))
        Chi2Data.append(sum(Chi2DataPerI)/len(Chi2DataPerI))
        ZhangData.append(sum(ZhangDataPerI) / len(ZhangDataPerI))
    
        print(f"{datetime.now().strftime("%H:%M:%S")} - {i}% Complete")

    print(f"Total time : {datetime.now() - start}")
    
    #Storing the info in SQL
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CompositeMethodvsEmbedRate
    (
        EmbedPercentage FLOAT NOT NULL,
        PSNR FLOAT NOT NULL,
        NumOfFailures FLOAT NOT NULL,
        NumOfLSB FLOAT NOT NULL,
        NumOfMatching FLOAT NOT NULL,
        NumOfPPM FLOAT NOT NULL,
        BlockSize INT NOT NULL,
        Chi2 FLOAT NOT NULL,
        Zhang FLOAT NOT NULL,
        Threshold FLOAT NOT NULL
    )""")
    conn.commit()
    
    for i in range(len(embedPercentage)):
        cursor.execute("""
        INSERT INTO CompositeMethodvsEmbedRate
        (EmbedPercentage, PSNR, NumOfFailures, NumOfLSB, NumOfMatching, NumOfPPM, BlockSize, Chi2, Zhang, Threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
        (embedPercentage[i], psnrData[i], failureData[i], lsbData[i],matchingData[i],ppmData[i], blockSize[0], Chi2Data[i], ZhangData[i], threshold))
    
    conn.commit()
    conn.close()

def DeriveGraphsFromSQL(settings : dict):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""SELECT * FROM ExistingMethodsvsEmbedRate WHERE EmbedType = 'LSB'""")
    lsbData = cursor.fetchall()
    cursor.execute("""SELECT * FROM ExistingMethodsvsEmbedRate WHERE EmbedType = 'Matching'""")
    matchingData = cursor.fetchall()
    cursor.execute("""SELECT * FROM ExistingMethodsvsEmbedRate WHERE EmbedType = 'PPM'""")
    ppmData = cursor.fetchall()
    
    embedPercentage = [lsbData[i][1] for i in range(len(lsbData))]
    
    #Chi2 data - existing
    plt.figure(figsize=(10, 5))
    plt.plot(embedPercentage, [lsbData[i][3] for i in range(len(lsbData))], label="LSB Replacement")
    plt.plot(embedPercentage, [matchingData[i][3] for i in range(len(matchingData))], label="LSB Matching")
    plt.plot(embedPercentage, [ppmData[i][3] for i in range(len(ppmData))], label="Pixel Pair Matching")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Chi-Square Statistic")
    plt.title(f"Chi Square Statistic vs Embedding Rate For Existing Static Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"Chi2 VS Embedding Rate Existing Methods 1000.png", dpi=600)
    
    #Zhang data - existing
    plt.figure(figsize=(10, 5))
    plt.plot(embedPercentage, [lsbData[i][4] for i in range(len(lsbData))], label="LSB Replacement")
    plt.plot(embedPercentage, [matchingData[i][4] for i in range(len(matchingData))], label="LSB Matching")
    plt.plot(embedPercentage, [ppmData[i][4] for i in range(len(ppmData))], label="Pixel Pair Matching")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Zhang's Analysis Score")
    plt.title(f"Zhang's Analysis Score vs Embedding Rate For Existing Static Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"Zhang VS Embedding Rate Existing Methods 1000.png", dpi=600)
    
    #PSNR data - existing
    plt.figure(figsize=(10, 5))
    plt.plot(embedPercentage, [lsbData[i][2] for i in range(len(lsbData))], label="LSB Replacement")
    plt.plot(embedPercentage, [matchingData[i][2] for i in range(len(matchingData))], label="LSB Matching")
    plt.plot(embedPercentage, [ppmData[i][2] for i in range(len(ppmData))], label="Pixel Pair Matching")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("PSNR value (dB)")
    plt.title(f"PSNR vs Embedding Rate For Existing Static Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"PSNR VS Embedding Rate Existing Methods 1000.png", dpi=600)
    
    #COMPOSITE GRAPHS
    compositeDataSets = []
    for setting in settings:
        cursor.execute("""SELECT * FROM CompositeMethodvsEmbedRate WHERE BlockSize = ? AND Threshold = ?""", settings[setting])
        compositeData = cursor.fetchall()
        #print(compositeData)
        #Because we have no data for 0 embedding we copy the data from the covers so plotting works as intended
        compositeData.insert(0, (0, lsbData[0][2], 0, 0, 0, 0, 0, lsbData[0][3], lsbData[0][4]))
        compositeDataSets.append(compositeData)
        
    #Chi2
    print(f"{len(lsbData)} | {len(matchingData)} | {len(ppmData)} | {len(compositeData)}")

    plt.figure(figsize=(10, 5))
    plt.plot(embedPercentage, [lsbData[i][3] for i in range(len(lsbData))], label="LSB Replacement")
    plt.plot(embedPercentage, [matchingData[i][3] for i in range(len(matchingData))], label="LSB Matching")
    plt.plot(embedPercentage, [ppmData[i][3] for i in range(len(ppmData))], label="Pixel Pair Matching")
    for k in range(len(compositeDataSets)):
        compositeData = compositeDataSets[k]
        plt.plot(embedPercentage, [compositeData[i][7] for i in range(len(compositeData))], label=f"Composite Method {list(settings.keys())[k]}")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Chi-Square Statistic")
    plt.title(f"Chi Square Statistic vs Embedding Rate For All Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"Chi2 VS Embedding Rate All Methods 1000.png", dpi=600)
    
    #Zhang data
    plt.figure(figsize=(10, 5))
    plt.plot(embedPercentage, [lsbData[i][4] for i in range(len(lsbData))], label="LSB Replacement")
    plt.plot(embedPercentage, [matchingData[i][4] for i in range(len(matchingData))], label="LSB Matching")
    plt.plot(embedPercentage, [ppmData[i][4] for i in range(len(ppmData))], label="Pixel Pair Matching")
    for k in range(len(compositeDataSets)):
        compositeData = compositeDataSets[k]
        plt.plot(embedPercentage, [compositeData[i][8] for i in range(len(compositeData))], label=f"Composite Method {list(settings.keys())[k]}")


    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Zhang's Analysis Score")
    plt.title(f"Zhang's Analysis Score vs Embedding Rate For All Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"Zhang VS Embedding Rate All Methods 1000.png", dpi=600)
    
    #PSNR data
    plt.figure(figsize=(10, 5))
    plt.plot(embedPercentage, [lsbData[i][2] for i in range(len(lsbData))], label="LSB Replacement")
    plt.plot(embedPercentage, [matchingData[i][2] for i in range(len(matchingData))], label="LSB Matching")
    plt.plot(embedPercentage, [ppmData[i][2] for i in range(len(ppmData))], label="Pixel Pair Matching")
    for k in range(len(compositeDataSets)):
        compositeData = compositeDataSets[k]
        plt.plot(embedPercentage, [compositeData[i][1] for i in range(len(compositeData))], label=f"Composite Method {list(settings.keys())[k]}")


    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("PSNR value (dB)")
    plt.title(f"PSNR vs Embedding Rate For All Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"PSNR VS Embedding Rate All Methods 1000.png", dpi=600)
    
    #Plotting frequency of blocks
    #We can remove the first data point for 0 because it does nothing useful here
    embedPercentage.pop(0)
    
    for k in range(len(compositeDataSets)):
        compositeData = compositeDataSets[k]
        compositeData.pop(0)
        plt.figure(figsize=(10, 5))
        plt.plot(embedPercentage, [compositeData[i][2] for i in range(len(compositeData))], label="Failure")
        plt.plot(embedPercentage, [compositeData[i][3] for i in range(len(compositeData))], label="LSB Replacement")
        plt.plot(embedPercentage, [compositeData[i][4] for i in range(len(compositeData))], label="LSB Matching")
        plt.plot(embedPercentage, [compositeData[i][5] for i in range(len(compositeData))], label="Pixel Pair Matching")

        plt.xlabel("Embedding Rate (%)")
        plt.ylabel("Block Frequency")
        plt.title(f"Block Frequency vs Embedding Rate For Composite Method {list(settings.keys())[k]} Across First 1000 Images Of BossBase")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"Block Frequency VS Embedding Rate Composite Method {list(settings.keys())[k]} 1000.png", dpi=600)
    
    conn.close()

#SQLStoreCompositeMethodPSNRvEmbedRate(blockSize=(32,32), threshold=3)
settings = {
    "A" : (64,3),
    "B" : (64,1),
    "C" : (64,5),
    "D" : (32,3),
    "E" : (32,1),
    "F" : (32,5)
}

#SQLStoreCompositeMethodPSNRvEmbedRate(blockSize=(32,32), threshold=5)
DeriveGraphsFromSQL(settings)