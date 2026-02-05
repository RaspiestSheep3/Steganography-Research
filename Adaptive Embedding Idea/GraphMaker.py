from Helpers.SteganalysisMethods import *
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
from Helpers.TestingScriptFuncVersion import CompositeMethod
import matplotlib.pyplot as plt
from math import ceil
from datetime import datetime
from copy import deepcopy
import sqlite3

print(f"{datetime.now().strftime("%H:%M:%S")} - Imports complete for GraphMaker")

BOSSBASE_FOLDER = GetPaths()["Bossbase Path"]
paths = os.listdir(BOSSBASE_FOLDER)

def GraphExistingMethodsPSNRvEmbedRate(embedPath = "Lipsum.txt", imageSize = (256,256)):
    """
    **Parameters** : str="Lipsum.txt" (filepath with the secret to be embedded), (int,int) = (256,256) (Size of processed images)
    **Returns** : None
    
    - Looks at all 3 chosen embed methods and plots the average PSNR against embed rate for each method for first 1k images of BossBase
    - Displays and saves the graph
    """
    
    print(f"{datetime.now().strftime("%H:%M:%S")} - Starting setup")
    
    embedPercentage = []
    LSBData = []
    LSBMatchingData = []
    PPMData = []
    
    embedData = []
    with open(embedPath, "r") as f:
        embedData = [*f.read()]
    
    coverBlocks = []
    for path in paths:
        block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
        coverBlocks.append(block)
    
    print(f"{datetime.now().strftime("%H:%M:%S")} - Starting increment loop")
    
    start = datetime.now()
    for i in range(2,101,2):
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
        
        for coverBlock in coverBlocks:
            secret = embedData[:secretBytesAmount]
            LSBBlock = StandardLSB(deepcopy(coverBlock), secret)
            LSBMatchingBlock = LSBMatching(deepcopy(coverBlock), secret)
            PPMBlock = PixelPairMatching(deepcopy(block), secret)

            LSBDataPerI.append(PSNR(coverBlock, LSBBlock))
            LSBMatchingDataPerI.append(PSNR(coverBlock, LSBMatchingBlock))
            PPMDataPerI.append(PSNR(coverBlock,PPMBlock))
                    
        LSBData.append(sum(LSBDataPerI)/len(LSBDataPerI))
        LSBMatchingData.append(sum(LSBMatchingDataPerI)/len(LSBMatchingDataPerI))
        PPMData.append(sum(PPMDataPerI)/len(PPMDataPerI))
        
        print(f"{datetime.now().strftime("%H:%M:%S")} - {i}% done")
    
    end = datetime.now()
    print(f"Time taken : {end - start}")
    
    print(f"LSB : {LSBData}")
    print(f"LSBM : {LSBMatchingData}")
    print(f"PPM : {PPMData}")
        
    #Plotting different methods
    plt.figure(figsize=(8, 5))
    plt.plot(embedPercentage, LSBData, label="Standard LSB")
    plt.plot(embedPercentage, LSBMatchingData, label="LSB Matching")
    plt.plot(embedPercentage, PPMData, label="Pixel Pair Matching")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Average PSNR (dB)")
    plt.title("PSNR vs Embedding Rate For Existing Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("PSNR VS Embedding Rate Existing Methods.png", dpi=300)
    #plt.show()

def GraphCompositeMethodPSNRvEmbedRate(imageSize : tuple = (256,256), blockSize : tuple = (64,64)):
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
    
    print(f"{datetime.now().strftime("%H:%M:%S")} - Loop Start")
    start = datetime.now()
    for i in range(1,101,1):
        if(i > 100):
            break
        
        psnrDataPerI = []
        failureDataPerI = []     
        lsbDataPerI = []
        matchingDataPerI = []
        ppmDataPerI = []   

        embedPercentage.append(i)
        
        #Finding amount of bytes to embed
        totalImageBytes = (imageSize[0] * imageSize[1]) / 8
        secretBytesAmount = int(totalImageBytes * (i/100))

        #Finding the bytes per section to attempt
        bytesPerSection = ceil(secretBytesAmount / ((imageSize[0] / blockSize[0]) ** 2))
        index = 0
        for path in paths:
            stegoBlock, (failureCount, lsbCount, matchingCount, ppmCount) = CompositeMethod(embedData[:secretBytesAmount], os.path.join(BOSSBASE_FOLDER, path), acceptableMappingThreshold=3, bytesPerSection=bytesPerSection)

            psnr = PSNR(coverBlocks[index], stegoBlock)
            psnrDataPerI.append(psnr)
            failureDataPerI.append(failureCount)
            lsbDataPerI.append(lsbCount)
            matchingDataPerI.append(matchingCount)
            ppmDataPerI.append(ppmCount)
            index += 1
        
        psnrData.append(sum(psnrDataPerI)/len(psnrDataPerI))
        failureData.append(sum(failureDataPerI)/len(psnrDataPerI))
        lsbData.append(sum(lsbDataPerI)/len(lsbDataPerI))
        matchingData.append(sum(matchingDataPerI)/len(matchingDataPerI))
        ppmData.append(sum(ppmDataPerI)/len(ppmDataPerI))
    
        print(f"{datetime.now().strftime("%H:%M:%S")} - {i}% Complete")

    print(f"Total time : {datetime.now() - start}")
    
    #Plotting different methods
    plt.figure(figsize=(8, 5))
    plt.plot(embedPercentage, psnrData,"#eb3a34", label="PSNR")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Average PSNR (dB)")
    plt.title("PSNR vs Embedding Rate For Composite Method Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("PSNR VS Embedding Rate Composite Method.png", dpi=300)
    
    #Plotting the graph of embedding rates 
    plt.figure(figsize=(9, 5))
    plt.plot(embedPercentage, failureData,"#eb3a34", label="Failure")
    plt.plot(embedPercentage, lsbData,"#0e5e1c", label="Standard LSB")
    plt.plot(embedPercentage, matchingData,"#0e1d5e", label="LSB Matching")
    plt.plot(embedPercentage, ppmData,"#0a98b5", label="Pixel Pair Matching")
    

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("No. of blocks")
    plt.title("Block Type vs Embedding Rate For Composite Method Across First 1000 Images Of BossBase - α=3")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Block Type VS Embedding Rate Composite Method.png", dpi=300)
    
    #plt.show()

def GraphExistingMethodsChi2AndZhangvEmbedRate(imageSize : tuple = (256,256), embedPath = "Lipsum.txt"):
    print(f"{datetime.now().strftime("%H:%M:%S")} - Starting setup")
    
    embedPercentage = []
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
        
        LSBChi2DataPerI = []
        LSBMatchingChi2DataPerI = []
        PPMChi2DataPerI = []
        LSBZhangDataPerI = []
        LSBMatchingZhangDataPerI = []
        PPMZhangDataPerI = []
        
        secret = "".join(embedData[:secretBytesAmount])
        
        for coverBlock in coverBlocks:
            LSBBlock = StandardLSB(deepcopy(coverBlock), secret)
            LSBMatchingBlock = LSBMatching(deepcopy(coverBlock), secret)
            PPMBlock = PixelPairMatching(deepcopy(coverBlock), secret)

            LSBChi2DataPerI.append(ChiSquareAttack(LSBBlock))
            LSBMatchingChi2DataPerI.append(ChiSquareAttack(LSBMatchingBlock))
            PPMChi2DataPerI.append(ChiSquareAttack(PPMBlock))
            
            #LSBZhangDataPerI.append(ZhangLSBMatching(LSBBlock))
            #LSBMatchingZhangDataPerI.append(ZhangLSBMatching(LSBMatchingBlock))
            #PPMZhangDataPerI.append(ZhangLSBMatching(PPMBlock))
                    
        LSBChi2Data.append(sum(LSBChi2DataPerI)/len(LSBChi2DataPerI))
        #LSBZhangData.append(sum(LSBZhangDataPerI)/len(LSBZhangDataPerI))
        LSBMatchingChi2Data.append(sum(LSBMatchingChi2DataPerI)/len(LSBMatchingChi2DataPerI))
        #LSBMatchingZhangData.append(sum(LSBMatchingZhangDataPerI)/len(LSBMatchingZhangDataPerI))
        PPMChi2Data.append(sum(PPMChi2DataPerI)/len(PPMChi2DataPerI))
        #PPMZhangData.append(sum(PPMZhangDataPerI)/len(PPMZhangDataPerI))
        
        print(f"{datetime.now().strftime("%H:%M:%S")} - {i}% done")
        
        if(i == 100):
            print(f"Chi2 max % : {sum(LSBChi2DataPerI)/len(LSBChi2DataPerI)}, {sum(LSBMatchingChi2DataPerI)/len(LSBMatchingChi2DataPerI)}, {sum(PPMChi2DataPerI)/len(PPMChi2DataPerI)}")
            #print(f"Zhang max % : {sum(LSBZhangDataPerI)/len(LSBZhangDataPerI)}, {sum(LSBMatchingZhangDataPerI)/len(LSBMatchingZhangDataPerI)}, {sum(PPMZhangDataPerI)/len(PPMZhangDataPerI)}")
    
    end = datetime.now()
    print(f"Time taken : {end - start}")
        
    #Plotting different methods
    plt.figure(figsize=(10, 5))
    plt.plot(embedPercentage, LSBChi2Data, label="Standard LSB")
    plt.plot(embedPercentage, LSBMatchingChi2Data, label="LSB Matching")
    plt.plot(embedPercentage, PPMChi2Data, label="Pixel Pair Matching")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Chi-Square Statistic")
    plt.title("Chi Square Statistic vs Embedding Rate For Existing Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Chi2 VS Embedding Rate Existing Methods 1k.png", dpi=300)
    
    """plt.figure(figsize=(8, 5))
    plt.plot(embedPercentage, LSBZhangData, label="Standard LSB")
    plt.plot(embedPercentage, LSBMatchingZhangData, label="LSB Matching")
    plt.plot(embedPercentage, PPMZhangData, label="Pixel Pair Matching")

    plt.xlabel("Embedding Rate (%)")
    plt.ylabel("Zhang Matching Score")
    plt.title("Zhang's LSB Matching Analysis Result vs Embedding Rate For Existing Methods Across First 1000 Images Of BossBase")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Zhang VS Embedding Rate Existing Methods 1k.png", dpi=300)"""
    
    #plt.show()

GraphCompositeMethodPSNRvEmbedRate()