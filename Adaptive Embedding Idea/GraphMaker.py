from Helpers.SteganalysisMethods import *
from Helpers.HelperFunctions import *
from Helpers.EmbeddingAlgorithms import *
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy

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
    plt.savefig("PSNR VS Embedding Rate Existing Models.png", dpi=300)
    #plt.show()

def GraphCompositeMethodPSNRvEmbedRate():
    """
    
    """
    pass

GraphExistingMethodsPSNRvEmbedRate()