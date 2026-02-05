from copy import deepcopy
from Helpers.HelperFunctions import SplitIntoBlocks
from Helpers.EmbeddingAlgorithms import StandardLSB, LSBMatching, PixelPairMatching
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, PSNR

print("Imports complete")

#Runtime constants
IMAGE_SIZE = 256
bytesPerSectionDefault = 128 #Currently an embed rate of 25%
#0.17,1.11,0.04

#See Derivation of Coefficients TXT for how I found this values
DEVIATION_COEFFICENTS = {
    "Chi Square Attack" : -1/581,
    "PSNR" : 1/28,
    "Zhang" : 1/1285.69
}
acceptableMappingThresholdDefault = 3
blocksPerSideDefault = 4

class Block():
    blockCol = 0.0
    blockRow = 0.0
    embedded = False
    embedType = None

#Creating a position dictionary before shuffling

#NTS for access : tempAccess = blocks[blockRow][blockCol][y][x]

#Loading in the secret

#Splitting the secret into sections
def CompositeMethod(secret : str, imagePath : str, bytesPerSection : int = bytesPerSectionDefault, blocksPerSide : int = blocksPerSideDefault, acceptableMappingThreshold : float = acceptableMappingThresholdDefault) -> list[list[int]]:
    
    blocks = SplitIntoBlocks(imagePath, blockSize=64)
    
    blockPositionDictOld = dict()

    counter = 0
    for block in blocks:
        blockPositionDictOld[counter] = block
        counter += 1
    
    TOTAL_BYTES = len(secret)
    secret = [*secret]
    secretSplit = [secret[i:i+bytesPerSection] for i in range(0,TOTAL_BYTES,bytesPerSection)]

    methodsUsed = []

    blockCounter = 0
    for i in range(len(secretSplit)):
        consideredBlock = blocks[blockCounter // blocksPerSide][blockCounter % blocksPerSide].copy()
        
        #Finding the cover mappings of each technique
        coverMappings = {
            "Chi Square Attack" : ChiSquareAttack(consideredBlock),
            "Zhang" : ZhangLSBMatching(consideredBlock)
        }
        
        #print(len(consideredBlock), len(consideredBlock[0]))
        
        #print(coverMappings)

        stegoChanges = {
            "LSB" : StandardLSB(deepcopy(consideredBlock), secretSplit[i]),
            "Matching" : LSBMatching(deepcopy(consideredBlock), secretSplit[i]),
            "PPM" : PixelPairMatching(deepcopy(consideredBlock), secretSplit[i])
        }
        
        #Running the tests
        stegoMappings = {
            "LSB" : (ChiSquareAttack(stegoChanges["LSB"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
            + (coverMappings["Zhang"] - ZhangLSBMatching(stegoChanges["LSB"])) * DEVIATION_COEFFICENTS["Zhang"] 
            + PSNR(consideredBlock, stegoChanges["LSB"]) * DEVIATION_COEFFICENTS["PSNR"],
            "Matching" : (ChiSquareAttack(stegoChanges["Matching"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
            + (coverMappings["Zhang"] - ZhangLSBMatching(stegoChanges["Matching"])) * DEVIATION_COEFFICENTS["Zhang"] 
            + PSNR(consideredBlock, stegoChanges["Matching"]) * DEVIATION_COEFFICENTS["PSNR"],
            "PPM" : (ChiSquareAttack(stegoChanges["PPM"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
            + (coverMappings["Zhang"] - ZhangLSBMatching(stegoChanges["PPM"])) * DEVIATION_COEFFICENTS["Zhang"] 
            + PSNR(consideredBlock, stegoChanges["PPM"]) * DEVIATION_COEFFICENTS["PSNR"],
        }
        
        #print(f"Cover : {coverMappings}, Stego : {stegoMappings}")
        
        #Comparing mappings
        bestMapping = min(stegoMappings["LSB"], stegoMappings["Matching"], stegoMappings["PPM"])
        if(bestMapping < acceptableMappingThreshold):
            #We have found our best system
            methodsUsed.append([key for key, val in stegoMappings.items() if val == bestMapping][0])
            blocks[blockCounter // blocksPerSide][blockCounter % blocksPerSide] = stegoChanges[methodsUsed[-1]]
        else:
            methodsUsed.append("Failure")
        
        blockCounter += 1

    #print(methodsUsed)

    #Reforming the stego
    stego = [[0 for _ in range(IMAGE_SIZE)] for __ in range(IMAGE_SIZE)]

    counter = 0

    for blockRow in blocks:
        for block in blockRow:
            xOffset = (counter % blocksPerSide) * (IMAGE_SIZE // blocksPerSide)
            yOffset = (counter // blocksPerSide) * (IMAGE_SIZE // blocksPerSide)
            
            for i in range(IMAGE_SIZE // blocksPerSide):
                for j in range(IMAGE_SIZE // blocksPerSide):
                    stego[yOffset + j][xOffset + i] = block[j][i]
            
            counter += 1

    return stego, (methodsUsed.count("Failure"), methodsUsed.count("LSB"), methodsUsed.count("Matching"), methodsUsed.count("PPM"))