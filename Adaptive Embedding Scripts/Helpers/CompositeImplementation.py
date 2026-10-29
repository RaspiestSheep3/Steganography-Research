import numpy as np
from math import ceil
from Helpers.HelperFunctions import SplitIntoBlocks
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, MSE
from Helpers.EmbeddingAlgorithms import LSBReplacement, LSBMatching, PixelPairMatching

"""
This code provides the implementation of the outlined composite method
"""

#Runtime constants
IMAGE_SIZE = 256
bytesPerSectionDefault = 512 #Currently an embed rate of 25%
#0.17,1.11,0.04

DEVIATION_COEFFICENTS = {
    ("Chi Square Attack", 8)  : np.float32(0.050) / 0.997,
    ("Chi Square Attack", 4)  : np.float32(0.002) / 0.988,
    ("Chi Square Attack", 2)  : np.float32(0.019) / 0.952,

    ("MSE", 8)  : np.float32(0.516) / 0.570,
    ("MSE", 4)  : np.float32(0.450) / 0.547,
    ("MSE", 2)  : np.float32(0.929) / 0.531,
    
    ("Zhang", 8)  : np.float32(0.434) / 1822,
    ("Zhang", 4)  : np.float32(0.548) / 4826,
    ("Zhang", 2)  : np.float32(0.052) / 12095
}
acceptableMappingThresholdDefault = 3
blocksPerSideDefault = 4

class Block():
    blockCol = 0.0
    blockRow = 0.0
    embedded = False
    embedType = None

#Creating a position dictionary before shuffling

def QuickDeepcopy(arr):
    return [row.copy() for row in arr]

#NTS for access : tempAccess = blocks[blockRow][blockCol][y][x]

#Loading in the secret

#Splitting the secret into sections
def CompositeMethod(secret : str, imagePath : str, bytesPerSection : int = bytesPerSectionDefault, blocksPerSide : int = blocksPerSideDefault, acceptableMappingThreshold : float = acceptableMappingThresholdDefault, indexBlockMethod : bool = True, customCoefficients : dict = None, reform : bool = True):
    
    if(customCoefficients != None):
        deviationCoefficients = customCoefficients
    else:
        deviationCoefficients = DEVIATION_COEFFICENTS
    
    chi2Coefficient = deviationCoefficients[("Chi Square Attack", blocksPerSide)]
    zhangCoefficient = deviationCoefficients[("Zhang", blocksPerSide)]
    mseCoefficient = deviationCoefficients[("MSE", blocksPerSide)]
    
    blocks = SplitIntoBlocks(imagePath, blockSize=IMAGE_SIZE//blocksPerSide)
    
    if(indexBlockMethod):
        indexBlock = QuickDeepcopy(blocks[0][0])
        """Index block plan:
        - We have 4 block states - failure (unchanged), LSB replacement, LSB matching, PPM
        - Therefore, we can use 2 bits to encode for each block - this leads to 32 bits being embedded for 64 * 64 on a 256 * 256 image
            - 00 = failure, 01 = replacement, 10 = matching, 11 = PPM
        - LSB replacement is used within this first block 
            - This method cannot be adaptive as the recipient must know what method is used within this block
        - This system minimises the amount of changes that must be made to the index block whilst still conveying all necessary data about the 
        """
    
    TOTAL_BYTES = len(secret)
    secret = [*secret]
    secretSplit = [secret[i:i+bytesPerSection] for i in range(0, TOTAL_BYTES, bytesPerSection)]
    maxAvailableBlocks = (blocksPerSide * blocksPerSide) - 1
    if len(secretSplit) > maxAvailableBlocks:
        secretSplit = secretSplit[:maxAvailableBlocks]

    methodsUsed = []
    
    bestMappings = []
    
    for i in range(len(secretSplit)):
        rowIdx = (i+1) // blocksPerSide
        colIdx = (i+1) % blocksPerSide
        
        consideredBlock = QuickDeepcopy(blocks[rowIdx][colIdx])
        
        #Finding the cover mappings of each technique
        coverMappings = {
            "Chi Square Attack" : ChiSquareAttack(consideredBlock),
            "Zhang" : ZhangLSBMatching(consideredBlock)
        }
        
        #print(len(consideredBlock), len(consideredBlock[0]))
        
        #print(coverMappings)

        stegoChanges = {
            "LSB" : LSBReplacement(QuickDeepcopy(consideredBlock), secretSplit[i], indexBlockMethod),
            "Matching" : LSBMatching(QuickDeepcopy(consideredBlock), secretSplit[i], indexBlockMethod),
            "PPM" : PixelPairMatching(QuickDeepcopy(consideredBlock), secretSplit[i], indexBlockMethod)
        }
        
        #Running the tests
        stegoMappings = {
            #*We take the abs() on the basis that the cover is the purest form of the image - we should always strive to match the cover as a theoretically perfect stego would = the cover
            "LSB" : 
            abs((ChiSquareAttack(stegoChanges["LSB"]) - coverMappings["Chi Square Attack"]) * chi2Coefficient)
            + abs((ZhangLSBMatching(stegoChanges["LSB"]) - coverMappings["Zhang"]) * zhangCoefficient)
            + abs(MSE(consideredBlock, stegoChanges["LSB"]) * mseCoefficient),
            
            "Matching" : 
            abs((ChiSquareAttack(stegoChanges["Matching"]) - coverMappings["Chi Square Attack"]) * chi2Coefficient) 
            + abs((ZhangLSBMatching(stegoChanges["Matching"]) - coverMappings["Zhang"]) * zhangCoefficient)
            + abs(MSE(consideredBlock, stegoChanges["Matching"]) * mseCoefficient),
            
            "PPM" : 
            abs((ChiSquareAttack(stegoChanges["PPM"]) - coverMappings["Chi Square Attack"]) * chi2Coefficient) 
            + abs((ZhangLSBMatching(stegoChanges["PPM"]) - coverMappings["Zhang"]) * zhangCoefficient)
            + abs(MSE(consideredBlock, stegoChanges["PPM"]) * mseCoefficient),
        }
        
        #print(f"Cover : {coverMappings}, Stego : {stegoMappings}")
        
        #Comparing mappings
        bestMapping = min(stegoMappings["LSB"], stegoMappings["Matching"], stegoMappings["PPM"])
        bestMappings.append(bestMapping)
        if(bestMapping < acceptableMappingThreshold):
            bestMethod = min(stegoMappings, key=stegoMappings.get)
            methodsUsed.append(bestMethod)
            blocks[rowIdx][colIdx] = stegoChanges[bestMethod]
        else:
            methodsUsed.append("Failure")

    #print(methodsUsed)

    #Forming the index block from the methods used
    binaryEmbedForMethodsUsed = []
    methodsUsedDict = {
        "Failure" : "00",
        "LSB" : "01",
        "Matching" : "10",
        "PPM" : "11"
    }
    
    for methodUsed in methodsUsed:
        binaryEmbedForMethodsUsed.extend(methodsUsedDict[methodUsed])
    
    #print(f"Binary Methods Used Len : {len(binaryEmbedForMethodsUsed)}\nBlock Count : {len(blocks) * len(blocks[0])}")
    #print(f"Secret Split : {len(secretSplit)}, Blocks Per Side^2 : {blocksPerSide**2}")
    #Embedding into the index block
    if(indexBlockMethod):
        binaryEmbedForMethodsUsedAsciiRepresentation = ""
        for i in range(ceil(len(binaryEmbedForMethodsUsed) / 8)):
            binaryEmbedForMethodsUsedAsciiRepresentation += chr(int("".join(binaryEmbedForMethodsUsed[i*8:i*8 +8]),2))

        indexBlock = LSBReplacement(QuickDeepcopy(indexBlock), binaryEmbedForMethodsUsedAsciiRepresentation)
        
        blocks[0][0] = indexBlock
    else:
        counter = 0
        
        for blockRow in blocks:
            for block in blockRow:
                try:
                    block[0][0] = (block[0][0] // 2) * 2 + int(binaryEmbedForMethodsUsed[counter])
                    block[0][1] = (block[0][1] // 2) * 2 + int(binaryEmbedForMethodsUsed[counter + 1])
                    counter += 2
                except:
                    print(f"Counter : {counter}")
                    assert(1==2)
    
    #Reforming the stego
    if(reform):
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
    else:
        stego = None

    return stego, (methodsUsed.count("Failure"), methodsUsed.count("LSB"), methodsUsed.count("Matching"), methodsUsed.count("PPM")), bestMappings