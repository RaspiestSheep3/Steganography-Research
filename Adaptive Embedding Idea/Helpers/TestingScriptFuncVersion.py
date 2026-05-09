from math import ceil
from copy import deepcopy
from Helpers.HelperFunctions import SplitIntoBlocks
from Helpers.EmbeddingAlgorithms import StandardLSB, LSBMatching, PixelPairMatching
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, PSNR

print("Imports complete")

#Runtime constants
IMAGE_SIZE = 256
bytesPerSectionDefault = 512 #Currently an embed rate of 25%
#0.17,1.11,0.04

#See Derivation of Coefficients TXT for how I found this values
DEVIATION_COEFFICENTS = {
    "Chi Square Attack" : -1/580.76,
    "PSNR" : 40,
    "Zhang" : 1/10856.838
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
def CompositeMethod(secret : str, imagePath : str, bytesPerSection : int = bytesPerSectionDefault, blocksPerSide : int = blocksPerSideDefault, acceptableMappingThreshold : float = acceptableMappingThresholdDefault, indexBlockMethod : bool = True) -> list[list[int]]:
    
    blocks = SplitIntoBlocks(imagePath, blockSize=IMAGE_SIZE//blocksPerSide)
    
    blockPositionDictOld = dict()

    counter = 0
    for block in blocks:
        blockPositionDictOld[counter] = block
        counter += 1
    
    
    if(indexBlockMethod):
        indexBlock = deepcopy(blocks[0][0])
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
    secretSplit = [secret[i:i+bytesPerSection] for i in range(0,TOTAL_BYTES,bytesPerSection)]

    methodsUsed = []

    blockCounter = 0
    
    sumOfThresholds = 0
    
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
            "LSB" : StandardLSB(deepcopy(consideredBlock), secretSplit[i], indexBlockMethod),
            "Matching" : LSBMatching(deepcopy(consideredBlock), secretSplit[i], indexBlockMethod),
            "PPM" : PixelPairMatching(deepcopy(consideredBlock), secretSplit[i], indexBlockMethod)
        }
        
        #Running the tests
        stegoMappings = {
            "LSB" : (ChiSquareAttack(stegoChanges["LSB"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
            + (ZhangLSBMatching(stegoChanges["LSB"]) - coverMappings["Zhang"]) * DEVIATION_COEFFICENTS["Zhang"] 
            + DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoChanges["LSB"]),
            "Matching" : (ChiSquareAttack(stegoChanges["Matching"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
            + (ZhangLSBMatching(stegoChanges["Matching"]) - coverMappings["Zhang"]) * DEVIATION_COEFFICENTS["Zhang"] 
            + DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoChanges["Matching"]),
            "PPM" : (ChiSquareAttack(stegoChanges["PPM"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
            + (ZhangLSBMatching(stegoChanges["PPM"]) - coverMappings["Zhang"]) * DEVIATION_COEFFICENTS["Zhang"] 
            + DEVIATION_COEFFICENTS["PSNR"] / PSNR(consideredBlock, stegoChanges["PPM"]),
        }
        
        #print(f"Cover : {coverMappings}, Stego : {stegoMappings}")
        
        #Comparing mappings
        bestMapping = min(stegoMappings["LSB"], stegoMappings["Matching"], stegoMappings["PPM"])
        sumOfThresholds += bestMapping
        if(bestMapping < acceptableMappingThreshold):
            #We have found our best system
            methodsUsed.append([key for key, val in stegoMappings.items() if val == bestMapping][0])
            blocks[blockCounter // blocksPerSide][blockCounter % blocksPerSide] = stegoChanges[methodsUsed[-1]]
        else:
            methodsUsed.append("Failure")
        
        blockCounter += 1

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
        methodBinaryRaw = [*methodsUsedDict[methodUsed]]
        for i in range(len(methodBinaryRaw)):
            binaryEmbedForMethodsUsed.append(methodBinaryRaw[i])
    
    #print(f"Binary Methods Used Len : {len(binaryEmbedForMethodsUsed)}\nBlock Count : {len(blocks) * len(blocks[0])}")
    #print(f"Secret Split : {len(secretSplit)}, Blocks Per Side^2 : {blocksPerSide**2}")
    #Embedding into the index block
    if(indexBlockMethod):
        binaryEmbedForMethodsUsedAsciiRepresentation = ""
        for i in range(ceil(len(binaryEmbedForMethodsUsed) / 8)):
            binaryEmbedForMethodsUsedAsciiRepresentation += chr(int("".join(binaryEmbedForMethodsUsed[i*8:i*8 +8]),2))

        indexBlock = StandardLSB(deepcopy(indexBlock), binaryEmbedForMethodsUsedAsciiRepresentation)
        
        blocks[0][0] = indexBlock
    else:
        counter = 0
        
        for blockRow in blocks:
            for block in blockRow:
                try:
                    block[0][0] = (block[0][0] // 2) + int(binaryEmbedForMethodsUsed[counter])
                    block[0][1] = (block[0][1] // 2) + int(binaryEmbedForMethodsUsed[counter + 1])
                    counter += 2
                except:
                    print(f"Counter : {counter}")
                    assert(1==2)
    
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

    return stego, (methodsUsed.count("Failure"), methodsUsed.count("LSB"), methodsUsed.count("Matching"), methodsUsed.count("PPM")), sumOfThresholds