from PIL import Image
from copy import deepcopy
from Helpers.HelperFunctions import GetPaths, SplitIntoBlocks
from Helpers.EmbeddingAlgorithms import StandardLSB, LSBMatching, WhiteSpaceEncoding
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, SamplePairAnalysis, PSNR

print("Imports complete")

#Runtime constants
OUT_SAVE_PATH = GetPaths()["Created Stegos"] + "\\Stego2.png"
TOTAL_BYTES = 2048
IMAGE_SIZE = 256
BYTES_PER_SECTION = 128 #Currently an embed rate of 25%
#0.17,1.11,0.04
DEVIATION_COEFFICENTS = {
    "Chi Square Attack" : 0.17,
    "Sample Pair Analysis" : 1.11,
    "PSNR" : 0.04
}
ACCEPTABLE_MAPPING_THRESHOLD = 4
BLOCKS_PER_SIDE = 4

class Block():
    blockCol = 0.0
    blockRow = 0.0
    embedded = False
    embedType = None

blocks = SplitIntoBlocks("Flapjack.png", blockSize=64)

#Creating a position dictionary before shuffling
blockPositionDictOld = dict()

counter = 0
for block in blocks:
    blockPositionDictOld[counter] = block
    counter += 1

#NTS for access : tempAccess = blocks[blockRow][blockCol][y][x]

#Loading in the secret
with open("Lipsum.txt", "r") as fileHandle:
    secret = fileHandle.read(TOTAL_BYTES) #If we can do this we get an embed rate of 12.5%, which is a good start
    secret.replace("\n","")

#Splitting the secret into sections 
secret = [*secret]
secretSplit = [secret[i:i+BYTES_PER_SECTION] for i in range(0,TOTAL_BYTES,BYTES_PER_SECTION)]

methodsUsed = []

blockCounter = 0
for i in range(len(secretSplit)):
    consideredBlock = blocks[blockCounter // BLOCKS_PER_SIDE][blockCounter % BLOCKS_PER_SIDE].copy()
    
    #Finding the cover mappings of each technique
    coverMappings = {
        "Chi Square Attack" : ChiSquareAttack(consideredBlock),
        "Sample Pair Analysis" : SamplePairAnalysis(consideredBlock),
    }
    
    #print(len(consideredBlock), len(consideredBlock[0]))
    
    #print(coverMappings)

    stegoChanges = {
        "LSB" : StandardLSB(deepcopy(consideredBlock), secretSplit[i]),
        "Matching" : LSBMatching(deepcopy(consideredBlock), secretSplit[i]),
        "Whitespace" : WhiteSpaceEncoding(deepcopy(consideredBlock), secretSplit[i])
    }
    
    #Running the tests
    stegoMappings = {
        "LSB" : (ChiSquareAttack(stegoChanges["LSB"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
        + SamplePairAnalysis(stegoChanges["LSB"]) * DEVIATION_COEFFICENTS["Sample Pair Analysis"] 
        + PSNR(consideredBlock, stegoChanges["LSB"]) * DEVIATION_COEFFICENTS["PSNR"],
        #"Matrix" : ChiSquareAttack(stegoChanges["Matrix"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] + SamplePairAnalysis(stegoChanges["Matrix"]) * DEVIATION_COEFFICENTS["Sample Pair Analysis"] + PSNR(consideredBlock, stegoChanges["Matrix"]) * DEVIATION_COEFFICENTS["PSNR"],
        #"Whitespace" : ChiSquareAttack(stegoChanges["Whitespace"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] + SamplePairAnalysis(stegoChanges["Whitespace"]) * DEVIATION_COEFFICENTS["Sample Pair Analysis"] + PSNR(consideredBlock, stegoChanges["Whitespace"]) * DEVIATION_COEFFICENTS["PSNR"]
        "Matching" : (ChiSquareAttack(stegoChanges["Matching"]) - coverMappings["Chi Square Attack"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] 
        + SamplePairAnalysis(stegoChanges["Matching"]) * DEVIATION_COEFFICENTS["Sample Pair Analysis"] 
        + PSNR(consideredBlock, stegoChanges["Matching"]) * DEVIATION_COEFFICENTS["PSNR"],
        "Whitespace" : 10000000
    }
    
    #print(f"Cover : {coverMappings}, Stego : {stegoMappings}")
    
    #Comparing mappings
    bestMapping = min(stegoMappings["LSB"], stegoMappings["Matching"], stegoMappings["Whitespace"])
    if(bestMapping < ACCEPTABLE_MAPPING_THRESHOLD):
        #We have found our best system
        methodsUsed.append([key for key, val in stegoMappings.items() if val == bestMapping][0])
        blocks[blockCounter // BLOCKS_PER_SIDE][blockCounter % BLOCKS_PER_SIDE] = stegoChanges[methodsUsed[-1]]
    else:
        methodsUsed.append("Failure")
    
    blockCounter += 1

print(methodsUsed)

#Reforming the stego
stego = Image.new("L", (256,256))

counter = 0

for blockRow in blocks:
    for block in blockRow:
        xOffset = (counter % BLOCKS_PER_SIDE) * (IMAGE_SIZE // BLOCKS_PER_SIDE)
        yOffset = (counter // BLOCKS_PER_SIDE) * (IMAGE_SIZE // BLOCKS_PER_SIDE)
        
        for i in range(IMAGE_SIZE // BLOCKS_PER_SIDE):
            for j in range(IMAGE_SIZE // BLOCKS_PER_SIDE):
                stego.putpixel((xOffset + i, yOffset + j), block[j][i])
        
        counter += 1

stego.save(OUT_SAVE_PATH)