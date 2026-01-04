from PIL import Image
import random
import math
from scipy.stats import chi2

print("Imports complete")

#Runtime constants
BYTES_PER_SECTION = 32
DEVIATION_COEFFICENTS = {
    "Chi Square Attack" : 10,
    "RS Group Analysis" : 0.01,
    "PSNR" : 1
}
ACCEPTABLE_MAPPING_THRESHOLD = 10
RANDOM_SEED = 1000

class Block():
    blockCol = 0.0
    blockRow = 0.0
    embedded = False
    embedType = None

def SplitIntoBlocks(imagePath, blockSize=32):
    # Load the image and convert to grayscale
    img = Image.open(imagePath).convert("L")
    
    # Get pixel data as a 2D list
    arr = list(img.getdata())
    width, height = img.size
    arr2D = [arr[y * width:(y + 1) * width] for y in range(height)]
    
    # Ensure image dimensions are multiples of blockSize
    if width % blockSize != 0 or height % blockSize != 0:
        raise ValueError("Image dimensions must be multiples of blockSize")
    
    numBlocksY = height // blockSize
    numBlocksX = width // blockSize
    
    # Split into blocks
    blocks = []
    for by in range(numBlocksY):
        rowBlocks = []
        for bx in range(numBlocksX):
            block = []
            for y in range(blockSize):
                blockRow = arr2D[by * blockSize + y][bx * blockSize : bx * blockSize + blockSize]
                block.append(blockRow)
            rowBlocks.append(block)
        blocks.append(rowBlocks)
    
    return blocks

def ChiSquareAttack(block):
    observed = {}
    for row in block:
        for value in row:
            pair = value // 2
            if pair not in observed:
                observed[pair] = [0, 0]
            observed[pair][value % 2] += 1  # 0 = even, 1 = odd

    chiSquare = 0.0
    degreesOfFreedom = 0

    for evenCount, oddCount in observed.values():
        total = evenCount + oddCount
        if total == 0:
            continue

        expected = total / 2.0
        chiSquare += ((evenCount - expected) ** 2) / expected
        chiSquare += ((oddCount - expected) ** 2) / expected
        degreesOfFreedom += 1

    if degreesOfFreedom == 0:
        return 0.0

    # p-value from chi-square CDF
    pValue = float(chi2.sf(chiSquare, degreesOfFreedom))

    return max(0.0, min(100.0, pValue * 100.0)) #If p value > 1, we solve it here

def RSGroupAnalysis(block):
    def discrimination(group):
        return (
            abs(group[0] - group[1]) +
            abs(group[1] - group[2]) +
            abs(group[2] - group[3])
        )

    def flipPlus(x):
        return x ^ 1

    def flipMinus(x):
        return x - 1 if (x & 1) else x + 1

    R_plus = S_plus = 0
    R_minus = S_minus = 0

    # Flatten block
    flat = [v for row in block for v in row]

    # Process groups of 4
    for i in range(0, len(flat), 4):
        group = flat[i:i+4]
        if len(group) < 4:
            continue

        D = discrimination(group)

        groupFp = [flipPlus(x) for x in group]
        groupFm = [flipMinus(x) for x in group]

        Dp = discrimination(groupFp)
        Dm = discrimination(groupFm)

        # Positive flipping
        if Dp > D:
            S_plus += 1
        elif Dp < D:
            R_plus += 1

        # Negative flipping
        if Dm > D:
            S_minus += 1
        elif Dm < D:
            R_minus += 1

    total = R_plus + S_plus + R_minus + S_minus
    if total == 0:
        return 0.0

    # Simple embedding estimate proxy
    embeddingEstimate = abs((S_plus - R_plus) + (S_minus - R_minus)) / total
    return embeddingEstimate * 100.0

def PSNR(coverBlock, block):
    #MSE
    m  = len(coverBlock)
    n = len(coverBlock[0])
    
    MSE = 0
    
    for i in range(m):
        for j in range(n):
            MSE += (coverBlock[i][j] - block[i][j])**2
    
    MSE *= (1/(m*n))
    
    if(MSE == 0):
        return float('inf')
    
    PSNR = 10 * math.log10(255 * 255 / MSE)
    
    return PSNR

def StandardLSB(block, secret):
    random.seed(RANDOM_SEED)
    targetedSections = []
    for secretData in secret:
        secretData = format(ord(secretData), '08b')
        
        #print(f"Secret Data : {secretData}")
        
        for bit in secretData:
            square = []
            while(square == []) or square in targetedSections:
                square = [random.randint(0,len(block) - 1), random.randint(0,len(block[0]) - 1)]
            
            bit = int(bit)
            #Setting the LSB
            if(block[square[0]][square[1]] & 1 != bit):
                if(block[square[0]][square[1]] % 2 ==0):
                    block[square[0]][square[1]] += 1
                else:
                    block[square[0]][square[1]] -= 1
            
            targetedSections.append(square) 
    
    return block        

def GenerateHammingMatrix(bitsPerBlock):
    n = 2**bitsPerBlock - 1
    
    hammingCodes = [[0 for _ in range(n)] for _ in range(bitsPerBlock)]
    
    for col in range(1, n + 1):
        bits = format(col, f'0{bitsPerBlock}b')

        for row in range(bitsPerBlock):
            hammingCodes[row][col - 1] = int(bits[row])
    
    return hammingCodes

def MatrixEncoding(block, secret):
    #Dividing the image into 
    
    return block

def WhiteSpaceEncoding(block, secret):
    return block

GenerateHammingMatrix(4)

random.seed(RANDOM_SEED)

blocks = SplitIntoBlocks("Male.png", blockSize=64)
random.shuffle(blocks)

#NTS for access : tempAccess = blocks[blockRow][blockCol][y][x]

#Loading in the secret
with open("Lipsum.txt", "r") as fileHandle:
    secret = fileHandle.read(1024) #If we can do this we get an embed rate of 12.5%, which is a good start
    secret.replace("\n","")

#Splitting the secret into sections 
secret = [*secret]
secretSplit = [secret[i:i+BYTES_PER_SECTION] for i in range(0,1024,BYTES_PER_SECTION)]

methodsUsed = []

blockCounter = 0
for i in range(len(secretSplit)):
    consideredBlock = blocks[blockCounter // 16][blockCounter % 16]
    
    #Finding the cover mappings of each technique
    coverMappings = {
        "Chi Square Attack" : ChiSquareAttack(consideredBlock),
        "RS Group Analysis" : RSGroupAnalysis(consideredBlock),
    }
    
    print(coverMappings)

    stegoChanges = {
        "LSB" : StandardLSB(consideredBlock, secretSplit[i]),
        "Matrix" : MatrixEncoding(consideredBlock, secretSplit[i]),
        "Whitespace" : WhiteSpaceEncoding(consideredBlock, secretSplit[i])
    }
    
    #Running the tests
    stegoMappings = {
        "LSB" : ChiSquareAttack(stegoChanges["LSB"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] + RSGroupAnalysis(stegoChanges["LSB"]) * DEVIATION_COEFFICENTS["RS Group Analysis"] + PSNR(consideredBlock, stegoChanges["LSB"]) * DEVIATION_COEFFICENTS["PSNR"],
        "Matrix" : ChiSquareAttack(stegoChanges["Matrix"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] + RSGroupAnalysis(stegoChanges["Matrix"]) * DEVIATION_COEFFICENTS["RS Group Analysis"] + PSNR(consideredBlock, stegoChanges["Matrix"]) * DEVIATION_COEFFICENTS["PSNR"],
        "Whitespace" : ChiSquareAttack(stegoChanges["Whitespace"]) * DEVIATION_COEFFICENTS["Chi Square Attack"] + RSGroupAnalysis(stegoChanges["Whitespace"]) * DEVIATION_COEFFICENTS["RS Group Analysis"] + PSNR(consideredBlock, stegoChanges["Whitespace"]) * DEVIATION_COEFFICENTS["PSNR"]
    }
    
    print(f"Cover : {coverMappings}, Stego : {stegoMappings}")
    
    #Comparing mappings
    bestMapping = max(stegoMappings["LSB"], stegoMappings["Matrix"], stegoMappings["Whitespace"])
    if(bestMapping < ACCEPTABLE_MAPPING_THRESHOLD):
        #We have found our best system
        methodsUsed.append([key for key, val in stegoMappings.items() if val == bestMapping][0])
        blocks[blockCounter // 16][blockCounter % 16] = stegoChanges[methodsUsed[-1]]
    else:
        methodsUsed.append("Failure")
    
    blockCounter += 1

print(methodsUsed)