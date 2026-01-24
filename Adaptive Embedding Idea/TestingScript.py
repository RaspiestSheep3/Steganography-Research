from PIL import Image
import random
import math
from copy import deepcopy
from scipy.stats import chi2

print("Imports complete")

#Runtime constants
OUT_SAVE_PATH = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Idea\Created Stegos\Stego2.png"
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
RANDOM_SEED = 1000
BLOCKS_PER_SIDE = 4

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

def ZhangLSBMatching(block):
    blockIndex = [0 for _ in range(256)]
    for row in block:
        for num in row:
            blockIndex[num] += 1
    
    extrema = []
    
    for i in range(1, 255):
        if((blockIndex[i] - blockIndex[i-1])*(blockIndex[i]-blockIndex[i+1]) > 0):
            extrema.append(i)
    
    D = 0
    for extremum in extrema:
        D += abs(2*blockIndex[extremum] - blockIndex[extremum - 1] - blockIndex[extremum + 1])
    
    return D

def SamplePairAnalysis(block):
    P = []
    for i in range(len(block)):
        for j in range(len(block[i]) - 1):
            P.append((block[i][j], block[i][j+1]))
    
    XDash = []
    VDash = []
    WDash = []
    ZDash = []
    
    for pair in P:
        if(pair[1] % 2 == 0 and pair[1] > pair[0]) or (pair[1] % 2 == 1 and pair[1] < pair[0]):
            XDash.append(pair)
        elif(pair[1] % 2 == 0 and pair[1] < pair[0]) or (pair[1] % 2 == 1 and pair[1] > pair[0]):
            if(abs(pair[0] - pair[1]) == 1):
                WDash.append(pair)
            else:
                VDash.append(pair)
        elif(pair[0] == pair[1]):
            ZDash.append(pair)
        else:
            print(f"SPA ERROR : {pair}")
    
    a = 0.5 * (len(WDash) + len(ZDash))
    b = 2 * len(XDash) - len(P)
    c = len(VDash) + len(WDash) - len(XDash)
    try:
        p1 = (-b + math.sqrt(b**2 - 4*a*c)) / (2 * a)
        p2 = (-b - math.sqrt(b**2 - 4*a*c)) / (2 * a)
        
        return p1 if p1 > 0 else p2
    except:
        print(f"Math domain error : disc = {b**2 - 4*a*c}, a = {a}, b = {b}, c = {c}")
        return 0

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

def FindSyndrome(vector, matrix, syndromeSize):
    syndrome = [0] * syndromeSize
    for i in range(syndromeSize):
        s = 0
        for j in range(len(vector)):
            if matrix[i][j] == 1:
                s ^= int(vector[j])
        syndrome[i] = s
    return syndrome

def MatrixEncoding(block, secretRaw):
    hammingMatrix = [
        [1,1,1,0,1,0,0],
        [1,1,0,1,0,1,0],
        [1,0,1,1,0,0,1]
    ]
    
    HT = [
        [1,1,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ]
    
    #print(f"Secret : {secretRaw}")
    
    #Processing the secret into something we can use
    secret = [0 for _ in range(len(secretRaw) * 8)]
    for i in range(len(secretRaw)):
        binary = format(ord(secretRaw[i]), '08b')
        for j in range(8):
            secret[8*i + j] = int(binary[j])
            
    #print(f"New Secret : {secret}")
    
    H = len(block)
    W = len(block[0])
    L = len(secret)
    
    ER = L/(H * W)
    
    if(0 <= ER <= 1):
        N1 = 0
        N2 = 0
        N3 = L/3
    elif(1 < ER <= 1.5):
        N1 = 0
        N2 = L - H * W
        N3 = H*W - (2*L)/3
    elif(1.5 < ER <= 3):
        N1 = (2*L/3) - H * W
        N2 = H * W - L/3
        N3 = 0
    
    Gs = [[] for _ in range(8)]
    
    for i in range(128):
        c = [*format(i, "07b")]
        s = FindSyndrome(c, HT, 3)
        u = s[0] * 4 + s[1] * 2 + s[2]
        Gs[u].append(c)
    
    def Algorithm1(bs, ds):
        u = ds[0] * 4 + ds[1] * 2 + ds[2]
        Gu = Gs[u]
        for set in Gu:
            if(set[0:4] == bs[0:4]):
                return set
    
    #Algorithm 2
    Is = [x for x in range(N1)]
    blockFlattened = [i for s in block for i in s]
    
    dsCounter = 0
    
    newBlockFlattened = deepcopy(blockFlattened)
    
    #Step 2
    for i in Is:
        pi = [blockFlattened[i]]
        piBits = [*format(pi, "08b")]
        
        bs = piBits[1:]
        bDashes = Algorithm1(bs, secret[dsCounter:dsCounter + 3])
        
        newBlockFlattened[i] = piBits[0] * (2**7)
        for k in range(7):
            newBlockFlattened[i] += bDashes[k] * (2 ** (6 - k))
        
        dsCounter += 3
    
    #Step 3
    Is = [i for i in range(N1 + 2*N2 + 1, N1 + 2*N2 + 3*N3 - 1, 3)]
    
    for i in Is:
        pi = [*format(blockFlattened[i], "08b")]
        piPlus1 = [*format(blockFlattened[i + 1], "08b")]
        bs = [pi[4], pi[5], piPlus1[5], pi[6], piPlus1[6], pi[7], piPlus1[7]]
        bDashes = Algorithm1(bs, secret[dsCounter:dsCounter + 3])
        
        newBlockFlattened[i] = pi[0] * (2**7) + pi[1] * (2**6) + pi[2] * (2**5) + pi[3] * (2**4) + bDashes[0] * (2**3) + bDashes[1] * (2**2) + bDashes[3] * (2**1) + bDashes[5]
        newBlockFlattened[i+1] = piPlus1[0] * (2**7) + piPlus1[1] * (2**6) + piPlus1[2] * (2**5) + piPlus1[3] * (2**4) + piPlus1[4] * (2**3) + bDashes[2] * (2**2) + bDashes[4] * (2**1) + bDashes[6]
    
        dsCounter += 3
        
    #Deflattening the block
    newBlock = []
    for i in range(64):
        newBlock.append(newBlockFlattened[64*i:64*i+64])
    
    return newBlock

def PixelPairMatching(block, secret):
    pass

def LSBMatching(block, secretRaw):
    targetedSquares = []
    
    #print(f"Secret : {secretRaw}")
    
    #Processing the secret into something we can use
    secret = [0 for _ in range(len(secretRaw) * 8)]
    for i in range(len(secretRaw)):
        binary = format(ord(secretRaw[i]), '08b')
        for j in range(8):
            secret[8*i + j] = int(binary[j])
            
    for i in range(len(secret)):
        point = (None, None)
        while(point == (None, None) or (point in targetedSquares)):
            point = (random.randint(0,len(block) - 1), random.randint(0, len(block[0]) - 1))
            pixelRaw = block[point[0]][point[1]]
            
            if(pixelRaw % 2 == secret[i]):
                continue
            
            else:
                pixelRaw += -1 if random.randint(0,1) == 0 else 1
            
            block[point[0]][point[1]] = pixelRaw
    
    return block

def WhiteSpaceEncoding(block, secret):
    return block

random.seed(RANDOM_SEED)

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
    
    