from PIL import Image
from scipy.stats import chi2
import math

coverPath = input("Cover : ").strip('"')
stegoPath = input("Stego : ").strip('"')

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

coverBlocks = SplitIntoBlocks(coverPath, 256)[0][0]
stegoBlocks = SplitIntoBlocks(stegoPath, 256)[0][0]
print(f"Cover Chi2  : {ChiSquareAttack(coverBlocks)} | Stego Chi2 : {ChiSquareAttack(stegoBlocks)}")
print(f"Cover SPA   : {SamplePairAnalysis(coverBlocks)} | Stego SPA  : {SamplePairAnalysis(stegoBlocks)}")
print(f"Cover Zhang : {ZhangLSBMatching(coverBlocks)} | Stego Zhang : {ZhangLSBMatching(stegoBlocks)}")
print(f"PSNR : {PSNR(coverBlocks, stegoBlocks)}")