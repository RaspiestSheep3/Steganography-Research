from scipy.stats import chi2
import math
import os
from PIL import Image

print("Imports complete")

#Constants
BOSSBASE_FOLDER = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Bossbase Dataset 1-1000"
DISPLAY_FORMAT = "{:.5f}"

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
    
    p1 = (-b + math.sqrt(b**2 - 4*a*c)) / (2 * a)
    p2 = (-b - math.sqrt(b**2 - 4*a*c)) / (2 * a)
    
    return p1 if p1 > 0 else p2

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

paths = os.listdir(BOSSBASE_FOLDER)

chi = []
spa = []

counter = 1
for path in paths:
    
    if(counter % 100 == 0):
        print(f"Processed {counter} / {len(paths)}")
    
    block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
    chi.append(ChiSquareAttack(block))
    spa.append(SamplePairAnalysis(block))
    counter += 1

print(f"Chi2 avg : {DISPLAY_FORMAT.format(sum(chi) / len(chi))}, Max Chi2 : {DISPLAY_FORMAT.format(max(chi))}, Min Chi2 : {DISPLAY_FORMAT.format(min(chi))}")
print(f"SPA avg  : {DISPLAY_FORMAT.format(sum(spa) / len(spa))}, Max SPA  : {DISPLAY_FORMAT.format(max(spa))}, Min SPA  : {DISPLAY_FORMAT.format(min(spa))}")