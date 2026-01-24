import os
from PIL import Image

BOSSBASE_FOLDER  = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Bossbase Dataset 1-1000"
LSB_MATCH_FOLDER = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Idea\LSB Matching Stegos"

differences = []
covers = []
coverPaths = os.listdir(BOSSBASE_FOLDER)
stegoPaths = os.listdir(LSB_MATCH_FOLDER)

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

for i in range(len(coverPaths)):
    
    if((i+1) % 100 == 0):
        print(f"Processed {i+1} / {len(coverPaths)}")
    
    cover = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, coverPaths[i]), 256)[0][0]
    stego = SplitIntoBlocks(os.path.join(LSB_MATCH_FOLDER, stegoPaths[i]), 256)[0][0]
    differences.append(ZhangLSBMatching(cover) - ZhangLSBMatching(stego))
    covers.append(ZhangLSBMatching(cover))

print(f"Differences on average : {sum(differences)/len(differences)}")
print(f"Cover average : {sum(covers)/len(covers)}")