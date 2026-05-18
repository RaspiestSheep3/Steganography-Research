#Goal of this script is to rederive coefficients for c and z using 64 * 64 blocks

from Helpers.EmbeddingAlgorithms import StandardLSB, LSBMatching, PixelPairMatching
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching
from Helpers.HelperFunctions import GetPaths, SplitIntoBlocks
import os
from copy import deepcopy

BOSSBASE_FOLDER = GetPaths()["Bossbase Path"]
paths = os.listdir(BOSSBASE_FOLDER)
embedPath = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Idea\Lipsum.txt"

blockSize = 16

embedData = []
with open(embedPath, "r") as f:
    embedData = [*f.read()]

coverBlocks = []
for path in paths:
    blocks = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), blockSize)
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            coverBlocks.append(blocks[i][j])
    
    
coverBlocksChi2 = []
coverBlocksZhang = []

LSBRBlocksChi2 = []
LSBRBlocksZhang = []

LSBMBlocksChi2 = []
LSBMBlocksZhang = []

PPMBlocksChi2 = []
PPMBlocksZhang = []

counter = 1
print("Start")
secret = "".join(embedData[:int((blockSize * blockSize)/8)]) #100% embed rate


for coverBlock in coverBlocks:
    if((counter * 100 / len(coverBlocks)) == int(counter * 100/len(coverBlocks))):
        print(f"{counter * 100 / len(coverBlocks)}%")
    
    coverBlocksChi2.append(ChiSquareAttack(coverBlock))
    coverBlocksZhang.append(ZhangLSBMatching(coverBlock))
    
    LSBR = StandardLSB(deepcopy(coverBlock), secret,True)
    LSBM = LSBMatching(deepcopy(coverBlock), secret, True)
    PPM = PixelPairMatching(deepcopy(coverBlock), secret, 1, True)
    
    LSBRBlocksChi2.append(ChiSquareAttack(LSBR))
    LSBRBlocksZhang.append(ZhangLSBMatching(LSBR))
    
    LSBMBlocksChi2.append(ChiSquareAttack(LSBM))
    LSBMBlocksZhang.append(ZhangLSBMatching(LSBM))
    
    PPMBlocksChi2.append(ChiSquareAttack(PPM))
    PPMBlocksZhang.append(ZhangLSBMatching(PPM))
    
    counter += 1
    
coverChi2AVG = sum(coverBlocksChi2) / len(coverBlocksChi2)
coverZhangAVG = sum(coverBlocksZhang) / len(coverBlocksZhang)

lsbrChi2AVG = sum(LSBRBlocksChi2) / len(LSBRBlocksChi2)
lsbrZhangAVG = sum(LSBRBlocksZhang) / len(LSBRBlocksZhang)

lsbmChi2AVG = sum(LSBMBlocksChi2) / len(LSBMBlocksChi2)
lsbmZhangAVG = sum(LSBMBlocksZhang) / len(LSBMBlocksZhang)

ppmChi2AVG = sum(PPMBlocksChi2) / len(PPMBlocksChi2)
ppmZhangAVG = sum(PPMBlocksZhang) / len(PPMBlocksZhang)

print(f"Chi2  Cover:  {coverChi2AVG}")
print(f"Zhang Cover:  {coverZhangAVG}")
print(f"Chi2  LSBR :  {lsbrChi2AVG}")
print(f"Zhang LSBR :  {lsbrZhangAVG}")
print(f"Chi2  LSBM :  {lsbmChi2AVG}")
print(f"Zhang LSBM :  {lsbmZhangAVG}")
print(f"Chi2  PPM  :  {ppmChi2AVG}")
print(f"Zhang PPM  :  {ppmZhangAVG}")

counter = 1