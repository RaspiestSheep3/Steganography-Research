import sys
import os

currentDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.abspath(os.path.join(currentDir, ".."))
sys.path.append(rootDir)

from PIL import Image
from copy import deepcopy
from datetime import datetime
from Helpers.HelperFunctions import SplitIntoBlocks
from Helpers.EmbeddingAlgorithms import StandardLSB, LSBMatching, PixelPairMatching

"""
This code generates existing method stegos for each cover image in Set A.
These stegos are then used to train SRM models.
"""

coverPathBase = input("Cover path : ")
lsbrOutput = input("LSBR Output : ")
lsbmOutput = input("LSBM Output : ")
ppmOutput = input("PPM Output : ")
embedPath = input("Embed Path : ")

def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return f"{int(hour)}h {int(min)} min {round(sec)} sec"

def SaveImage(blocks, path):
    stego = Image.new("L", (256,256))
    
    counter = 0
    for blockRow in blocks:
        for block in blockRow:
            xOffset = (counter % 1) * (256 // 1)
            yOffset = (counter // 1) * (256 // 1)
            
            for i in range(256 // 1):
                for j in range(256 // 1):
                    stego.putpixel((xOffset + i, yOffset + j), block[j][i])
            
            counter += 1
    
    stego.save(path)

timesTaken = []

with open(embedPath, "r") as f:
    embedData = [*f.read()]

for i in range(1, 1001):
    
    iterationStartTime = datetime.now()
    
    coverPath = os.path.join(coverPathBase, f"{i}.png")
    imgBlockForm = SplitIntoBlocks(coverPath, 256)[0][0]
    
    for j in range(2, 12, 2):
        bpp = j / 10
        numBytes = round(256 * 256 * bpp / 8)
        
        secret = "".join(embedData[:numBytes])
        
        lsbrBlockForm = StandardLSB(deepcopy(imgBlockForm), secret)
        lsbmBlockForm = LSBMatching(deepcopy(imgBlockForm), secret)
        ppmBlockForm = PixelPairMatching(deepcopy(imgBlockForm), secret)
    
        SaveImage([[lsbrBlockForm]], os.path.join(lsbrOutput, f"{i}-{bpp}.png"))
        SaveImage([[lsbmBlockForm]], os.path.join(lsbmOutput, f"{i}-{bpp}.png"))
        SaveImage([[ppmBlockForm]], os.path.join(ppmOutput, f"{i}-{bpp}.png"))
    
    iterationEndTime = datetime.now()
    
    timesTaken.append((iterationEndTime - iterationStartTime).total_seconds())
    
    if((i-1) % 10 == 0):
        print(f"{datetime.now().strftime("%H:%M:%S")} - {(i -1) / 10}% complete")
        print(f"Estimated time remaining : {convert(sum(timesTaken) / len(timesTaken) * (1001 - i))}")
    
    