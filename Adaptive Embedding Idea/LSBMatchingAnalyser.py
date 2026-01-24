import os
from Helpers.HelperFunctions import GetPaths, SplitIntoBlocks
from Helpers.SteganalysisMethods import ZhangLSBMatching

paths = GetPaths()
BOSSBASE_FOLDER  = paths["Bossbase Folder"]
LSB_MATCH_FOLDER = paths["LSB Matching Stegos"]

differences = []
covers = []
coverPaths = os.listdir(BOSSBASE_FOLDER)
stegoPaths = os.listdir(LSB_MATCH_FOLDER)

for i in range(len(coverPaths)):
    
    if((i+1) % 100 == 0):
        print(f"Processed {i+1} / {len(coverPaths)}")
    
    cover = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, coverPaths[i]), 256)[0][0]
    stego = SplitIntoBlocks(os.path.join(LSB_MATCH_FOLDER, stegoPaths[i]), 256)[0][0]
    differences.append(ZhangLSBMatching(cover) - ZhangLSBMatching(stego))
    covers.append(ZhangLSBMatching(cover))

print(f"Differences on average : {sum(differences)/len(differences)}")
print(f"Cover average : {sum(covers)/len(covers)}")