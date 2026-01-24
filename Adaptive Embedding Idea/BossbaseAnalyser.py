import os
from Helpers.HelperFunctions import GetPaths, SplitIntoBlocks
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, SamplePairAnalysis

print("Imports complete")

#Constants
BOSSBASE_FOLDER = GetPaths()["Bossbase Path"]
DISPLAY_FORMAT = "{:.5f}"

paths = os.listdir(BOSSBASE_FOLDER)

chi = []
spa = []
zhang = []

counter = 1
for path in paths:
    
    if(counter % 100 == 0):
        print(f"Processed {counter} / {len(paths)}")
    
    block = SplitIntoBlocks(os.path.join(BOSSBASE_FOLDER, path), 256)[0][0]
    chi.append(ChiSquareAttack(block))
    spa.append(SamplePairAnalysis(block))
    zhang.append(ZhangLSBMatching(block))
    counter += 1

print(f"Chi2 avg   : {DISPLAY_FORMAT.format(sum(chi) / len(chi))}, Max Chi2   : {DISPLAY_FORMAT.format(max(chi))}, Min Chi2   : {DISPLAY_FORMAT.format(min(chi))}")
print(f"SPA avg    : {DISPLAY_FORMAT.format(sum(spa) / len(spa))}, Max SPA    : {DISPLAY_FORMAT.format(max(spa))}, Min SPA    : {DISPLAY_FORMAT.format(min(spa))}")
print(f"Zhang avg  : {DISPLAY_FORMAT.format(sum(zhang) / len(zhang))}, Max Zhang  : {DISPLAY_FORMAT.format(max(zhang))}, Min Zhang  : {DISPLAY_FORMAT.format(min(zhang))}")