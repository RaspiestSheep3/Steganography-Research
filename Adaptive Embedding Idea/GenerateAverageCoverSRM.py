import os
import numpy as np
from PIL import Image
from datetime import datetime
from sealwatch.srm import srmq1

BOSSBASE_FOLDER = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Bossbase Dataset 1001-2000"
SAVE_PATH = r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Idea\SRMSetup"

def FlattenSRM(srmDict):
    vectors = []
    for key in sorted(srmDict.keys()):
        vectors.append(np.array(srmDict[key]).flatten())
    return np.concatenate(vectors)

paths = os.listdir(BOSSBASE_FOLDER)

srms = []

index = 1
count = 1000
timesTaken = []

def ConvertSeconds(seconds):
    minutes, secondsRemaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)} min {round(secondsRemaining)} sec"

for path in paths: 
    start = datetime.now()
    cover = Image.open(os.path.join(BOSSBASE_FOLDER,path)).convert('L')
    srm = srmq1.extract(np.array(cover, dtype=np.uint8))

    srms.append(FlattenSRM(srm))
    
    timesTaken.append((datetime.now() - start).total_seconds())
    print(f"\r\033[K{datetime.now().strftime('%H:%M:%S')} - Completed {index} / {count} (ETA = {ConvertSeconds(sum(timesTaken) / len(timesTaken) * (count - index))})", end="", flush=True)
    index += 1
    
srmsNP = np.array(srms)
meanCover = np.mean(srmsNP, axis=0)

np.save(os.path.join(SAVE_PATH, "MeanCoverSRM.npy"), meanCover)