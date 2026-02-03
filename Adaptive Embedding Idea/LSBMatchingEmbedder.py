import os
import random
from PIL import Image
from Helpers.HelperFunctions import GetPaths

EMBED_AMOUNT = 4096 #In B, 25% for 256^2 image
paths = GetPaths()
SAVE_PATH_FOLDER = paths["Created Stegos"]
BOSSBASE_FOLDER = paths["Bossbase Path"]
FULL_DATABASE_OUT = paths["LSB Matching Stegos"]

fullDatabaseMode = input("Full Database Mode? (Y/N) : ") == "Y"

def EmbedIntoImage(cover):
    embed = []

    for i in range(EMBED_AMOUNT * 8):
        embed.append(random.randint(0,1)) #Creating random bits

    usedPoints = []
    for bit in embed:
        point = (None, None)
        while(point == (None, None) or point in usedPoints):
            point = (random.randint(0,255), random.randint(0,255))
        
        value = cover.getpixel(point)
        if(value % 2 != bit):
            if(value ==0):
                value += 1
            elif(value == 255):
                value -= 1
            else:
                value += 1 if random.randint(0,1) == 0 else -1
            cover.putpixel(point, value)
    
    return cover

if(fullDatabaseMode):
    paths = os.listdir(BOSSBASE_FOLDER)
    
    counter = 1
    for path in paths:
        
        if(counter % 100 == 0):
            print(f"Processed {counter} / {len(paths)}")
        
        cover = EmbedIntoImage(Image.open(os.path.join(BOSSBASE_FOLDER, path)))

        cover.save(os.path.join(FULL_DATABASE_OUT, f"LSB Match Stego {counter}.png"))
        counter += 1
else:
    coverPath = input("Cover : ")
    saveName = input("Save name : ") + ".png"
    cover = EmbedIntoImage(Image.open(coverPath))

    cover.save(os.path.join(SAVE_PATH_FOLDER, saveName))