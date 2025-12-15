#Imports
from PIL import Image
from scipy.stats import chi2

#Setup variables
COVER_IMAGE_PATH = r"Steganography Research\Hand.png"
STEGO_IMAGE_PATH = r"Steganography Research\Hand_Stego.png"
SECRET_PATH = r"Steganography Research\TestOneInput.txt"

#Constants - can be changed between runs
IMAGE_DIMENSIONS = (256,256)
CHUNK_OFFSET = (0,0)
CHUNK_SIZE = (32,32)
XOR_VALUE = 213

#Functions
def VisualiseGrid(grid : list):
    gridHeight = len(grid)
    
    for i in range(gridHeight):
        print(grid[i])
        
def GetPairDistribution(grid : list, percentageScaled : bool):
    gridHeight = len(grid)
    gridLength = len(grid[0])
    
    count = [0,0]
    
    for i in range(gridHeight):
        for j in range(gridLength):
            if((grid[i][j] & 1) == 0):
                count[0] += 1
            else:
                count[1] += 1
    
    if(percentageScaled):
        countScaled = [(count[0]/(count[0] + count[1])), (count[1]/(count[0] + count[1]))]
        return countScaled
    else:
        return count

def ChiSquareAttack(grid : list, flattened : bool = False):
    if(not flattened):
        gridFlattened = [i for s in grid for i in s]
    else:
        gridFlattened = grid
    total = 0
    df = 127
    
    for i in range(128):
        gridFlattenedCount = [gridFlattened.count(2*i), gridFlattened.count(2*i + 1)]
        
        expectedCount = (gridFlattenedCount[0] + gridFlattenedCount[1]) / 2
        if(expectedCount == 0):
            df -= 1
            continue
        total += (((gridFlattenedCount[0] - expectedCount) ** 2) / expectedCount) + (((gridFlattenedCount[1] - expectedCount) ** 2) / expectedCount)

    p_value = chi2.sf(total, df)   # survival function = 1 - cdf
    print("chi2 =", total, " df =", df, " p =", p_value)
    return p_value

#Runtime variables
coverImage = Image.open(COVER_IMAGE_PATH).convert("L")
stegoImage = coverImage.copy()

#Test 1 - considering only a 16 * 16 slice, finding its plane
topLeftCover = coverImage.crop((CHUNK_OFFSET[0], CHUNK_OFFSET[1], CHUNK_OFFSET[0] + CHUNK_SIZE[0], CHUNK_OFFSET[1] + CHUNK_SIZE[1])) 

chunkPlaneCover = [[topLeftCover.getpixel((i,j)) for i in range(CHUNK_SIZE[0])] for j in range(CHUNK_SIZE[1])]
chunkPlaneStego = [row[:] for row in chunkPlaneCover]

#print("Cover Display : ")
#VisualiseGrid(chunkPlaneCover)

print(f"Cover Pair Distribution: {GetPairDistribution(chunkPlaneCover, False)}")

print(f"Cover Chi Square Attack: {ChiSquareAttack(chunkPlaneCover)}")

#Obtaining the secret
with open(SECRET_PATH, "r") as fileHandle:
    secretData = [*fileHandle.read().strip()]

#XORing the secret - for this I am going to use a custom string for testing purposes
xorB = [XOR_VALUE] * len(secretData)
xorC = [0 for _ in range(len(secretData))]

for i in range(len(secretData)):
    xorC[i] = ord(secretData[i]) ^ xorB[i]

#Dividing xorC into bit chunks
xorCDivided = [0 for _ in range(len(secretData) * 8)]

for i in range(len(secretData)):
    for j in range(7,-1,-1):
        xorCDivided[i*8 + (7-j)] = (xorC[i] & (2**j)) >> j

#print(xorCDivided) 

#Flipping the chunks
chunkPlaneStegoFlattened = [i for s in chunkPlaneStego for i in s]
for i in range(len(chunkPlaneStegoFlattened)):
    remainder = chunkPlaneStegoFlattened[i] % 2
    if(xorCDivided[i] == 0 and remainder == 1):
        chunkPlaneStegoFlattened[i] = chunkPlaneStegoFlattened[i] - 1
    elif(xorCDivided[i] == 1 and remainder == 0):
        chunkPlaneStegoFlattened[i] = chunkPlaneStegoFlattened[i] + 1

#print(f"Stego Flattened Display : {chunkPlaneStegoFlattened}")
print(f"Chi Square Attack On Stego : {ChiSquareAttack(chunkPlaneStegoFlattened, True)}")