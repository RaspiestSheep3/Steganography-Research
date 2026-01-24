import random
from copy import deepcopy
RANDOM_SEED = 1000

def StandardLSB(block : list, secret : str) -> list:
    """
    **Parameters** : list[list[int]] (A section of a grayscale cover image displayed in 2D array format), str (the secret to be embedded)
    **Returns** : list[list[int]] (A section of a grayscale stego image displayed in 2D array format)
    
    - LSB steganography works by altering the LSB of pixels within an image to hide the data
    - An input string is converted using ASCII into bits, which can then be embedded into the image    
    """
    
    random.seed(RANDOM_SEED)
    
    targetedSections = []
    for secretData in secret:
        #Converting the data into binary
        secretData = format(ord(secretData), '08b')
        
        #print(f"Secret Data : {secretData}")
        
        for bit in secretData:
            square = []
            
            #Checks in place to make sure we don't accidentally double-embed
            while(square == []) or square in targetedSections:
                square = [random.randint(0,len(block) - 1), random.randint(0,len(block[0]) - 1)]
            
            bit = int(bit)
            #Setting the LSB
            if(block[square[0]][square[1]] & 1 != bit):
                if(block[square[0]][square[1]] % 2 ==0):
                    block[square[0]][square[1]] += 1
                else:
                    block[square[0]][square[1]] -= 1
            
            targetedSections.append(square) 
    
    return block        

#*Note - not using Matrix Encoding because I do not fully understand it and can't make it work
def GenerateHammingMatrix(bitsPerBlock):
    n = 2**bitsPerBlock - 1
    
    hammingCodes = [[0 for _ in range(n)] for _ in range(bitsPerBlock)]
    
    for col in range(1, n + 1):
        bits = format(col, f'0{bitsPerBlock}b')

        for row in range(bitsPerBlock):
            hammingCodes[row][col - 1] = int(bits[row])
    
    return hammingCodes

def FindSyndrome(vector, matrix, syndromeSize):
    syndrome = [0] * syndromeSize
    for i in range(syndromeSize):
        s = 0
        for j in range(len(vector)):
            if matrix[i][j] == 1:
                s ^= int(vector[j])
        syndrome[i] = s
    return syndrome

def MatrixEncoding(block, secretRaw):
    hammingMatrix = [
        [1,1,1,0,1,0,0],
        [1,1,0,1,0,1,0],
        [1,0,1,1,0,0,1]
    ]
    
    HT = [
        [1,1,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ]
    
    #print(f"Secret : {secretRaw}")
    
    #Processing the secret into something we can use
    secret = [0 for _ in range(len(secretRaw) * 8)]
    for i in range(len(secretRaw)):
        binary = format(ord(secretRaw[i]), '08b')
        for j in range(8):
            secret[8*i + j] = int(binary[j])
            
    #print(f"New Secret : {secret}")
    
    H = len(block)
    W = len(block[0])
    L = len(secret)
    
    ER = L/(H * W)
    
    if(0 <= ER <= 1):
        N1 = 0
        N2 = 0
        N3 = L/3
    elif(1 < ER <= 1.5):
        N1 = 0
        N2 = L - H * W
        N3 = H*W - (2*L)/3
    elif(1.5 < ER <= 3):
        N1 = (2*L/3) - H * W
        N2 = H * W - L/3
        N3 = 0
    
    Gs = [[] for _ in range(8)]
    
    for i in range(128):
        c = [*format(i, "07b")]
        s = FindSyndrome(c, HT, 3)
        u = s[0] * 4 + s[1] * 2 + s[2]
        Gs[u].append(c)
    
    def Algorithm1(bs, ds):
        u = ds[0] * 4 + ds[1] * 2 + ds[2]
        Gu = Gs[u]
        for set in Gu:
            if(set[0:4] == bs[0:4]):
                return set
    
    #Algorithm 2
    Is = [x for x in range(N1)]
    blockFlattened = [i for s in block for i in s]
    
    dsCounter = 0
    
    newBlockFlattened = deepcopy(blockFlattened)
    
    #Step 2
    for i in Is:
        pi = [blockFlattened[i]]
        piBits = [*format(pi, "08b")]
        
        bs = piBits[1:]
        bDashes = Algorithm1(bs, secret[dsCounter:dsCounter + 3])
        
        newBlockFlattened[i] = piBits[0] * (2**7)
        for k in range(7):
            newBlockFlattened[i] += bDashes[k] * (2 ** (6 - k))
        
        dsCounter += 3
    
    #Step 3
    Is = [i for i in range(N1 + 2*N2 + 1, N1 + 2*N2 + 3*N3 - 1, 3)]
    
    for i in Is:
        pi = [*format(blockFlattened[i], "08b")]
        piPlus1 = [*format(blockFlattened[i + 1], "08b")]
        bs = [pi[4], pi[5], piPlus1[5], pi[6], piPlus1[6], pi[7], piPlus1[7]]
        bDashes = Algorithm1(bs, secret[dsCounter:dsCounter + 3])
        
        newBlockFlattened[i] = pi[0] * (2**7) + pi[1] * (2**6) + pi[2] * (2**5) + pi[3] * (2**4) + bDashes[0] * (2**3) + bDashes[1] * (2**2) + bDashes[3] * (2**1) + bDashes[5]
        newBlockFlattened[i+1] = piPlus1[0] * (2**7) + piPlus1[1] * (2**6) + piPlus1[2] * (2**5) + piPlus1[3] * (2**4) + piPlus1[4] * (2**3) + bDashes[2] * (2**2) + bDashes[4] * (2**1) + bDashes[6]
    
        dsCounter += 3
        
    #Deflattening the block
    newBlock = []
    for i in range(64):
        newBlock.append(newBlockFlattened[64*i:64*i+64])
    
    return newBlock

def PixelPairMatching(block, secret):
    pass

def LSBMatching(block, secretRaw):
    """
    **Parameters** : list[list[int]] (A section of a grayscale cover image displayed in 2D array format), str (the secret to be embedded)
    **Returns** : list[list[int]] (A section of a grayscale stego image displayed in 2D array format)
    
    - LSB Matching works by changing the pixel by +-1 if the LSB is different to the embed LSB
    - This can be stronger than standard LSB because it does not flatten LSB pairs, unlike normal LSB
    """
    
    targetedSquares = []
    
    #print(f"Secret : {secretRaw}")
    
    #Processing the secret into something we can use
    secret = [0 for _ in range(len(secretRaw) * 8)]
    for i in range(len(secretRaw)):
        binary = format(ord(secretRaw[i]), '08b')
        for j in range(8):
            secret[8*i + j] = int(binary[j])
            
    for i in range(len(secret)):
        point = (None, None)
        while(point == (None, None) or (point in targetedSquares)):
            point = (random.randint(0,len(block) - 1), random.randint(0, len(block[0]) - 1))
            pixelRaw = block[point[0]][point[1]]
            
            if(pixelRaw % 2 == secret[i]):
                continue
            
            else:
                if(pixelRaw == 0):
                    pixelRaw += 1
                elif(pixelRaw == 255):
                    pixelRaw -= 1
                else:
                    pixelRaw += -1 if random.randint(0,1) == 0 else 1
            
            block[point[0]][point[1]] = pixelRaw
    
    return block

def WhiteSpaceEncoding(block, secret):
    return block