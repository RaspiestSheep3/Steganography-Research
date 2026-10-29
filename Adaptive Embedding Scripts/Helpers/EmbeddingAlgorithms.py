import random

"""
This code provides the existing embedding algorithms (LSBR, LSBM and PPM) used for the composite method and for comparisons
"""

def GetSecretBits(secretRaw : str):
    secretBits = []
    for char in secretRaw:
        binaryStr = format(ord(char), "08b")
        secretBits.extend(int(bit) for bit in binaryStr)
    return secretBits

def LSBReplacement(block : list, secret : str, indexBlockMethod : bool = True) -> list:
    """
    **Parameters** : list[list[int]] (A section of a grayscale cover image displayed in 2D array format), str (the secret to be embedded)
    **Returns** : list[list[int]] (A section of a grayscale stego image displayed in 2D array format)
    
    - LSB steganography works by altering the LSB of pixels within an image to hide the data
    - An input string is converted using ASCII into bits, which can then be embedded into the image    
    """
    
    freesections = [(a,b) for a in range(len(block)) for b in range(len(block[0]))]
    if(not indexBlockMethod):
       freesections = freesections[2:]
    
    random.shuffle(freesections)
    
    #print(f"Secret : {secret} | Len : {len(secret)} | {len(freesections)}")
    secretBits = GetSecretBits(secret)
    
    if(len(secretBits) > len(freesections)):
        secretBits = secretBits[:len(freesections)]
    
    pointer = 0
    for bit in secretBits:
        square = freesections[pointer]
        pointer += 1
        
        pixel = block[square[0]][square[1]]
        if(pixel & 1 != bit):
            if(pixel % 2 == 0):
                block[square[0]][square[1]] = pixel + 1
            else:
                block[square[0]][square[1]] = pixel - 1
    
    return block        

def PixelPairMatching(block : list, secretRaw : str, neighbourhoodSize : int = 1, indexBlockMethod : bool = True) -> list:
    """
    **Parameters** : list[list[int]] (A section of a grayscale cover image displayed in 2D array format), str (the secret to be embedded), int = 1 (base neighbourhood set size)
    **Returns** : list[list[int]] (A section of a grayscale stego image displayed in 2D array format)
    
    - PPM works by considering a pair of pixels and carrying out a function MOD 2 to find their associated binary value
    - If the binary value is not the same as the secret bit, the pair is replaced with a neighbouring pair in the numerical domain with the correct associated binary value    
    """
    
    def PairValueFunction(x : int, y : int) -> int:
        return (x+y) % 2
    
    def ClampNum(n : int, minInclusive : int = 0, maxInclusive : int = 255):
        return max(minInclusive, min(n, maxInclusive))
    
    #Saving every possible pixel pair - pairs are horizontal in this case
    possiblePairs = []
    for i in range(len(block)):
        for j in range(len(block[0]) - 1):
            possiblePairs.append((block[i][j], block[i][j + 1], i, j))
    
    if(not indexBlockMethod):
        possiblePairs = possiblePairs[2:]
    
    random.shuffle(possiblePairs)
    
    #Converting the secret into binary
    secretBits = secretBits = GetSecretBits(secretRaw)
    
    if(len(secretBits) > len(possiblePairs)):
        secretBits = secretBits[:len(possiblePairs)]
    
    pointer = 0
    for bit in secretBits:
        chosenPair = possiblePairs[pointer]
        pointer += 1
        
        fVal = PairValueFunction(chosenPair[0],chosenPair[1])
        if(fVal != bit):
            options = []
            
            while(len(options) == 0):
                for dx in range(-neighbourhoodSize, neighbourhoodSize + 1):
                    for dy in range(-neighbourhoodSize, neighbourhoodSize + 1):
                        if(PairValueFunction(ClampNum(chosenPair[0] + dx), ClampNum(chosenPair[1] + dy)) == bit):
                            options.append((ClampNum(chosenPair[0] + dx), ClampNum(chosenPair[1] + dy)))

                if(len(options) > 0):
                    chosenOption = random.choice(options)
                    block[chosenPair[2]][chosenPair[3]] = chosenOption[0]
                    block[chosenPair[2]][chosenPair[3] + 1] = chosenOption[1]
                else:
                    #print(f"Incrementing neighbourhood size from {neighbourhoodSize} to {neighbourhoodSize + 1}")
                    neighbourhoodSize += 1
        
    return block
     
def LSBMatching(block : list, secretRaw : str, indexBlockMethod : bool = True) -> list:
    """
    **Parameters** : list[list[int]] (A section of a grayscale cover image displayed in 2D array format), str (the secret to be embedded)
    **Returns** : list[list[int]] (A section of a grayscale stego image displayed in 2D array format)
    
    - LSB Matching works by changing the pixel by +-1 if the LSB is different to the embed LSB
    - This can be stronger than standard LSB because it does not flatten LSB pairs, unlike normal LSB
    """
    
    freesections = [(a,b) for a in range(len(block)) for b in range(len(block[0]))]
    if(not indexBlockMethod):
        freesections = freesections[2:]
    
    random.shuffle(freesections)
    #print(f"Secret : {secretRaw}")
    
    #Processing the secret into something we can use
    secretBits = GetSecretBits(secretRaw)

    if(len(secretBits) > len(freesections)):
        secretBits = secretBits[:len(freesections)]

    pointer = 0
    for bit in secretBits:
        point = freesections[pointer]
        pointer += 1
        
        pixelRaw = block[point[0]][point[1]]
        
        if(pixelRaw % 2 == bit):
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