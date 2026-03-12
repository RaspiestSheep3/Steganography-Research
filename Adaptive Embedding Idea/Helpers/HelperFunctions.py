import os
import json
from PIL import Image

def SplitIntoBlocks(imagePath, blockSize=32) -> list:
    """
    **Parameters** : StrOrBytesPath | IO[bytes] (path to the image being loaded in), int=32 (size of the blocks, 32 * 32 by default) 
    **Returns** : list[list[list[list[int]]]] (A measurement of how similar to two sections are)
    
    - Takes in an image path and loads it in, converts it into grayscale and creates a 2d array of blocks
    - Each block is itself a 2d array of ints, with each int representing a single pixel's value
    - In order to access a specific pixel in a specific block, the method is blocks[blockRow][blockCol][y][x]
    """
    
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

def GetPaths():
    """
    **Parameters** : None
    **Returns** : dict (Dictionary of paths)
    
    - Loads in paths from "Paths.json" for use in scripting
    - Paths.json should be in the same folder as the script calling this function
    """
    pathsJSONPath = os.path.join(os.getcwd(), "PathsPC.json")
    with open(pathsJSONPath, "r") as f:
        pathsData = json.load(f)
    return pathsData