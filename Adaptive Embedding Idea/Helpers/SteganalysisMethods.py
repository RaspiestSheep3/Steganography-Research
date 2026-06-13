import math
from scipy.stats import chi2

def ChiSquareAttack(block : list) -> float:
    """
    **Parameters** : list[list[int]] (A section of a grayscale image displayed in 2D array format)
    **Returns** : float (A % between 0% and 100% describing how likely the given block is a stego block or a cover block)
    
    - The Chi Square Test is a statistical test used to find the difference between observed and expected data frequencies
    - In this case, the Chi Square Attack works by analysing the frequencies of pixels {2k, 2k + 1} for 0 <= k <= 127 in a block
        - In a cover image these values are generally quite different
        - Standard LSB steganography tends to equalise these pixel pairs
        - This equalisation is detected by the Chi Square Test, and used to predict whether an image is a stego
    """
    
    #Finding the frequences of each pixel
    observed = {}
    for row in block:
        for value in row:
            pair = value // 2
            if pair not in observed:
                observed[pair] = [0, 0]
            observed[pair][value % 2] += 1  # 0 = even, 1 = odd

    chiSquare = 0
    degreesOfFreedom = 0

    #Comapring the pixel pairs for each value
    for evenCount, oddCount in observed.values():
        total = (evenCount + oddCount) / 2
        if total < 5:
            continue

        chiSquare += ((evenCount - total) ** 2) / total
        chiSquare += ((oddCount - total) ** 2) / total
        degreesOfFreedom += 1

    if degreesOfFreedom == 0:
        return 0

    #pValue = 100*(1 - float(chi2.sf(chiSquare, degreesOfFreedom)))

    return chiSquare

def ZhangLSBMatching(block : list) -> int:
    """
    **Parameters** : list[list[int]] (A section of a grayscale image displayed in 2D array format)
    **Returns** : int (A measurement of the smoothness of the pixel value histogram, where a higher number indicaters more roughness)
    
    - Zhang et al.'s LSB Matching Steganalysis method is specifically designed to detect LSB matching
    - The method works by analysing the frequency histogram of each pixel
        - A local maxima is defined where the value of a point is greater than its neighbouring points, and a local minima is defined where a point is lower than its neighbouring points
        - LSB matching tends to increase the values of local minima and decrease the values of local maxima, smoothing the histogram
    - Zhang's method finds all local extrema of the considered block and calculated the absolute difference between the extremum and its neighbours for each extremum
        - The total difference (D) is the sum of these differences
        - Because LSB matching smooths the histogram, Dcover > Dstego for most images
    - By analysing the D returned the analyst can make an educated guess over whether the image has been altered through LSB matching
    """
    
    #Finding the pixel histogram
    blockIndex = [0 for _ in range(256)]
    for row in block:
        for num in row:
            blockIndex[num] += 1
    
    extrema = []
    
    #Finding all extrema
    for i in range(1, 255):
        if((blockIndex[i] - blockIndex[i-1])*(blockIndex[i]-blockIndex[i+1]) > 0):
            extrema.append(i)
    
    #Calculating the sum of the differences of each extremum
    D = 0
    for extremum in extrema:
        D += abs(2*blockIndex[extremum] - blockIndex[extremum - 1] - blockIndex[extremum + 1])
    
    return D

def SamplePairAnalysis(block : list) -> float:
    P = []
    for i in range(len(block)):
        for j in range(len(block[i]) - 1):
            P.append((block[i][j], block[i][j+1]))
    
    XDash = []
    VDash = []
    WDash = []
    ZDash = []
    
    for pair in P:
        if(pair[1] % 2 == 0 and pair[1] > pair[0]) or (pair[1] % 2 == 1 and pair[1] < pair[0]):
            XDash.append(pair)
        elif(pair[1] % 2 == 0 and pair[1] < pair[0]) or (pair[1] % 2 == 1 and pair[1] > pair[0]):
            if(abs(pair[0] - pair[1]) == 1):
                WDash.append(pair)
            else:
                VDash.append(pair)
        elif(pair[0] == pair[1]):
            ZDash.append(pair)
        else:
            print(f"SPA ERROR : {pair}")
    
    a = 0.5 * (len(WDash) + len(ZDash))
    b = 2 * len(XDash) - len(P)
    c = len(VDash) + len(WDash) - len(XDash)
    
    #Guarding against errors
    if(a == 0 or (b**2 - 4*a*c) < 0):
        return 0
    
    p1 = (-b + math.sqrt(b**2 - 4*a*c)) / (2 * a)
    p2 = (-b - math.sqrt(b**2 - 4*a*c)) / (2 * a)
    
    candidates = [p for p in [p1, p2] if 0 <= p <= 1]
    if not candidates:
        return 0

    return min(candidates)

def PSNR(coverBlock : list, block : list) -> float:
    """
    **Parameters** : list[list[int]] (A section of a grayscale **cover** image displayed in 2D array format), list[list[int]] (A section of a grayscale **stego** image displayed in 2D array format)
    **Returns** : float (A measurement of how similar to two sections are)
    
    - Peak Signal-to-Noise Ratio (PSNR) is a measurement of how similar two images are, with a higher PSNR meaning that the images are more similar
    - PSNR uses MSE, which means that large differences are punished more aggressively than small differences
    - PSNR is also logarithmic, which scales values to within a reasonable frame
    - If the two images are identical MSE = 0 and the PSNR is infinite / undefined
    """
    
    #MSE
    m  = len(coverBlock)
    n = len(coverBlock[0])
    
    MSE = 0
    
    for i in range(m):
        for j in range(n):
            MSE += (coverBlock[i][j] - block[i][j])**2
    
    MSE *= (1/(m*n))
    
    if(MSE == 0):
        return float('inf')
    
    PSNR = 10 * math.log10(255 * 255 / MSE)
    
    return PSNR

def WS(block : list) -> float:
    rSum = []
    for i in range(len(block)):
        for j in range(len(block[0])):
            x = block[i][j]
            xNeighbours = [
                block[i-1][j] if i > 0 else x,
                block[i+1][j] if (i+1) < len(block) else x,
                block[i][j-1] if j > 0 else x,
                block[i][j+1] if (j+1) < len(block[0]) else x
            ]
            
            xHat = sum(xNeighbours) / 4
            rSum.append(x - xHat)
    
    return -sum(rSum)/len(rSum)