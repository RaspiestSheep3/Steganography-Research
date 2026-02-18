import random
from copy import deepcopy
import matplotlib.pyplot as plt
from Helpers.HelperFunctions import SplitIntoBlocks
from Helpers.EmbeddingAlgorithms import StandardLSB

random.seed(1000)
imageBlocks = SplitIntoBlocks("Flapjack.png", 256)[0][0]
embedData = [chr(random.randint(0, 255)) for _ in range(8192)]
processed = StandardLSB(deepcopy(imageBlocks), "".join(embedData))

coverCount = dict()
stegoCount = dict()

coverDifferences = dict()
stegoDifferences = dict()

for row in imageBlocks:
    for pixel in row:
        if pixel in coverCount.keys():
            coverCount[pixel] += 1
        else:
            coverCount[pixel] = 1

for row in processed:
    for pixel in row:
        if pixel in stegoCount.keys():
            stegoCount[pixel] += 1
        else:
            stegoCount[pixel] = 1
            
            
for i in range(256):
    if not(i in coverCount.keys()):
        coverCount[i] = 0
    if not(i in stegoCount.keys()):
        stegoCount[i] = 0

for i in range(128):
    coverDifferences[i] = abs(coverCount[2*i] - coverCount[2*i + 1])
    stegoDifferences[i] = abs(stegoCount[2*i] - stegoCount[2*i + 1])

indices = list(range(128))
width = 0.4
plt.figure(figsize=(9, 5), dpi=300)

plt.bar(
    [i - width/2 for i in indices],
    coverDifferences.values(),
    width=width,
    label="Cover",
)

plt.bar(
    [i + width/2 for i in indices],
    stegoDifferences.values(),
    width=width,
    label="LSB Replacement Stego",
)
plt.title("Graph of difference within PoVs for Flapjack.png")
plt.xlabel("PoV index k (2k, 2k+1)")
plt.ylabel("|count(2k) − count(2k+1)|")
plt.legend()
plt.tight_layout()
plt.savefig("PoV characterisation for LSB replacement.png")