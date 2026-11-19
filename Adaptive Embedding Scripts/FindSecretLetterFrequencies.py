import matplotlib.pyplot as plt
from Helpers.HelperFunctions import *

"""
This code generates a bar chart showing the relative frequencies of different characters within Lipsum.
This is used when discussing the structured text against random binary and ciphertext.
"""

SETUP_PATHS = GetPaths()
LIPSUM = SETUP_PATHS["Structured Secret"]
GRAPHS = SETUP_PATHS["Graphs"]

with open(LIPSUM, "r") as f:
    lines = f.read()

sum = len(lines)
charDict = dict()

for char in lines:
    #char = char.lower()
    if(char in charDict.keys()):
        charDict[char] += 1
    else:
        charDict[char] = 1

x = []
y = []

keysSorted = list(charDict.keys())
keysSorted.sort()
for idx in keysSorted:
    if(idx == "\n"):
        x.append("\\n")
    elif(idx == " "):
        x.append('" "')
    else:
        x.append(idx)
    y.append(charDict[idx] * 100 / sum)

plt.figure(figsize=(10, 5))

plt.bar(
    x,
    y
)

plt.xlabel("Character", fontsize=12)
plt.ylabel("Relative Frequency / %", fontsize=12)
plt.title("Graph of Letter Distribution Within Secret Structured Text")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPHS, "StructuredSecretLetterDistribution.png"),
    dpi=600
)