from PIL import Image
from SRNet import UseSRNetModel

testModel =UseSRNetModel.SteganalysisModel(
    r"C:\Users\iniga\OneDrive\Programming\Steganography Research\Adaptive Embedding Scripts\SRMSetup\SRNets-E Models\LSBM Model.pt"
)

res1 = testModel.PredictImage(
    imgObject = Image.open(
        r"C:\Users\iniga\Datasets\Steganography Research Paper\Model Data\Training\Existing LSBM\1.png"
    )
)

res2 = testModel.PredictImages(
    imgObjects= [
        Image.open(r"C:\Users\iniga\Datasets\Steganography Research Paper\Model Data\Training\Existing LSBM\2.png"),
        Image.open(r"C:\Users\iniga\Datasets\Steganography Research Paper\Model Data\Training\cover\3.png"),
        Image.open(r"C:\Users\iniga\Datasets\Steganography Research Paper\Model Data\Training\cover\4.png")
    ]
)

print(f"Result 1: {res1}")

print(f"Result 2 : {res2}")