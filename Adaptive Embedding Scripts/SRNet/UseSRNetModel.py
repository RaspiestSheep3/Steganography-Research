
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
from SRNet.SRNet import Model

class SteganalysisModel:
    def __init__(self, checkpointPath):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialise the SRNet architecture
        self.model = Model().to(self.device)

        # Load trained weights
        checkpoint = torch.load(
            checkpointPath,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.imageMode = "RGB"

        self.transform = transforms.ToTensor()

        self.classNames = {
            0: "Cover",
            1: "Stego"
        }

    def LoadImage(self, imagePath = None, imgObject = None):
        if(imagePath != None):
            with Image.open(imagePath) as image:
                image = image.convert(self.imageMode)
                imageTensor = self.transform(image)
        else:
            imageTensor = self.transform(imgObject.convert("RGB"))

        return imageTensor

    def PredictImage(self, imagePath = None, imgObject = None):
        imageTensor = self.LoadImage(imagePath, imgObject)

        # Add batch dimension: C,H,W -> 1,C,H,W
        imageTensor = imageTensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(imageTensor)

            probabilities = F.softmax(outputs, dim=1)

            confidence, prediction = torch.max(
                probabilities, dim=1
            )

        predictedClass = prediction.item()

        return {
            #"filename": Path(imagePath).name,
            "prediction": self.classNames[predictedClass],
            "confidence": confidence.item(),
            "coverProbability": probabilities[0, 0].item(),
            "stegoProbability": probabilities[0, 1].item()
        }

    def PredictImages(self, imagePaths = None, imgObjects = None):
        results = []

        if(imagePaths != None):
            for imagePath in imagePaths:
                results.append(self.PredictImage(imagePath = imagePath))

        else:
            for imgObject in imgObjects:
                results.append(self.PredictImage(imgObject = imgObject))

        return results
