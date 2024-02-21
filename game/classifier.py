import torch
from PIL import Image
import cv2

from model import PalmFistModel, test_tf

class Classifier:
    '''
    Respobsible for classification with the given model.
    '''
    def __init__(self):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.MODEL = PalmFistModel().to(self.DEVICE)
        self.MODEL.load_state_dict(torch.load('data/PalmFistClassificationModel.pth'))
        self.CAP = cv2.VideoCapture(0)

    def pred(self, image):
        self.MODEL.eval()
        with torch.inference_mode():
            tensor_image = test_tf(Image.fromarray(image))
            tensor_image = tensor_image.unsqueeze(dim=0).to(self.DEVICE)
            pred = self.MODEL(tensor_image)
            label_pred = int(pred.round().item())
        return label_pred

    def classify(self):
        ret, frame = self.CAP.read()
        label_pred = self.pred(frame)
        return label_pred