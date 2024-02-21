import torch
import torch.nn as nn
from torchvision import transforms as tf
import cv2
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

test_tf = tf.Compose([
    tf.Resize((224, 224)),
    tf.ToTensor(),
])

class PalmFistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*32, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.classifier(self.conv_block(x))

model = PalmFistModel().to(DEVICE)
model.load_state_dict(torch.load('PalmFistClassificationModel.pth'))

classes = ['Fist', 'Palm']
def pred(model, tf, image):
    model.eval()
    with torch.inference_mode():
        tensor_image = tf(Image.fromarray(image))
        tensor_image = tensor_image.unsqueeze(dim=0).to(DEVICE)
        pred = model(tensor_image)
        label_pred = int(pred.round().item())
        np_image = cv2.putText(image.copy(), classes[label_pred], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return np_image

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    img = pred(model, test_tf, frame)
    cv2.imshow('Image', img)
    if cv2.waitKey(22) == ord('q'):
        break

cv2.destroyAllWindows()