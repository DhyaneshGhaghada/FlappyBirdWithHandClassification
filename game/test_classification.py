import torch
import cv2
from PIL import Image
from model import PalmFistModel, test_tf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL = PalmFistModel().to(DEVICE)
MODEL.load_state_dict(torch.load('data/PalmFistClassificationModel.pth'))

classes = ['Fist', 'Palm']
def pred(model, tf, image):
    '''
    this function is responsible for classifing and returning the webcam capture.
    '''
    model.eval()
    with torch.inference_mode():
        tensor_image = tf(Image.fromarray(image))
        tensor_image = tensor_image.unsqueeze(dim=0).to(DEVICE)
        pred = model(tensor_image)
        label_pred = int(pred.round().item())
        np_image = cv2.putText(image.copy(), classes[label_pred], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    return np_image

def test():
    '''
    this function is responsible for testing the classification model.
    '''
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img = pred(MODEL, test_tf, frame)
        cv2.imshow('Testing Model Classification Before Game.', img)
        if cv2.waitKey(22) == ord('q'):
            break

    cv2.destroyAllWindows()