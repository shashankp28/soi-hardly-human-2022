import torch
from PIL import Image
from torch.autograd import Variable
from ..models.transforms import transform
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_predict(model, img_path):
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor).to(device)
    output = model(input_img).cpu()
    output = output.data.numpy().tolist()
    return output

def create_data(model_path):
    model = torch.load(model_path)
    wd = "../data/train"
    train = pd.read_csv(wd+"/"+"train.csv")
