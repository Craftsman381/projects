from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import cuda, device, exp, no_grad, load, from_numpy
import json

from model import get_model
from PIL import Image
import numpy as np

BS= 64
RESIZE= 256
ROTATION= 25
CROP_SIZE= 224
MEAN= [0.485, 0.456, 0.406]
STD= [0.229, 0.224, 0.225]

data_dir = './flowers'

train_dir = data_dir + '/train'
val_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def prepare_data():
    train_transformer=transforms.Compose([transforms.RandomRotation(ROTATION),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(CROP_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize(MEAN,STD)])

    val_transformer = transforms.Compose([transforms.Resize(RESIZE),
                                          transforms.CenterCrop(CROP_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize(MEAN,STD)])

    test_transformer = transforms.Compose([transforms.Resize(RESIZE),
                                          transforms.CenterCrop(CROP_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize(MEAN,STD)])

    train_dataset = datasets.ImageFolder(train_dir, train_transformer)
    val_dataset = datasets.ImageFolder(val_dir, val_transformer)
    test_dataset = datasets.ImageFolder(test_dir, test_transformer)

    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=True)
    
    print("[INFO] Data Loaded Successfully!\n")
    
    return train_dataloader, val_dataloader, test_dataloader, train_dataset.class_to_idx


def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(img_path)
    img_transforms = transforms.Compose([transforms.Resize(RESIZE),
                                         transforms.CenterCrop(CROP_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(MEAN,STD)])
    
    return img_transforms(image)    

def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    image = process_image(image_path).numpy()
    image = from_numpy(np.array([image])).float()

    with no_grad():
        pred_log = model.forward(image.cuda())
        
    pred = exp(pred_log).data
    
    return pred.topk(topk)