import torchvision.models as models
from torch.nn import Sequential, NLLLoss, Linear, ReLU, Dropout, LogSoftmax
from torch import mean, optim, exp, no_grad, save, device, cuda, FloatTensor, load, from_numpy
import json

labels_path = 'cat_to_name.json'

def get_model(architecture= 'vgg16', dropout= 0.025, hidden_units= 1024, lr= 0.0015, gpu= True):
    
    logits = len(get_labels())  
    
    print(f"[INFO] Architecture: {architecture}")
    print(f"[INFO] Dropout: {dropout}")
    print(f"[INFO] Hidden Units: {hidden_units}")
    print(f"[INFO] Learning Rate: {lr}")
    print(f"[INFO] Logits: {logits}")
    
    device = set_device(gpu)
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_shape = model.classifier[0].in_features
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_shape = model.classifier[0].in_features
    
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier = Sequential(Linear(input_shape, hidden_units),
                                  ReLU(),
                                  Dropout(dropout),
                                  Linear(hidden_units, 256),
                                  ReLU(),
                                  Linear(256, logits),
                                  LogSoftmax(dim=1))
    
    model = model.to(device)
    model_loss = NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, model_loss, optimizer

def load_model(model_path, gpu):
    checkpoint = load(model_path)
    architecture = checkpoint['structure']
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    
    model,_,_ = get_model(architecture,dropout,hidden_units,lr,gpu)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def set_device(gpu= False):
    if cuda.is_available() and gpu == True:
        current_device = device("cuda:0")
        print(f"[INFO] Device: {current_device}\n")
    else:
        current_device = device("cpu")
        print(f"[INFO] Device: {current_device}\n")
    return current_device

def get_labels(file_path= labels_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
