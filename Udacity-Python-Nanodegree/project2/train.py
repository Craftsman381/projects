from torch import mean, optim, exp, no_grad, save, FloatTensor, __version__
from utils import prepare_data
from model import get_model, set_device, get_labels
print(f"Pytorch Version: {__version__}")

import argparse
import numpy as np
import json

parser = argparse.ArgumentParser( description = 'Parser for train.py')

parser.add_argument('--save_dir', action= "store", default= "./checkpoint.pth")
parser.add_argument('--arch', action= "store", default= "vgg16")
parser.add_argument('--learning_rate', action= "store", type= float, default= 0.0015)
parser.add_argument('--hidden_units', action= "store", dest= "hidden_units", type= int, default= 1024)
parser.add_argument('--epochs', action= "store", type= int, default= 2)
parser.add_argument('--dropout', action= "store", type= float, default= 0.025)
parser.add_argument('--gpu', action= "store", default= True)

args = parser.parse_args()

model_path = args.save_dir
lr = args.learning_rate
architecture = args.arch
hidden_units = args.hidden_units
gpu = args.gpu
epochs = args.epochs
dropout = args.dropout

def main():
    device = set_device(gpu)
    
    train_dataloader, val_dataloader, test_dataloader, class_ids = prepare_data()
    model, model_loss, optimizer = get_model(architecture,dropout,hidden_units,lr,gpu)
    print(f"\n[INFO] Model Summary: {model}\n")
    
    steps= 0
    train_loss= 0
    INTERVAL= 10
    
    print("[INFO] Training started...")
    
    for epoch in range(epochs):
        for images, labels in train_dataloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            pred = model.forward(images)
            loss = model_loss(pred, labels)
            
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if steps % INTERVAL == 0:
                model.eval()
                val_loss, val_accuracy = 0, 0

                with no_grad():
                    for images, labels in val_dataloader:

                        images, labels = images.to(device), labels.to(device)
                        pred_log = model.forward(images)
                        batch_loss = model_loss(pred_log, labels)

                        val_loss += batch_loss.item()

                        pred = exp(pred_log)
                        top_pred, top_class = pred.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)

                        val_accuracy += mean(equals.type(FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss/INTERVAL:.3f} | "
                      f"Val Loss: {val_loss/len(val_dataloader):.3f} | "
                      f"Val Accuracy: {val_accuracy/len(val_dataloader):.3f}")

                model.train()
    print("\n[INFO] Training Completed!")
    
    model.class_to_idx=  class_ids
    save({  'structure': architecture,
            'hidden_units': hidden_units,
            'dropout': dropout,
            'learning_rate': lr,
            'no_of_epochs': epochs,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}, model_path)
    
    print("[INFO] Checkpoint Saved Successfully!")
    
if __name__ == "__main__":
    main()