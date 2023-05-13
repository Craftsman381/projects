from utils import predict
from model import load_model, get_labels

import argparse

parser = argparse.ArgumentParser(description = 'Parser for predict.py')

parser.add_argument('--image_path', action= "store", type= str, default= './flowers/test/1/image_06752.jpg')
parser.add_argument('--checkpoint', action= "store", type= str, default= './checkpoint.pth')
parser.add_argument('--top_k', action= "store", type= int, default= 1)
parser.add_argument('--labels_path', action= "store", default= 'cat_to_name.json')
parser.add_argument('--gpu', action= "store", default= True)

args = parser.parse_args()
image_path = args.image_path
labels_path = args.labels_path
model_path = args.checkpoint
top_k = args.top_k
gpu = args.gpu

def main():
    
    print("[INFO] Predicting...\n")
    labels_dict = get_labels(labels_path)
    model = load_model(model_path, gpu)
    probabilities, class_ids = predict(image_path, model, top_k)
    class_names = [labels_dict[str(index+1)] for index in class_ids.tolist()[0]]
    class_ids = [id+1 for id in class_ids.tolist()[0]]

    print(f"Probabilities: {probabilities.tolist()[0]}")
    print(f"Class IDs: {class_ids}")
    print(f"Class Names: {class_names}")
    
    print(f"\n[INFO] Pridiction Complete!")

if __name__== "__main__":
    main()