# Define a function to make predictions
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


if __name__ == '__main__': 
    
    
   # Load the model for inference
    model = models.googlenet(pretrained=False, aux_logits=True)  # Set aux_logits to True to match the saved model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15)  # Adjust num_classes to match your dataset
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, 15)
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, 15)

    model_path = 'F:\\fyp_system\\utils\\model\\colour.pth'
    model.load_state_dict(torch.load(model_path))
    
    model.eval()  # Set the model to evaluation mode
    print("Model loaded and ready for inference")

    def predict(image_path, model):
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Move the image to the appropriate device
        

        # Set the model to evaluation mode and make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    # Example usage
    image_path = "F:\\fyp_system\\result\\2024-06-07 15-24-53\\crop\\JSL4468.jpg"
    colour_class = ['beige','black','blue','brown','gold','green','grey','orange','pink','purple','red','silver','tan','white','yellow']
    class_index = predict(image_path, model)
    print(f'Predicted class index: {class_index}')
    print(f'Predicted class : {colour_class[class_index]}')
    

