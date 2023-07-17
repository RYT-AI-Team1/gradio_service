import gradio as gr
import torch
import torchvision
import requests
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Resize

# api_url = "http://10.1.38.211:5202/upload"

# Step 1: Load the trained model
checkpoint = torch.load("api-server/model/best_mbv2.pt")

# Model
model = torchvision.models.mobilenet_v2(pretrained=True)

# Replace the last layer with a new fully connected layer
num_classes = 7  # Replace this with the actual number of output classes
model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

model.load_state_dict(checkpoint['state_dict'])
# model.eval()

transform = transforms.Compose([
    Resize((256, 256)),
    transforms.ToTensor()
])

with open("api-server/server/static/classes.txt", "r") as f:
    labels = f.read().split("\n")

# response = requests.get("api-server/server/static/classes.txt")
# labels = response.text.split("\n")

def predict_alignment(inp):
    # Add your classification logic here
    text_input = gr.Textbox()
    return text_input

def predict_classification(image, image_name):
    inp = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(7)}
        max_confidence = max(confidences, key=confidences.get)
        class_id = labels.index(max_confidence)
    result = {
        "image_name": image_name,
        "class": max_confidence,
        "class_id": class_id,
        "confidence": confidences[max_confidence]
    }
    return result

def display_result(result):
    return f"Image name: {result['image_name']}\nClass: {result['class']} (ID: {result['class_id']})\nConfidence: {result['confidence']:.2f}"

def predict_and_display(image_name, image):
    result = predict_classification(image, image_name)
    return display_result(result)

with gr.Blocks() as interface:
    with gr.Tab("Alignment"):
        with gr.Row():
            image_input = gr.Image()
        image_button = gr.Button("Run")
    with gr.Tab("Classification"):
        with gr.Row():
            image_name_input = gr.Textbox(label="Image name")
            image_classification_input = gr.Image(type="pil", label="Image")
        image_classification_button = gr.Button("Run")
        
        image_classification_button.click(predict_and_display,
                                           inputs=[image_name_input, image_classification_input],
                                           outputs=gr.outputs.Textbox(label="Result", type="text"))

interface.launch()
