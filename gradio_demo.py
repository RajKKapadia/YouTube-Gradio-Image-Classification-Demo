import gradio as gr
import torch
import requests
from torchvision import transforms
from torchvision.models import ResNet18_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.hub.load(
    source='local',
    repo_or_dir='pytorch-vision-b68adcf/',
    model='resnet18',
    weights=ResNet18_Weights.DEFAULT).eval().to(device)

response = requests.get(
    "https://raw.githubusercontent.com/gradio-app/mobilenet-example/master/labels.txt")
labels = response.text.split("\n")


def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences


demo = gr.Interface(fn=predict,
                    title='Image Classification',
                    description='This is an image classification demo using PyTorch ResNet18 implementation.',
                    inputs=gr.components.Image(type="pil"),
                    outputs=gr.components.Label(num_top_classes=5),
                    examples=[["sample/cheetah.jpeg"], ["sample/dog.jpg"],
                              ["sample/peguin.webp"], ["sample/zebra.jpeg"], ["sample/monkey.webp"], ["sample/chair.webp"]],
                    theme=gr.themes.Monochrome(),
                    allow_flagging='never'
                    )

demo.launch()
