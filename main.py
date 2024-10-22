import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from base64 import b64encode

# Inicijalizirajte FastAPI aplikaciju
app = FastAPI()

# Montirajte statičke datoteke
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicijalizirajte predloške
templates = Jinja2Templates(directory="templates")

# Učitajte pretrenirani model
model = models.resnet18(pretrained=True)
model.eval()

# Definirajte transformacije slike
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Učitajte klase ImageNet
labels = []
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_predictions = []
    for i in range(top5_prob.size(0)):
        top5_predictions.append((labels[top5_catid[i]], top5_prob[i].item()))

    # Kodirajte sliku u base64 za prikaz u HTML-u
    image_data_base64 = b64encode(image_data).decode('utf-8')

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predictions": top5_predictions,
        "image_data": image_data_base64
    })
