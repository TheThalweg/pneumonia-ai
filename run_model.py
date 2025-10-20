from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pneumonia_resnet18.pth", map_location=device))
model.eval()

# 1. Load your image
while True:
    img_path = input("Enter your file path: ")
    try:
        img = Image.open(img_path).convert("RGB")  # convert to 3 channels
    except:
        print("image can't be opened")
        continue

    # 2. Apply transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension

    # 3. Move to device
    img_tensor = img_tensor.to(device)

    # 4. Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    # 5. Map prediction to label
    classes = ['NORMAL', 'PNEUMONIA']
    print("Prediction:", classes[pred.item()])
