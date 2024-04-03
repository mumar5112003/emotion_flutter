import torch
import cv2
from torchvision import transforms
from PIL import Image
from argparse import ArgumentParser
from torch import nn


class EmotionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv and pooling on (48, 48) img --> (24, 24) img
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding="same")
        self.batchnorm1 = nn.BatchNorm2d(64)  
        self.pool1 = nn.MaxPool2d(kernel_size=2) 
        self.dropout_conv1 = nn.Dropout(0.3) 
        
        # Receptive field of (7, 7) on (24, 24) img using 2 Conv and then pool --> (12, 12) img
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.batchnorm2 = nn.BatchNorm2d(128)  
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.batchnorm3 = nn.BatchNorm2d(128)  
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_conv3 = nn.Dropout(0.3)  
        
        # Receptive field of (9, 9) on (12, 12) img using 3 Conv and then pool --> (6, 6) img
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.batchnorm4 = nn.BatchNorm2d(256)  
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.batchnorm5 = nn.BatchNorm2d(256)  
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding="same")
        self.batchnorm6 = nn.BatchNorm2d(128)  
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout_conv6 = nn.Dropout(0.3)  
        
        self.flatten = nn.Flatten() # (Batch_size, 128, 6, 6) --> (Batch_size, 128 * 6 * 6)
        
        # Linear layers with higher dropout:
        self.fc1 = nn.Linear(128 * 6 * 6, 256) 
        self.batchnorm7 = nn.BatchNorm1d(256)  
        self.relu = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.5)  
        
        self.fc2 = nn.Linear(256, 128)
        self.batchnorm8 = nn.BatchNorm1d(128)  
        self.relu = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(0.5)  
        
        # Classifier with logits output
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout_conv1(x)
        
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout_conv3(x)
        
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.relu(self.batchnorm5(self.conv5(x)))
        x = self.relu(self.batchnorm6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout_conv6(x)
        
        x = self.flatten(x)
        
        x = self.relu(self.batchnorm7(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = self.relu(self.batchnorm8(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x
    

def predict_emotion(model, img, inference_transform, classes, device):
    # Transform image
    img_tensor = inference_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Make prediction
    with torch.inference_mode():
        output = model(img_tensor)

    # Get predicted probabilities
    probs = torch.softmax(output, dim=1).squeeze().cpu().detach().numpy()

    # Get predicted emotion label
    pred_class = classes[torch.argmax(output).item()]
    
    return probs, pred_class


def predict(img_path, save_path, model, inference_transform, classes, face_cascade, device):
    # Load the image
    image_array = cv2.imread(img_path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    gray_image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image_array, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop face from the image
        face_img_array = image_array[y:y+h, x:x+w]
        # Predict emotion for the face
        face_img = Image.fromarray(face_img_array)
        probs, pred_class = predict_emotion(model, face_img, inference_transform, classes, device)
        # Draw bounding box around the face
        thinkness = max(int(image_array.shape[0] / 200), 1)
        font_size = max(image_array.shape[0] / 400, 1)
        cv2.rectangle(image_array, (x, y), (x+w, y+h), (255, 0, 0), thinkness)
        # Display predicted emotion
        cv2.putText(image_array, f"{pred_class}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), thinkness)
    
    # Save the result
    print("Saving output...")
    cv2.imwrite(save_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the emotion classification model
    print("Loading the emotion classifier model")
    model = torch.load(args.model_path, map_location=device)
    model.eval()

    # Define the emotion classes the model was trained on
    classes = ['anger', 'happiness', 'neutral', 'sadness', 'surprise']

    # Define face classifier
    print("Loading the face classifier model")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    inference_transform = transforms.Compose([
        transforms.Resize(size=(48, 48)),  # Resize the image to 48x48
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor normalized to [0, 1]
    ])

    print("Predicting...")
    predict(args.img_path, args.save_path, model, inference_transform, classes, face_cascade, device)
    print(f"Ouput image has been saved to {args.save_path}")















from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import torch
import cv2

app = Flask(__name__)

# Load the model
model = torch.load("emotion_83.pt", map_location=torch.device('cpu'))
model.eval()

# Define emotion classes
classes = ['anger', 'happiness', 'neutral', 'sadness', 'surprise']

# Define image preprocessing transform
inference_transform = transforms.Compose([
    transforms.Resize(size=(48, 48)),  
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  
])

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get image file from request
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        # Preprocess image
        img_tensor = inference_transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
        
        # Get predicted probabilities
        probs = torch.softmax(output, dim=1).squeeze().cpu().detach().numpy()
        # Convert probs to a JSON-serializable type (Python list of floats)
        probs_json = [float(prob) for prob in probs]
        # Get predicted emotion label
        pred_class = classes[torch.argmax(output).item()]

        return jsonify({"predictions": {classes[i]: probs_json[i] for i in range(len(classes))}, "predicted_class": pred_class})

# @app.route("/predict", methods=["POST"])
# def predict():
#     if request.method == "POST":
#         # Get image file from request
#         file = request.files["image"]
#         img = Image.open(file.stream).convert("RGB")

#         # Preprocess image
#         img_tensor = inference_transform(img)
#         img_tensor = img_tensor.unsqueeze(0)

#         # Make prediction
#         with torch.no_grad():
#             output = model(img_tensor)
        
#         # Get predicted probabilities
#         probs = torch.softmax(output, dim=1).squeeze().cpu().detach().numpy()
#         # Get predicted emotion label
#         pred_class = classes[torch.argmax(output).item()]

#         return jsonify({"predictions": {classes[i]: probs[i] for i in range(len(classes))}, "predicted_class": pred_class})

if __name__ == "__main__":
    app.run(debug=True)










