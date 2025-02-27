import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16, ResNet50
import pyttsx3

# Load pretrained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Load pretrained ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
          10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
          19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Initialize text-to-speech
engine = pyttsx3.init()

def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    img_array = np.array(resized).reshape(1, 224, 224, 3) / 255.0  # Normalize
    return img_array

def recognize_sign_vgg(frame):
    img_array = preprocess_frame(frame)
    features = vgg_model.predict(img_array)
    prediction = np.argmax(features)
    return labels.get(prediction, "Unknown")

def recognize_sign_resnet(frame):
    img_array = preprocess_frame(frame)
    features = resnet_model.predict(img_array)
    prediction = np.argmax(features)
    return labels.get(prediction, "Unknown")

# Open webcam for real-time sign detection
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    sign_vgg = recognize_sign_vgg(frame)
    sign_resnet = recognize_sign_resnet(frame)
    
    cv2.putText(frame, f"VGG: {sign_vgg}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"ResNet: {sign_resnet}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)
    
    engine.say(f"VGG detected {sign_vgg}. ResNet detected {sign_resnet}.")
    engine.runAndWait()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
