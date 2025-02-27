import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3

# Load the trained ASL model (Ensure you train and save the model first)
model = load_model("asl_model.h5")  # Replace with your trained model file
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
          10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
          19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Initialize text-to-speech
engine = pyttsx3.init()

def recognize_sign(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))  # Adjust based on dataset resolution
    img_array = np.array(resized).reshape(1, 28, 28, 1) / 255.0  # Normalize
    prediction = model.predict(img_array)
    return labels[np.argmax(prediction)]

# Open webcam for real-time sign detection
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    sign = recognize_sign(frame)
    cv2.putText(frame, f"Detected: {sign}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)
    
    engine.say(sign)
    engine.runAndWait()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
