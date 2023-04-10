import numpy as np
from tensorflow.keras.models import model_from_json
import operator
import cv2
import sys
import os
import pyttsx3, time



# Loading the model
json_file_alphabet = open("model-bw_alphabet.json", "r")
model_json_alphabet = json_file_alphabet.read()
json_file_alphabet.close()
loaded_model = model_from_json(model_json_alphabet)
# load weights into new model
loaded_model.load_weights("model-bw_alphabet.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {'0': 0, 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E' ,'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}
##categories = { '0':0,'BOIRE': 'BOIRE', 'MERCI': 'MERCI', 'PLAIT': 'PLAIT'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROIQ
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (96, 96))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 96, 96, 1))
    prediction = { 'Rien détecter':[0][0],
                  'BOIRE': result[0][1],
                  'MERCIE': result[0][2],
                  'PLAIT': result[0][3],


                 }
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.imshow("Frame", frame)
    engine = pyttsx3.init()
    engine.say(prediction[0][0])
    engine.runAndWait()
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break

cap.release()
cv2.destroyAllWindows()
