import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define a function to process images and extract hand landmarks
def process_image(image_path, label):
    image = cv2.imread(image_path)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Extract hand landmarks from results
    landmarks = []
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return landmarks, label

# Define a list to store data points (image landmarks and labels)
data = []

# Specify the folders containing Bangla and English sign language images
bangla_folder = "D:/SignLanguage/data/Bangla/BdSL-D1500/0"  # Replace with your folder path
english_folder = "D:/SignLanguage/data/English/A/a"  # Replace with your folder path

# Iterate through the Bangla sign language images
for filename in os.listdir(bangla_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(bangla_folder, filename)
        landmarks, label = process_image(image_path, "Bangla")
        # Check if landmarks are not empty
        if landmarks:
            data.append({"ImagePath": image_path, **{f"Landmark_{i}": val for i, val in enumerate(landmarks)}, "Label": label})

# Iterate through the English sign language images
for filename in os.listdir(english_folder):
    if filename.endswith ((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(english_folder, filename)
        landmarks, label = process_image(image_path, "English")
        # Check if landmarks are not empty
        if landmarks:
            data.append({"ImagePath": image_path, **{f"Landmark_{i}": val for i, val in enumerate(landmarks)}, "Label": label})

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("sign_language_data.csv", index=False)
