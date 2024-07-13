
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

# Specify the folders containing sign language images
folders = ["D:/SignLanguage/data/Bangla/banglaWords/bhalobashi",
           "D:/SignLanguage/data/Bangla/banglaWords/bujhlam",
           "D:/SignLanguage/data/Bangla/banglaWords/dhonnobad",
           "D:/SignLanguage/data/Bangla/banglaWords/dirgho",
           "D:/SignLanguage/data/Bangla/banglaWords/hello",
           "D:/SignLanguage/data/Bangla/banglaWords/ji",
           "D:/SignLanguage/data/Bangla/banglaWords/kharap",
           "D:/SignLanguage/data/Bangla/banglaWords/khushi",
           "D:/SignLanguage/data/Bangla/banglaWords/na",
           "D:/SignLanguage/data/Bangla/banglaWords/sahajjo",
           "D:/SignLanguage/data/Bangla/banglaWords/somoy",
           "D:/SignLanguage/data/Bangla/banglaWords/taka",
           "D:/SignLanguage/data/Bangla/banglaWords/valo", 
           "D:/SignLanguage/data/Bangla/bdSL-D1500/0",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/1",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/2",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/3",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/4",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/5",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/6",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/7",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/8",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/9",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_o_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_a_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_i_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_u_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_ro_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_e_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_Oi_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_O_CC",
            "D:/SignLanguage/data/Bangla/bdSL-D1500/S1_OU_CC", ]  # Add folder names as needed

# Define an array of labels corresponding to the folders
labels = ["bhalobashi", "bujhlam", "dhonnobad", "dirgho", "hello", "ji", "kharap",
           "khushi", "na", "sahajjo", "somoy", "taka", "valo", "0", "1", "2", "3",
            "4", "5", "6", "7", "8", "9", "o", "a", "i",]  # Add labels in the same order as folders

# Iterate through the folders
for folder, label in zip(folders, labels):
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder, filename)
            landmarks, label = process_image(image_path, label)
            # Check if landmarks are not empty
            if landmarks:
                data.append({**{f"Landmark_{i}": val for i, val in enumerate(landmarks)}, "Label": label})

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("sign_language_data.csv", index=False)
