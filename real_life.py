import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model


scal = pickle.load(open('Pipeline/stan_scale.sav', 'rb'))
#model = pickle.load(open('Model/random_forest.sav', 'rb'))
model = load_model('Model/cnn.keras')
label = pickle.load(open('Pipeline/label_encoder.sav', 'rb'))

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define a function to process images and extract hand landmarks
def process_frame(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Extract hand landmarks from results
    landmarks = []
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return landmarks, results

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame and get landmarks
    landmarks, results = process_frame(frame)

    # Draw landmarks on the frame for visualization
    if landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    
    #print(landmarks)
    #print(len(landmarks))

    if landmarks:  # Check if landmarks is not empty
        landmarks = np.array(landmarks) #scal.transform(np.array(landmarks).reshape(1, -1))
        result = np.argmax(model.predict(landmarks.reshape(1,63)))
        #print(result)
        print(label.inverse_transform([result]))

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()