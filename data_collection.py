#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install mediapipe')


# In[12]:


import mediapipe as mp
import numpy as np
import cv2


# In[13]:


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# In[14]:


name = input("Enter the name of the data : ")


# In[15]:


mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

holis = mp_holistic.Holistic()

hands = mp_hands.Hands()
drawing = mp_drawing


# In[16]:


X = []
data_size = 0
name="dataset"


# In[17]:


while True:
    ret, frm = cap.read()

    if not ret:  # ✅ Check if frame was captured
        print("Failed to capture frame. Exiting...")
        break

    lst = []
    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        X.append(lst)
        data_size += 1

    # Draw landmarks
    drawing.draw_landmarks(frm, res.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display data size
    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or data_size > 99:  # Press 'Esc' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save data
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)


# In[ ]:




