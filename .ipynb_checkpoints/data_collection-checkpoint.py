{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "96cddc17-ce09-43ff-83b6-aad4c26e836b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: mediapipe in c:\\users\\asus\\anaconda3\\lib\\site-packages (0.10.21)\n",
      "Requirement already satisfied: absl-py in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (2.1.0)\n",
      "Requirement already satisfied: attrs>=19.1.0 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (23.1.0)\n",
      "Requirement already satisfied: flatbuffers>=2.0 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (25.2.10)\n",
      "Requirement already satisfied: jax in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (0.5.3)\n",
      "Requirement already satisfied: jaxlib in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (0.5.3)\n",
      "Requirement already satisfied: matplotlib in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (3.9.2)\n",
      "Requirement already satisfied: numpy<2 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (1.26.4)\n",
      "Requirement already satisfied: opencv-contrib-python in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (4.11.0.86)\n",
      "Requirement already satisfied: protobuf<5,>=4.25.3 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (4.25.3)\n",
      "Requirement already satisfied: sounddevice>=0.4.4 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (0.5.1)\n",
      "Requirement already satisfied: sentencepiece in c:\\users\\asus\\anaconda3\\lib\\site-packages (from mediapipe) (0.2.0)\n",
      "Requirement already satisfied: CFFI>=1.0 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from sounddevice>=0.4.4->mediapipe) (1.17.1)\n",
      "Requirement already satisfied: ml_dtypes>=0.4.0 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from jax->mediapipe) (0.5.1)\n",
      "Requirement already satisfied: opt_einsum in c:\\users\\asus\\anaconda3\\lib\\site-packages (from jax->mediapipe) (3.4.0)\n",
      "Requirement already satisfied: scipy>=1.11.1 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from jax->mediapipe) (1.13.1)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (1.4.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (24.1)\n",
      "Requirement already satisfied: pillow>=8 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (10.4.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (3.1.2)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (2.9.0.post0)\n",
      "Requirement already satisfied: pycparser in c:\\users\\asus\\anaconda3\\lib\\site-packages (from CFFI>=1.0->sounddevice>=0.4.4->mediapipe) (2.21)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\asus\\anaconda3\\lib\\site-packages (from python-dateutil>=2.7->matplotlib->mediapipe) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install mediapipe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2492fb49-d8fb-4762-97d5-3683d940f63c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import mediapipe as mp\n",
    "import numpy as np\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8837365c-f7b2-4fe9-992a-a3e7541f5ef1",
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "if not cap.isOpened():\n",
    "    print(\"Error: Could not open webcam.\")\n",
    "    exit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c7a10690-139f-45ff-85a7-4720f640913c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Enter the name of the data :  badluck\n"
     ]
    }
   ],
   "source": [
    "name = input(\"Enter the name of the data : \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "02714157-5889-425d-90df-8c9a68710b67",
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_holistic = mp.solutions.holistic\n",
    "mp_hands = mp.solutions.hands\n",
    "mp_face_mesh = mp.solutions.face_mesh\n",
    "mp_drawing = mp.solutions.drawing_utils\n",
    "\n",
    "holis = mp_holistic.Holistic()\n",
    "\n",
    "hands = mp_hands.Hands()\n",
    "drawing = mp_drawing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c318ec26-3296-4ca4-b465-63c12fa846ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = []\n",
    "data_size = 0\n",
    "name=\"dataset\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "eedd0495-ccbe-40d6-bc42-0db4dc346d81",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(100, 1020)\n"
     ]
    }
   ],
   "source": [
    "while True:\n",
    "    ret, frm = cap.read()\n",
    "\n",
    "    if not ret:  # ✅ Check if frame was captured\n",
    "        print(\"Failed to capture frame. Exiting...\")\n",
    "        break\n",
    "\n",
    "    lst = []\n",
    "    frm = cv2.flip(frm, 1)\n",
    "\n",
    "    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))\n",
    "\n",
    "    if res.face_landmarks:\n",
    "        for i in res.face_landmarks.landmark:\n",
    "            lst.append(i.x - res.face_landmarks.landmark[1].x)\n",
    "            lst.append(i.y - res.face_landmarks.landmark[1].y)\n",
    "\n",
    "        if res.left_hand_landmarks:\n",
    "            for i in res.left_hand_landmarks.landmark:\n",
    "                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)\n",
    "                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)\n",
    "        else:\n",
    "            for i in range(42):\n",
    "                lst.append(0.0)\n",
    "\n",
    "        if res.right_hand_landmarks:\n",
    "            for i in res.right_hand_landmarks.landmark:\n",
    "                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)\n",
    "                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)\n",
    "        else:\n",
    "            for i in range(42):\n",
    "                lst.append(0.0)\n",
    "\n",
    "        X.append(lst)\n",
    "        data_size += 1\n",
    "\n",
    "    # Draw landmarks\n",
    "    drawing.draw_landmarks(frm, res.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)\n",
    "    drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)\n",
    "    drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)\n",
    "\n",
    "    # Display data size\n",
    "    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)\n",
    "    cv2.imshow(\"window\", frm)\n",
    "\n",
    "    if cv2.waitKey(1) == 27 or data_size > 99:  # Press 'Esc' to exit\n",
    "        break\n",
    "\n",
    "# Cleanup\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n",
    "\n",
    "# Save data\n",
    "np.save(f\"{name}.npy\", np.array(X))\n",
    "print(np.array(X).shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fc93aa29-5ae4-4c7b-ba4a-d81c5d32eaca",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
