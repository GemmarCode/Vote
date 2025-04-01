import os

# Face Verification Settings
FACE_DATA_DIR = os.path.join(BASE_DIR, 'media', 'face_data')
os.makedirs(FACE_DATA_DIR, exist_ok=True) 