import face_recognition
import cv2
import numpy as np
import os

IMAGES_DIR = 'known_faces'
EXIT_KEY = "q"
SCALE_FACTOR = 0.5

# Load the known face images and encodings
known_faces = []
known_names = []
for filename in os.listdir(IMAGES_DIR):
    image_path = os.path.join(IMAGES_DIR, filename)

    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]

    known_names.append(filename.split(".")[0])
    known_faces.append(encoding)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    frame = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    
    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Iterate over each detected face
    for (i, face_encoding) in enumerate(face_encodings):
        # Determine if the face is a match for any known face
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)

        if len(known_faces) == 0:  # Check if there are no known faces
            name = "Unknown"
        else:
            # Determine the best match for the detected face
            best_match_idx = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
            if matches[best_match_idx]:
                name = known_names[best_match_idx] 
            else:
                name = "Unknown"
        
        cv2.rectangle(frame, (face_locations[i][3], face_locations[i][0]), (face_locations[i][1], face_locations[i][2]), (0, 255, 0), 2)
        cv2.putText(frame, name, (face_locations[i][3], face_locations[i][2]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord(EXIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()
