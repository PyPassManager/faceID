import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def create_face_detector():
    return face_mesh

def capture_faces(face_detector, duration=15, fps=20):
    cap = cv2.VideoCapture(0)
    faces_captured = []
    landmarks_captured = []
    
    print("Bougez lentement votre tête dans tous les sens pendant 10 secondes...")
    start_time = time.time()
    last_capture_time = start_time
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        cv2.imshow('Capture visage', frame)
        
        current_time = time.time()
        if current_time - last_capture_time >= 1/fps:
            if results.multi_face_landmarks:
                faces_captured.append(frame)
                landmarks_captured.append(results.multi_face_landmarks[0])
                print(f"Image capturée. Total: {len(faces_captured)}")
                last_capture_time = current_time
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return faces_captured, landmarks_captured

def save_faces(faces, landmarks, name):
    user_dir = os.path.join('known_faces', name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    for i, (face, landmark) in enumerate(zip(faces, landmarks)):
        cv2.imwrite(os.path.join(user_dir, f'{i}.jpg'), face)
        landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmark.landmark])
        np.save(os.path.join(user_dir, f'{i}_landmarks.npy'), landmark_array)

def load_known_faces():
    known_faces = {}
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    for user_dir in os.listdir('known_faces'):
        user_path = os.path.join('known_faces', user_dir)
        if os.path.isdir(user_path):
            known_faces[user_dir] = []
            for filename in os.listdir(user_path):
                if filename.endswith('.jpg'):
                    face = cv2.imread(os.path.join(user_path, filename))
                    landmarks = np.load(os.path.join(user_path, f'{filename[:-4]}_landmarks.npy'))
                    known_faces[user_dir].append((face, landmarks))
    return known_faces

def compare_faces(landmarks, known_faces):
    for name, known_data in known_faces.items():
        matches = 0
        for _, known_landmarks in known_data:
            difference = np.mean(np.linalg.norm(landmarks - known_landmarks, axis=1))
            if difference < 0.135:  # Vous pouvez ajuster ce seuil
                matches += 1
        if matches >= len(known_data) // 2:
            return name
    return "Inconnu"

def main():
    face_detector = create_face_detector()
    known_faces = load_known_faces()

    if len(known_faces) == 0:
        print("Aucun visage connu. Capturez un visage...")
        faces, landmarks = capture_faces(face_detector)
        if faces is not None:
            name = input("Entrez le nom de cette personne : ")
            save_faces(faces, landmarks, name)
            known_faces[name] = [(face, np.array([[lm.x, lm.y, lm.z] for lm in landmark.landmark])) 
                                 for face, landmark in zip(faces, landmarks)]
            print(f"Visages de {name} capturés.")
    
    print("Début de la reconnaissance faciale...")
    print("Appuyez sur 'n' pour ajouter un nouvel utilisateur.")
    print("Appuyez sur 'q' pour quitter.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                name = compare_faces(landmarks, known_faces)
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Reconnaissance faciale', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            cap.release()
            cv2.destroyAllWindows()
            print("Capture d'un nouvel utilisateur...")
            faces, landmarks = capture_faces(face_detector)
            if faces is not None:
                name = input("Entrez le nom de cette personne : ")
                save_faces(faces, landmarks, name)
                known_faces[name] = [(face, np.array([[lm.x, lm.y, lm.z] for lm in landmark.landmark])) 
                                     for face, landmark in zip(faces, landmarks)]
                print(f"Visages de {name} capturés.")
            cap = cv2.VideoCapture(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
