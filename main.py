from deepface.modules.representation import represent
from deepface.modules.verification import find_cosine_distance
from ultralytics import YOLO
import numpy as np
import cv2
import os



model = YOLO('Yolo weight/yolov8n-face.pt')
recognize_model = 'Facenet'
detector_backend = 'ssd'
THRESHOLD = 0.45



def create_face_db(db_face_path):
    db_face = {}
    names = os.listdir(path=db_face_path)
    for name in names:
        imgs = os.listdir(db_face_path + '/' + name)
        db_face[name] = []
        for img in imgs:
            emb = represent(db_face_path + f'/{name}/' +img, 
                            detector_backend=detector_backend,
                            enforce_detection=False,
                            model_name=recognize_model)[0]['embedding']
            
            db_face[name].append(emb)
        db_face[name] = np.array(db_face[name], dtype=np.float64)
    return db_face

def detect_face(frame):
    faces = []
    results = model(frame)
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.cpu().numpy()
            faces.append((int(x1), int(y1), int(x2), int(y2)))
    return faces

def get_min_dist_and_name(target_emb, db_face: dict) -> tuple:
    name: str
    min_dist = float('inf')
    for id in db_face.keys():
        for vector in db_face[id]:
            dist = find_cosine_distance(target_emb, vector)
            if dist < min_dist:
                min_dist = dist
                name = id
    return name, min_dist


def recognize_faces(faces_inf, face_db):
    recog_faces = []
    for face_inf in faces_inf:
        embedding = face_inf['embedding']
        embedding = np.asarray(embedding)

        facial_area = face_inf['facial_area']
        x1, y1 = facial_area['x'], facial_area['y']
        x2, y2 = x1 + facial_area['w'], y1 + facial_area['h']

        name, min_distance = get_min_dist_and_name(embedding, face_db)

        if min_distance > THRESHOLD:
            name = 'Unknow'
        
        recog_faces.append((x1, y1, x2, y2, name))
    return recog_faces

def use_webcam(face_db):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces_inf = represent(frame, detector_backend=detector_backend, 
                              enforce_detection=False, 
                              model_name=recognize_model)
        
        results = recognize_faces(faces_inf, face_db)

        for (x1, y1, x2, y2, name) in results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow('Face Recognition', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    face_db = create_face_db('Raw Database')
    use_webcam(face_db)


if __name__ == "__main__":
    main()