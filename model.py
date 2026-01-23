# import cv2 
# from retinaface import RetinaFace 
# import matplotlib as plt
# import os

# os.makedirs("faces", exist_ok=True)
# image_name = 'YO.jpg'
# img = cv2.imread(image_name)

# if img is None:
#    raise ValueError(f"Error al cargar la imagen: {image_name}")

# print(f"Imagen cargada. Buscando rostro...")
# response = RetinaFace.detect_faces(img)    

# if response is None:
#     print("No se detectaron rostros.")
#     exit()

# print(f"Se encontraron {len(response)} Rostros.")

# for face_id, identity in response.items():
#     x1, y1, x2, y2 = identity["facial_area"]

#     face_img = img[y1:y2, x1:x2]

#     if face_img.size == 0:
#         continue

#     filename = f"faces/{face_id}.jpg"
#     cv2.imwrite(filename, face_img)
#     print(f"Guardando {filename} con tamaño {face_img.shape}")



#     cv2.rectangle(
#         img,
#         (x1, y1),
#         (x2, y2),
#         (0, 255, 0),
#         2
#     )

#     for point in identity["landmarks"].values():
#         cv2.circle(
#             img,
#             (int(point[0]), int(point[1])),
#             2,
#             (0, 0, 255),
#             -1
#         )



# cv2.imshow('detector retinaFace',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#version de video
import cv2
from retinaface import RetinaFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

frame_count=0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count %5 !=0:
        cv2.imshow("camara",frame)
        cv2.waitKey(1)
        continue
    
    
    small = cv2.resize(frame, None, fx=0.5, f7=0.5 )
    response = RetinaFace.detect_faces(small)

    if response:
        for face_id, identity in response.items():
            x1, y1, x2, y2 = identity["facial_area"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.imshow("RetinaFace - Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
