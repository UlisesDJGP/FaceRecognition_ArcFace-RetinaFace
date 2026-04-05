import cv2

#aqui va lo relacionado con la ejecucion de la camara, por default busca la webcam, cambiando los valores se puede reconocer otro dispositivo


def open_camera(camera_index=0, width=640, height=360):
    cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    return cap
