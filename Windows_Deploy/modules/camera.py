import cv2
import threading
import time
import socket
from urllib.parse import urlparse

class ThreadedCamera:
    def __init__(self, src=0, width=640, height=360):
        self.capture = None
        best_cam_src = None
        
        # 1. Validar conexión TCP antes de intentar OpenCV (evita freeze de 12 segundos)
        if isinstance(src, str):
            print(f"[Cámara] Validando puerto TCP para la cámara IP: {src}")
            parsed = urlparse(src)
            host = parsed.hostname
            port = parsed.port if parsed.port else 80
            
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect((host, port))
                print(f"[Cámara] TCP Activo. Escaneando flujo de video en {host}:{port}...")
                self.capture = cv2.VideoCapture(src)
                if self.capture.isOpened():
                    best_cam_src = src
            except Exception:
                print(f"[Cámara] Servidor TCP Inactivo o Red Inalcanzable. Abortando IP de Red.")
                self.capture = None

        # 2. Fallback a webcam local usando DirectShow (backend nativo de Windows)
        if not self.capture or not self.capture.isOpened():
            local_cam_index = 0
            
            print(f"[Cámara] Abriendo webcam local (Index: {local_cam_index}) con DirectShow...")
            self.capture = cv2.VideoCapture(local_cam_index, cv2.CAP_DSHOW)
            
            if self.capture.isOpened():
                best_cam_src = local_cam_index
                print(f"[Cámara] ¡Éxito! Webcam Windows enlazada en el puerto: {best_cam_src}")
            else:
                self.capture.release()
                print(f"[Cámara] Fallo Crítico. Windows rechazó ceder la cámara Index {local_cam_index}.")

        if not self.capture or not self.capture.isOpened():
            raise RuntimeError("CRASH FINAL: No se pudo enlazar ni la Cámara IP ni la Webcam local.")

        # Optimizar para flujos de red: tamaño de buffer en 1 elimina la latencia (lag) acumulada
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
            
        # Leer el primer frame para asegurar que status y frame no sean None
        (self.status, self.frame) = self.capture.read()
        self.stopped = False
        
        # Start frame retrieval thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            else:
                self.stopped = True
            
    def read(self):
        return self.status, self.frame
        
    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.5)
            
        if self.capture.isOpened():
            self.capture.release()

def open_camera(camera_index=0, width=640, height=360):
    return ThreadedCamera(camera_index, width, height)
