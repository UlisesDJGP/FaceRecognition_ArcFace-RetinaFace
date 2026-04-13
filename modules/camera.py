import cv2
import platform
import threading
import time

import socket
from urllib.parse import urlparse

class ThreadedCamera:
    def __init__(self, src=0, width=640, height=360):
        self.capture = None
        best_cam_src = None
        
        # 1. Validar conexión PING TCP puramente sin involucrar OpenCV (Cero lag al abortar)
        if isinstance(src, str):
            print(f"[Cámara] Validando puerto TCP para la cámara IP: {src}")
            parsed = urlparse(src)
            host = parsed.hostname
            # Si no hay puerto en el URL, asume puerto 80 nativo http
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

        # 2. Si falló la RED, abrir la webcam local con el backend disponible
        if not self.capture or not self.capture.isOpened():
            local_cam_index = 0
            if isinstance(src, int):
                local_cam_index = src
            elif platform.system() == "Linux":
                import glob
                import os
                # Buscar cámaras y preferir la de Sony (Thunderbolt) o cualquier cámara no integrada
                for path in sorted(glob.glob('/sys/class/video4linux/video*')):
                    name_file = os.path.join(path, 'name')
                    if os.path.exists(name_file):
                        try:
                            with open(name_file, 'r', encoding='utf-8', errors='ignore') as f:
                                cam_name = f.read().strip()
                                if "ILME" in cam_name or "Sony" in cam_name:
                                    idx_str = os.path.basename(path).replace('video', '')
                                    if idx_str.isdigit():
                                        local_cam_index = int(idx_str)
                                        print(f"[Cámara] Cámara Thunderbolt detectada: {cam_name} (Index: {local_cam_index})")
                                        break
                        except Exception:
                            pass

            # Backend condicional: V4L2 solo disponible en Linux
            backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
            backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "AUTO"
            print(f"[Cámara] Abriendo webcam local (Index: {local_cam_index}, backend: {backend_name})...")
            self.capture = cv2.VideoCapture(local_cam_index, backend)

            if self.capture.isOpened():
                if platform.system() == "Linux":
                    # Forzar codec MJPG para garantizar color y mayor FPS en Linux
                    self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                    # Restaurar controles V4L2 a valores de fábrica
                    self.capture.set(cv2.CAP_PROP_SATURATION, 64)
                    self.capture.set(cv2.CAP_PROP_BRIGHTNESS, 128)
                    self.capture.set(cv2.CAP_PROP_CONTRAST,   32)
                    self.capture.set(cv2.CAP_PROP_HUE,         0)
                    print(f"[Cámara] Controles V4L2 restaurados: SAT=64 BRI=128 CON=32 HUE=0")

                # Truco de la industria: pedir una resolución imposible.
                # El driver V4L2 la corrige automáticamente al máximo real del sensor,
                # evitando hardcodear modelos de cámara o llamadas prematuras a CAP_PROP_FRAME_WIDTH.
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  10000)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
                real_max_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                real_max_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if real_max_w > 0 and real_max_h > 0:
                    width  = min(width,  real_max_w)
                    height = min(height, real_max_h)
                    print(f"[Cámara] Sensor detectado: máximo {real_max_w}x{real_max_h} → usando {width}x{height}")

                best_cam_src = local_cam_index
                print(f"[Cámara] Webcam local activa en index: {local_cam_index}")
            else:
                self.capture.release()
                print(f"[Cámara] Fallo Crítico: No se pudo abrir la webcam local en index {local_cam_index}.")

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
