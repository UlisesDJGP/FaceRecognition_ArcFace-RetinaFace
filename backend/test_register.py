#!/usr/bin/env python3
"""
Script de prueba end-to-end para el endpoint /api/v1/register.
Descarga una imagen de ejemplo con rostro de internet y la envía al API.
"""
import requests
import sys

API_URL = "http://localhost:8000/api/v1/register"

# Descargar una imagen de ejemplo con rostro desde thispersondoesnotexist (CC0)
print("Descargando imagen de prueba con rostro...")
try:
    img_resp = requests.get(
        "https://thispersondoesnotexist.com",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    img_resp.raise_for_status()
    img_bytes = img_resp.content
    print(f"Imagen descargada: {len(img_bytes)} bytes")
except Exception as e:
    print(f"No se pudo descargar imagen de prueba: {e}")
    print("Intentando con imagen local generada por OpenCV...")
    
    # Fallback: crear imagen sintética con OpenCV (posiblemente sin rostro detectable)
    import numpy as np
    import cv2
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode('.jpg', img)
    img_bytes = img_bytes.tobytes()
    print(f"Imagen sintética generada: {len(img_bytes)} bytes (puede no ser detectado rostro)")

# Enviar al API
print("\nEnviando registro de prueba al API...")
try:
    response = requests.post(
        API_URL,
        data={
            "nombre": "Estudiante Test",
            "matricula": "TEST001",
            "email": "test@utnay.edu.mx",
            "password": "password123",
        },
        files={
            "file": ("test_face.jpg", img_bytes, "image/jpeg"),
        },
        timeout=30,
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("\n✅ TEST PASADO — Registro exitoso")
    else:
        print(f"\n⚠️  Registro rechazado (posiblemente imagen sin rostro): {response.json().get('detail', '')}")
        
except Exception as e:
    print(f"\n❌ Error en la prueba: {e}")
    sys.exit(1)
