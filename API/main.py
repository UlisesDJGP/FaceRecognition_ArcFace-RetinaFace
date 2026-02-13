from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import shutil
import json
from datetime import datetime

app = FastAPI()

# 1. Endpoint para REGISTRAR un alumno nuevo (Guardar en BD)
@app.post("/registrar_alumno/")
async def registrar(
    nombre: str = Form(...),
    matricula: str = Form(...),
    grado: str = Form(...),
    foto: UploadFile = File(...)
):
    # 1. Guardar la foto en disco o procesarla
    ubicacion_foto = f"static/{matricula}.jpg"
    with open(ubicacion_foto, "wb") as buffer:
        shutil.copyfileobj(foto.file, buffer)
    
    # 2. (IMPORTANTE) Aquí tu modelo debe extraer el 'encoding' de la cara
    # encoding = modelo.obtener_encoding(ubicacion_foto)
    
    # 3. Guardar datos en MongoDB
    alumno_data = {
        "nombre": nombre,
        "matricula": matricula,
        "grado": grado,
        "foto_path": ubicacion_foto,
        # "face_encoding": encoding  <-- Guarda esto para comparaciones rápidas
    }
    # collection_alumnos.insert_one(alumno_data)
    
    return {"status": "Alumno registrado exitosamente"}

# 2. Endpoint para TOMAR ASISTENCIA (Validar cara y generar JSON)
@app.post("/verificar_asistencia/")
async def verificar(foto_entrada: UploadFile = File(...)):
    
    # 1. Guardar foto temporal
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(foto_entrada.file, buffer)
    
    # 2. Tu modelo compara 'temp.jpg' con las caras en la BD
    # alumno_encontrado = modelo.buscar_cara("temp.jpg", collection_alumnos)
    
    # Simulación de resultado
    alumno_encontrado = {
        "nombre": "Juan Perez",
        "matricula": "12345",
        "existe": True
    }

    # 3. Lógica de Asistencia y Creación de JSON
    resultado_asistencia = {}
    
    if alumno_encontrado and alumno_encontrado["existe"]:
        estado = "ASISTIO"
        datos_alumno = alumno_encontrado
    else:
        estado = "NO_IDENTIFICADO"
        datos_alumno = None

    resultado_asistencia = {
        "fecha": datetime.now().isoformat(),
        "estado": estado,
        "alumno_detectado": datos_alumno
    }

    # 4. Generar el archivo .json físico como pediste
    nombre_archivo = f"asistencia_logs/asistencia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(nombre_archivo, "w") as f:
        json.dump(resultado_asistencia, f, indent=4)

    return resultado_asistencia

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)