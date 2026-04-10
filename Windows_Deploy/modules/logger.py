import os
from datetime import datetime

LOG_FOLDER = "logs"
LOG_FILE = "logs/system.log"


# crear carpeta logs si no existe
os.makedirs(LOG_FOLDER, exist_ok=True)


def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line = f"[{timestamp}] {message}"

    # imprimir en consola
    print(line)

    # guardar en archivo
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")