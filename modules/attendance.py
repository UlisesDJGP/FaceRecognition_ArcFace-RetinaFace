import csv
import os
from datetime import datetime

ATTENDANCE_FILE = "attendance.csv"

# Horario simplificado (puedes ajustar después)
SCHEDULE = {
    "Monday": {
        (7,0,7,50): "Ecuaciones Diferenciales",
        (7,50,8,40): "Ingles V",
        (8,40,9,30): "Aprendizaje Maquina",
        (10,0,10,50): "Proyecto Integrador II",
        (10,50,11,40): "Liderazgo de Equipo",
        (11,40,12,30): "Fundamentos de Vision",
        (12,30,13,20): "Mineria de Datos"
    }
}

def get_current_subject():
    now = datetime.now()
    day = now.strftime("%A")

    if day not in SCHEDULE:
        return None

    current_minutes = now.hour * 60 + now.minute

    for (h1, m1, h2, m2), subject in SCHEDULE[day].items():
        start = h1 * 60 + m1
        end = h2 * 60 + m2

        if start <= current_minutes <= end:
            return subject

    return None

def get_current_subject():
    now = datetime.now()
    day = now.strftime("%A")

    if day not in SCHEDULE:
        return None

    current_minutes = now.hour * 60 + now.minute

    for (h1, m1, h2, m2), subject in SCHEDULE[day].items():
        start = h1 * 60 + m1
        end = h2 * 60 + m2

        if start <= current_minutes <= end:
            return subject

    return None

def init_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nombre", "fecha", "hora", "materia"])
            

def already_registered(name, subject, date):
    if not os.path.exists(ATTENDANCE_FILE):
        return False

    with open(ATTENDANCE_FILE, mode="r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if row[0] == name and row[1] == date and row[3] == subject:
                return True

    return False

def register_attendance(name):

    subject = get_current_subject()

    if subject is None:
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if already_registered(name, subject, date):
        return

    with open(ATTENDANCE_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time, subject])

    print(f"Asistencia registrada: {name} - {subject}")