import csv
import os
from datetime import datetime

ATTENDANCE_FILE = "attendance.csv"

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.isfile(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([name, date_str, time_str])