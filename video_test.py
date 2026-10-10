import os
import csv
import cv2
import time
from ultralytics import YOLO


# SETTINGS

VIDEO_PATH = "dataset/testing_videos/night.avi"
OUTPUT_PATH = "results/01_yolo11n_night_person_alarm.mp4"
MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.4

MOTION_THRESHOLD = 25
MIN_MOTION_AREA = 150

MOTION_RATIO = 0.02
MIN_MOVEMENT_FRAMES = 3

USE_CLAHE = False


# CREATE RESULTS FOLDER

os.makedirs("results", exist_ok=True)


# LOAD MODEL

model = YOLO(MODEL_PATH)


# OPEN VIDEO

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Не удалось открыть видео:", VIDEO_PATH)
    raise SystemExit(1)


fps_video = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps_video <= 0:
    fps_video = 25


# OUTPUT VIDEO

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps_video,
    (width, height)
)


# CLAHE

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# VARIABLES

previous_gray = None
movement_counter = {}

frame_number = 0
alarm_frames = 0
total_people_detections = 0
processing_start = time.time()

motion_results = []


# START

print("======================================")
print("Система видеонаблюдения запущена")
print("======================================")
print("Видео:", VIDEO_PATH)
print("Модель:", MODEL_PATH)
print("CLAHE:", USE_CLAHE)
print("Confidence:", CONFIDENCE)
print("Режим ALARM: обнаружение человека")
print("Для остановки нажмите Q")
print()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # IMAGE PREPROCESSING

    processed_frame = frame.copy()

    if USE_CLAHE:
        lab = cv2.cvtColor(
            processed_frame,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        l_channel = clahe.apply(l_channel)

        lab = cv2.merge(
            (l_channel, a_channel, b_channel)
        )

        processed_frame = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

    # GRAYSCALE + GAUSSIAN BLUR

    gray = cv2.cvtColor(
        processed_frame,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # MOTION DETECTION

    motion_mask = None

    if previous_gray is not None:

        difference = cv2.absdiff(
            previous_gray,
            blurred
        )

        _, motion_mask = cv2.threshold(
            difference,
            MOTION_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

    previous_gray = blurred.copy()

    # FIND MOTION CONTOURS

    motion_contours = []

    if motion_mask is not None:

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area >= MIN_MOTION_AREA:
                motion_contours.append(contour)

    # YOLO PERSON DETECTION

    results = model.track(
        processed_frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    current_people_count = 0
    max_motion_ratio = 0.0

    # ALARM ONLY IF A PERSON IS DETECTED

    if boxes is not None and len(boxes) > 0:

        current_people_count = len(boxes)

        total_people_detections += current_people_count

        # ALARM does not depend on background motion
        alarm = True

        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        if boxes.id is not None:
            ids = (
                boxes.id.cpu().numpy().astype(int)
            )
        else:
            ids = [-1] * current_people_count

        # PROCESS EACH PERSON

        for person_id, box, confidence in zip(
            ids,
            coordinates,
            confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            person_area = max(
                (x2 - x1) * (y2 - y1),
                1
            )

            # MEASURE MOTION INSIDE PERSON BOX

            moving_area = 0

            for contour in motion_contours:

                contour_area = cv2.contourArea(contour)

                cx, cy, cw, ch = cv2.boundingRect(contour)

                overlap_x1 = max(x1, cx)
                overlap_y1 = max(y1, cy)
                overlap_x2 = min(x2, cx + cw)
                overlap_y2 = min(y2, cy + ch)

                if (
                    overlap_x2 > overlap_x1
                    and overlap_y2 > overlap_y1
                ):

                    overlap_area = (
                        (overlap_x2 - overlap_x1)
                        * (overlap_y2 - overlap_y1)
                    )

                    moving_area += min(
                        contour_area,
                        overlap_area
                    )

            motion_ratio = moving_area / person_area

            max_motion_ratio = max(
                max_motion_ratio,
                motion_ratio
            )

            # MOVEMENT COUNTER FOR TRACKED PERSON

            if person_id != -1:

                if motion_ratio >= MOTION_RATIO:
                    movement_counter[person_id] = (
                        movement_counter.get(person_id, 0) + 1
                    )
                else:
                    movement_counter[person_id] = 0

                is_moving = (
                    movement_counter.get(person_id, 0)
                    >= MIN_MOVEMENT_FRAMES
                )

            else:
                is_moving = motion_ratio >= MOTION_RATIO

            # DRAW PERSON

            box_color = (0, 0, 255)

            if person_id != -1:
                label = (
                    f"PERSON {person_id} "
                    f"Conf: {confidence:.2f}"
                )
            else:
                label = f"PERSON Conf: {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

    else:
        # No person detected: no alarm
        alarm = False

    # SAVE METRICS

    current_time = frame_number / fps_video

    motion_results.append({
        "frame": frame_number,
        "time": current_time,
        "motion_ratio": max_motion_ratio,
        "alarm": int(alarm),
        "num_people": current_people_count
    })

    # DISPLAY ALARM

    if alarm:

        alarm_frames += 1

        cv2.putText(
            frame,
            "ALARM: PERSON DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (width - 1, height - 1),
            (0, 0, 255),
            5
        )

    else:

        cv2.putText(
            frame,
            "SYSTEM NORMAL - NO PERSON",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # FRAME NUMBER

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # WRITE VIDEO

    out.write(frame)

    # DISPLAY

    cv2.imshow(
        "Video Surveillance",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# RELEASE

cap.release()
out.release()
cv2.destroyAllWindows()


# METRICS

processing_time = time.time() - processing_start

processing_fps = (
    frame_number / processing_time
    if processing_time > 0
    else 0
)

alarm_percentage = (
    alarm_frames / frame_number * 100
    if frame_number > 0
    else 0
)


# SAVE CSV

motion_results_path = "results/motion_results.csv"

with open(
    motion_results_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "frame",
            "time",
            "motion_ratio",
            "alarm",
            "num_people"
        ]
    )

    writer.writeheader()
    writer.writerows(motion_results)


# PRINT RESULTS

print()
print("======================================")
print("ОБРАБОТКА ЗАВЕРШЕНА")
print("======================================")
print(f"Всего кадров: {frame_number}")
print(f"Время обработки: {processing_time:.2f} сек.")
print(f"FPS обработки: {processing_fps:.2f}")
print(f"Всего обнаружений людей: {total_people_detections}")
print(f"Кадров с ALARM: {alarm_frames}")
print(f"Доля кадров с ALARM: {alarm_percentage:.2f}%")
print()
print(f"Видео сохранено: {OUTPUT_PATH}")
print(f"Метрики сохранены: {motion_results_path}")
print("======================================")
