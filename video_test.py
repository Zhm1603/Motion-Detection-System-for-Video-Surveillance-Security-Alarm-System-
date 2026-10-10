import os
import csv
import cv2
import time
from ultralytics import YOLO



VIDEO_PATH = "dataset/testing_videos/01.avi"
OUTPUT_PATH = "results_11s/01_yolo11s_clahe.mp4"

MODEL_PATH = "yolo11s.pt"

CONFIDENCE = 0.4

MOTION_THRESHOLD = 25
MIN_MOTION_AREA = 150

MOTION_RATIO = 0.02
MIN_MOVEMENT_FRAMES = 3

USE_CLAHE = True


# CREATE RESULTS FOLDER

os.makedirs(
    "results",
    exist_ok=True
)


# LOAD MODEL

model = YOLO(
    MODEL_PATH
)


# OPEN VIDEO


cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("Не удалось открыть видео")

    exit()


fps_video = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

if fps_video <= 0:

    fps_video = 25



# OUTPUT VIDEO


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

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

# Data for visualization
motion_results = []



# START

print("======================================")
print("Система видеонаблюдения запущена")
print("======================================")

print(
    "Видео:",
    VIDEO_PATH
)

print(
    "Модель:",
    MODEL_PATH
)

print(
    "CLAHE:",
    USE_CLAHE
)

print(
    "Motion ratio threshold:",
    MOTION_RATIO
)

print(
    "Минимальное количество кадров:",
    MIN_MOVEMENT_FRAMES
)

print()

print(
    "Для остановки нажмите Q"
)

print()




while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # IMAGE PREPROCESSING


    processed_frame = frame.copy()


    # CLAHE

    if USE_CLAHE:

        gray_for_clahe = cv2.cvtColor(
            processed_frame,
            cv2.COLOR_BGR2GRAY
        )

        enhanced_gray = clahe.apply(
            gray_for_clahe
        )

        processed_frame = cv2.cvtColor(
            enhanced_gray,
            cv2.COLOR_GRAY2BGR
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



        # MORPHOLOGICAL PROCESSING


        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        # Opening
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Closing
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

            area = cv2.contourArea(
                contour
            )

            if area >= MIN_MOTION_AREA:

                motion_contours.append(
                    contour
                )


    # YOLO TRACKING


    results = model.track(
        processed_frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )



    # VARIABLES FOR CURRENT FRAME


    alarm = False

    max_motion_ratio = 0.0

    current_people_count = 0


    # PROCESS DETECTED PEOPLE

    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        ids = (
            boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        coordinates = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )


        current_people_count = len(
            ids
        )

        total_people_detections += (
            current_people_count
        )


        # PROCESS EACH PERSON

        for person_id, box, confidence in zip(
            ids,
            coordinates,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # Keep coordinates inside frame

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width,
                x2
            )

            y2 = min(
                height,
                y2
            )


            # PERSON AREA

            person_area = max(
                (x2 - x1) * (y2 - y1),
                1
            )


            # MOTION AREA INSIDE PERSON

            moving_area = 0


            for contour in motion_contours:

                contour_area = cv2.contourArea(
                    contour
                )


                cx, cy, cw, ch = cv2.boundingRect(
                    contour
                )


                contour_x1 = cx
                contour_y1 = cy

                contour_x2 = cx + cw
                contour_y2 = cy + ch


                # INTERSECTION

                overlap_x1 = max(
                    x1,
                    contour_x1
                )

                overlap_y1 = max(
                    y1,
                    contour_y1
                )

                overlap_x2 = min(
                    x2,
                    contour_x2
                )

                overlap_y2 = min(
                    y2,
                    contour_y2
                )


                if (
                    overlap_x2 > overlap_x1
                    and
                    overlap_y2 > overlap_y1
                ):

                    overlap_width = (
                        overlap_x2
                        -
                        overlap_x1
                    )

                    overlap_height = (
                        overlap_y2
                        -
                        overlap_y1
                    )

                    overlap_area = (
                        overlap_width
                        *
                        overlap_height
                    )


                    moving_area += min(
                        contour_area,
                        overlap_area
                    )


            # MOTION RATIO

            motion_ratio = (
                moving_area
                /
                person_area
            )


            # Save maximum ratio for this frame

            max_motion_ratio = max(
                max_motion_ratio,
                motion_ratio
            )


            # MOVEMENT COUNTER

            if motion_ratio >= MOTION_RATIO:

                movement_counter[person_id] = (
                    movement_counter.get(
                        person_id,
                        0
                    )
                    + 1
                )

            else:

                movement_counter[person_id] = 0



            # MOVING PERSON

            is_moving = (
                movement_counter.get(
                    person_id,
                    0
                )
                >= MIN_MOVEMENT_FRAMES
            )



            # DRAW PERSON

            if is_moving:

                box_color = (
                    0,
                    0,
                    255
                )

                label = (
                    f"PERSON {person_id} MOVING"
                )

                alarm = True

            else:

                box_color = (
                    0,
                    255,
                    0
                )

                label = (
                    f"PERSON {person_id}"
                )


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
                (
                    x1,
                    max(
                        y1 - 10,
                        25
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )


    # SAVE METRICS FOR CURRENT FRAME

    current_time = (
        frame_number
        /
        fps_video
    )


    motion_results.append(
        {
            "frame": frame_number,
            "time": current_time,
            "motion_ratio": max_motion_ratio,
            "alarm": int(alarm),
            "num_people": current_people_count
        }
    )


    # ALARM

    if alarm:

        alarm_frames += 1


        cv2.putText(
            frame,
            "ALARM",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            4
        )


        cv2.putText(
            frame,
            "MOTION DETECTED",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
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
            "SYSTEM NORMAL",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


    # FRAME NUMBER

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (
            20,
            height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )



    # WRITE VIDEO

    out.write(
        frame
    )



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

processing_time = (
    time.time()
    -
    processing_start
)


processing_fps = (
    frame_number
    /
    processing_time
    if processing_time > 0
    else 0
)


alarm_percentage = (
    alarm_frames
    /
    frame_number
    *
    100
    if frame_number > 0
    else 0
)


# SAVE CSV

motion_results_path = (
    "results/motion_results.csv"
)


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

    writer.writerows(
        motion_results
    )



print()

print(
    "======================================"
)

print(
    "ОБРАБОТКА ЗАВЕРШЕНА"
)

print(
    "======================================"
)

print(
    f"Всего кадров: {frame_number}"
)

print(
    f"Время обработки: "
    f"{processing_time:.2f} сек."
)

print(
    f"FPS обработки: "
    f"{processing_fps:.2f}"
)

print(
    f"Всего обнаружений людей: "
    f"{total_people_detections}"
)

print(
    f"Кадров с ALARM: "
    f"{alarm_frames}"
)

print(
    f"Доля кадров с ALARM: "
    f"{alarm_percentage:.2f}%"
)

print()

print(
    f"Видео сохранено: "
    f"{OUTPUT_PATH}"
)

print(
    f"Метрики сохранены: "
    f"{motion_results_path}"
)

print(
    "======================================"
)