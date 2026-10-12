# Night Vision Detection and Alarm System

A computer vision project for detecting people and identifying suspicious motion in low-light video footage using YOLO11 and OpenCV.

## Overview

This project implements a video surveillance pipeline designed to detect people and identify potential motion-related security events in nighttime footage.

The system combines object detection with classical computer vision techniques to improve motion analysis under challenging lighting conditions.

## Features

- Person detection using YOLO11n and YOLO11s.
- Video processing with OpenCV.
- Low-light image enhancement using CLAHE.
- Motion detection using frame differencing.
- Noise reduction using Gaussian blur and morphological operations.
- Motion analysis using contours and area filtering.
- Alarm generation based on motion thresholds and consecutive detection frames.
- Performance evaluation using FPS, detection counts, and alarm statistics.
- Comparison of YOLO model variants and preprocessing configurations.

## Technologies

- Python
- OpenCV
- Ultralytics YOLO11
- NumPy
- Pandas
- Matplotlib

## Detection Pipeline

The processing pipeline consists of the following stages:

1. **Video input** — load a video file.
2. **Preprocessing** — convert frames to grayscale and optionally apply CLAHE.
3. **Noise reduction** — apply Gaussian blur.
4. **Motion detection** — calculate frame differences and apply thresholding.
5. **Morphological processing** — use opening and closing to reduce noise and improve motion masks.
6. **Contour analysis** — filter motion regions based on their area.
7. **Person detection** — run YOLO11 to detect people.
8. **Alarm decision** — evaluate motion ratio, minimum area, and consecutive moving frames.
9. **Output** — save the processed video, CSV metrics, and visualizations.

## Project Structure

```text
night-vision-detection/
├── dataset/
│   └── testing_videos/
│       └── night.avi
├── results/
│   ├── processed_video.mp4
│   ├── motion_results.csv
│   └── plots/
├── main.py
├── requirements.txt
└── README.md
```

*The structure above is illustrative. Adjust filenames to match the actual repository.*

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd night-vision-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Place the input video in `dataset/testing_videos/`.

Run the project using the actual Python entry-point file:

```bash
python main.py
```

The script processes the video and saves the output files according to the configured paths.

Make sure the YOLO model weights, such as `yolo11n.pt` or `yolo11s.pt`, are available in the project directory or can be downloaded by Ultralytics.

## Evaluation Metrics

The system can be evaluated using the following metrics:

| Metric | Description |
|---|---|
| Processing FPS | Number of video frames processed per second |
| Total frames | Number of frames processed |
| Person detections | Total person detections across the video |
| Alarm frames | Number of frames marked as alarm events |
| Alarm frame share | Percentage of frames marked as alarm events |
| Processing time | Total time required to process the video |
| False alarms | Alarm events that do not correspond to a real security threat |

Alarm frame share is calculated as:

\[
\text{Alarm Share} =
\frac{\text{Alarm Frames}}{\text{Total Frames}}\times100\%
\]

## Experiments

The project supports comparisons between:

- YOLO11n and YOLO11s.
- Processing with and without CLAHE.
- Detection performance and processing speed.
- Alarm frequency and false-positive behavior.

For a fair comparison, use the same input video, confidence threshold, motion parameters, and evaluation procedure. Change only the configuration being tested.

## Limitations

- Performance depends on lighting conditions, camera quality, and scene complexity.
- Low-light noise may cause false motion detections.
- YOLO detections do not independently establish whether a person poses a security threat.
- Alarm thresholds may require calibration for different environments.
- Processing speed depends on hardware and model size.

## Future Improvements

- Evaluate the system on multiple nighttime videos.
- Measure precision, recall, and false-positive rate using manually annotated ground truth.
- Compare detection results with and without CLAHE.
- Add real-time notifications when an alarm is triggered.
- Optimize inference performance for deployment.

## References

1. Ultralytics. [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
2. Ultralytics. [Predict Mode](https://docs.ultralytics.com/modes/predict/)
3. OpenCV. [Eroding and Dilating](https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html)
4. OpenCV. [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
5. OpenCV. [Image Filtering](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
6. OpenCV. [Histogram Equalization and CLAHE](https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html)

## Author

**Zhanel Marat**  
Data Science Student | Computer Vision | Machine Learning
