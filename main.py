import os
import pandas as pd
import matplotlib.pyplot as plt




RESULTS_DIR = "results"

CSV_FILE = os.path.join(
    RESULTS_DIR,
    "motion_results.csv"
)

MOTION_THRESHOLD = 0.02


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)



def plot_motion_ratio(df):

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["time"],
        df["motion_ratio"],
        label="Motion Ratio"
    )

    plt.axhline(
        MOTION_THRESHOLD,
        linestyle="--",
        label=f"Threshold = {MOTION_THRESHOLD}"
    )

    alarm_df = df[
        df["alarm"] == 1
    ]

    if not alarm_df.empty:

        plt.scatter(
            alarm_df["time"],
            alarm_df["motion_ratio"],
            label="Alarm",
            zorder=5
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Motion Ratio")
    plt.title("Motion Ratio and Alarm Threshold")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "motion_ratio.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 2. NUMBER OF PEOPLE
# ============================================================

def plot_people_count(df):

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["time"],
        df["num_people"],
        label="Detected people"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Number of people")
    plt.title("Number of Detected People Over Time")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "people_count.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 3. ALARM OVER TIME
# ============================================================

def plot_alarm_over_time(df):

    plt.figure(figsize=(12, 4))

    plt.step(
        df["time"],
        df["alarm"],
        where="post"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Alarm")

    plt.yticks(
        [0, 1],
        ["Normal", "Alarm"]
    )

    plt.title("Alarm Activation Over Time")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "alarm_over_time.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 4. SUMMARY
# ============================================================

def print_summary(df):

    total_frames = len(df)

    alarm_frames = int(
        df["alarm"].sum()
    )

    max_motion = df[
        "motion_ratio"
    ].max()

    average_motion = df[
        "motion_ratio"
    ].mean()

    max_people = int(
        df["num_people"].max()
    )

    alarm_percentage = (
        alarm_frames
        / total_frames
        * 100
        if total_frames > 0
        else 0
    )

    print()
    print("======================================")
    print("VISUALIZATION SUMMARY")
    print("======================================")

    print(f"Total frames: {total_frames}")
    print(f"Maximum motion ratio: {max_motion:.4f}")
    print(f"Average motion ratio: {average_motion:.4f}")
    print(f"Maximum detected people: {max_people}")
    print(f"Alarm frames: {alarm_frames}")
    print(f"Alarm percentage: {alarm_percentage:.2f}%")

    print("======================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not os.path.exists(CSV_FILE):

        print(
            "Ошибка: файл с результатами не найден:"
        )

        print(CSV_FILE)
        print()
        print("Сначала запусти main.py")

        exit()


    # Read CSV

    df = pd.read_csv(
        CSV_FILE
    )


    # Check columns

    required_columns = [
        "frame",
        "time",
        "motion_ratio",
        "alarm",
        "num_people"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "В CSV отсутствуют столбцы:"
        )

        print(missing_columns)

        exit()


    # Create visualizations

    print()
    print("Создание визуализаций...")
    print()


    plot_motion_ratio(df)

    plot_people_count(df)

    plot_alarm_over_time(df)


    # Print summary

    print_summary(df)


    print()
    print("Все графики сохранены в папку results.")