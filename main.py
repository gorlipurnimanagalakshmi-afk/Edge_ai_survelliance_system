import cv2
import time
from pathlib import Path
from tkinter import Tk, filedialog

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# =====================================================
# EDGE AI SURVEILLANCE SYSTEM
# Person + Object + Vehicle + Fire + Smoke
# Optimized for CPU processing
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

PERSON_MODEL_PATH = BASE_DIR / "models" / "yolo26s.pt"
FIRE_MODEL_PATH = BASE_DIR / "models" / "fire_smoke.pt"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "detected_video.mp4"


# =====================================================
# CONFIGURATION - OPTIMIZED SETTINGS
# =====================================================

CONFIDENCE = 0.35
FIRE_CONFIDENCE = 0.40

FIRE_IMAGE_SIZE = 640

PROCESS_EVERY_N_FRAMES = 10

SLICE_SIZE = 512
SLICE_OVERLAP = 0.10

DEVICE = "cpu"

PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
}


# =====================================================
# HELPER FUNCTION: KEEP BOXES INSIDE VIDEO
# =====================================================

def clamp_box(box, frame_width, frame_height):

    x1, y1, x2, y2 = box

    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))

    x2 = max(0, min(int(x2), frame_width - 1))
    y2 = max(0, min(int(y2), frame_height - 1))

    return x1, y1, x2, y2


# =====================================================
# PERSON TRACKING
# =====================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


class PersonTracker:

    def __init__(self):

        self.tracks = {}
        self.next_id = 1
        self.max_missed = 8

    def update(self, detections):

        person_detections = [
            d for d in detections
            if d["label"].lower() == "person"
        ]

        matched_tracks = set()
        matched_detections = set()

        # Match existing tracks
        for track_id, track in list(self.tracks.items()):

            best_iou = 0
            best_index = None

            for index, detection in enumerate(person_detections):

                if index in matched_detections:
                    continue

                iou = calculate_iou(
                    track["box"],
                    detection["box"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is not None and best_iou >= 0.30:

                detection = person_detections[best_index]

                track["box"] = detection["box"]
                track["missed"] = 0

                detection["person_id"] = track_id

                matched_tracks.add(track_id)
                matched_detections.add(best_index)

        # Create new tracks
        for index, detection in enumerate(person_detections):

            if index not in matched_detections:

                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = {
                    "box": detection["box"],
                    "missed": 0
                }

                detection["person_id"] = track_id

                matched_tracks.add(track_id)

        # Remove old tracks
        for track_id in list(self.tracks.keys()):

            if track_id not in matched_tracks:

                self.tracks[track_id]["missed"] += 1

            if (
                track_id in self.tracks
                and self.tracks[track_id]["missed"] > self.max_missed
            ):
                del self.tracks[track_id]

        return detections


# =====================================================
# CHECK MODEL FILES
# =====================================================

if not PERSON_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Person/object model not found: {PERSON_MODEL_PATH}"
    )

if not FIRE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fire/smoke model not found: {FIRE_MODEL_PATH}"
    )


# =====================================================
# LOAD PERSON AND OBJECT MODEL
# =====================================================

print("\nLoading person and object detection model...")

person_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(PERSON_MODEL_PATH),
    confidence_threshold=CONFIDENCE,
    device=DEVICE
)

print("Person/object model loaded successfully.")


# =====================================================
# LOAD FIRE AND SMOKE MODEL
# =====================================================

print("\nLoading fire and smoke model...")

fire_smoke_model = YOLO(str(FIRE_MODEL_PATH))

print("Fire/smoke model loaded successfully.")

print("Fire model classes:", fire_smoke_model.names)


# =====================================================
# SELECT VIDEO
# =====================================================

root = Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select Video for AI Surveillance",
    filetypes=[
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not video_path:
    print("No video selected.")
    raise SystemExit


# =====================================================
# OPEN VIDEO
# =====================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Unable to open selected video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Unable to create output video.")


# =====================================================
# INITIALIZE VARIABLES
# =====================================================

tracker = PersonTracker()

frame_number = 0

cached_detections = []

unique_person_ids = set()

max_persons_simultaneously = 0
max_vehicles_simultaneously = 0

found_classes = set()

fire_frames = 0
smoke_frames = 0

current_status = "NORMAL"

start_time = time.time()


# =====================================================
# CREATE RESIZABLE PREVIEW WINDOW
# =====================================================

cv2.namedWindow(
    "Edge AI Surveillance System",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Edge AI Surveillance System",
    PREVIEW_WIDTH,
    PREVIEW_HEIGHT
)


print("\nVideo processing started...")
print("Processing every", PROCESS_EVERY_N_FRAMES, "frames.")


# =====================================================
# MAIN VIDEO PROCESSING
# =====================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Run both models on sampled frames
        if (
            frame_number == 1
            or frame_number % PROCESS_EVERY_N_FRAMES == 0
        ):

            detections = []

            # -----------------------------------------
            # PERSON AND GENERAL OBJECT DETECTION
            # -----------------------------------------

            try:

                result = get_sliced_prediction(
                    frame,
                    person_model,
                    slice_height=SLICE_SIZE,
                    slice_width=SLICE_SIZE,
                    overlap_height_ratio=SLICE_OVERLAP,
                    overlap_width_ratio=SLICE_OVERLAP,
                    verbose=0
                )

                for prediction in result.object_prediction_list:

                    x1, y1, x2, y2 = prediction.bbox.to_xyxy()

                    box = clamp_box(
                        (x1, y1, x2, y2),
                        width,
                        height
                    )

                    label = str(
                        prediction.category.name
                    ).strip()

                    confidence = float(
                        prediction.score.value
                    )

                    detections.append({
                        "box": box,
                        "label": label,
                        "confidence": confidence,
                        "person_id": None
                    })

            except Exception as e:

                print("Person/object detection error:", e)

            # -----------------------------------------
            # FIRE AND SMOKE DETECTION
            # -----------------------------------------

            try:

                fire_results = fire_smoke_model.predict(
                    source=frame,
                    imgsz=FIRE_IMAGE_SIZE,
                    conf=FIRE_CONFIDENCE,
                    device=DEVICE,
                    verbose=False
                )[0]

                for box in fire_results.boxes:

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    box_coordinates = clamp_box(
                        (x1, y1, x2, y2),
                        width,
                        height
                    )

                    class_id = int(box.cls[0])

                    label = str(
                        fire_results.names[class_id]
                    ).strip()

                    confidence = float(box.conf[0])

                    detections.append({
                        "box": box_coordinates,
                        "label": label,
                        "confidence": confidence,
                        "person_id": None
                    })

            except Exception as e:

                print("Fire/smoke detection error:", e)

            # -----------------------------------------
            # UPDATE PERSON TRACKER
            # -----------------------------------------

            detections = tracker.update(detections)

            current_persons = [
                d for d in detections
                if d["label"].lower() == "person"
            ]

            current_person_count = len(current_persons)

            for detection in current_persons:

                if detection["person_id"] is not None:

                    unique_person_ids.add(
                        detection["person_id"]
                    )

            # -----------------------------------------
            # VEHICLE DETECTION
            # -----------------------------------------

            current_vehicles = [
                d for d in detections
                if d["label"].lower() in VEHICLE_CLASSES
            ]

            current_vehicle_count = len(current_vehicles)

            # -----------------------------------------
            # MAXIMUM SIMULTANEOUS COUNTS
            # -----------------------------------------

            max_persons_simultaneously = max(
                max_persons_simultaneously,
                current_person_count
            )

            max_vehicles_simultaneously = max(
                max_vehicles_simultaneously,
                current_vehicle_count
            )

            # -----------------------------------------
            # RECORD DETECTED CLASSES
            # -----------------------------------------

            for detection in detections:

                found_classes.add(detection["label"])

            cached_detections = detections

            # -----------------------------------------
            # FIRE AND SMOKE FRAME COUNTS
            # -----------------------------------------

            fire_present = any(
                d["label"].lower() == "fire"
                for d in detections
            )

            smoke_present = any(
                d["label"].lower() == "smoke"
                for d in detections
            )

            if fire_present:
                fire_frames += 1

            if smoke_present:
                smoke_frames += 1

            # -----------------------------------------
            # DETERMINE ALERT STATUS
            # -----------------------------------------

            if fire_present:
                current_status = "FIRE DETECTED"

            elif smoke_present:
                current_status = "SMOKE DETECTED"

            elif current_person_count > 0:
                current_status = "PERSON DETECTED"

            elif current_vehicle_count > 0:
                current_status = "VEHICLE DETECTED"

            else:
                current_status = "NORMAL"

        # =================================================
        # DRAW DETECTION BOXES
        # =================================================

        for detection in cached_detections:

            x1, y1, x2, y2 = detection["box"]

            label = detection["label"]
            confidence = detection["confidence"]

            label_lower = label.lower()

            if label_lower == "fire":
                color = (0, 0, 255)

            elif label_lower == "smoke":
                color = (128, 128, 128)

            elif label_lower == "person":
                color = (0, 255, 0)

            elif label_lower in VEHICLE_CLASSES:
                color = (255, 180, 0)

            else:
                color = (255, 255, 0)

            person_id = detection.get("person_id")

            if person_id is not None:
                display_label = f"Person ID {person_id}"
            else:
                display_label = label

            display_label += f" {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                display_label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # =================================================
        # STATUS AND COUNTER OVERLAY
        # =================================================

        persons_now = sum(
            1 for d in cached_detections
            if d["label"].lower() == "person"
        )

        vehicles_now = sum(
            1 for d in cached_detections
            if d["label"].lower() in VEHICLE_CLASSES
        )

        if current_status == "FIRE DETECTED":
            status_color = (0, 0, 255)

        elif current_status == "SMOKE DETECTED":
            status_color = (0, 165, 255)

        elif current_status == "PERSON DETECTED":
            status_color = (0, 255, 0)

        elif current_status == "VEHICLE DETECTED":
            status_color = (255, 180, 0)

        else:
            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            f"STATUS: {current_status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"Persons: {persons_now}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Vehicles: {vehicles_now}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # =================================================
        # SAVE ORIGINAL RESOLUTION VIDEO
        # =================================================

        writer.write(frame)

        # =================================================
        # RESIZE ONLY THE PREVIEW
        # =================================================

        scale = min(
            PREVIEW_WIDTH / width,
            PREVIEW_HEIGHT / height
        )

        preview_width = max(1, int(width * scale))
        preview_height = max(1, int(height * scale))

        preview = cv2.resize(
            frame,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(
            "Edge AI Surveillance System",
            preview
        )

        # =================================================
        # TERMINAL PROGRESS - EVERY 30 FRAMES
        # =================================================

        if frame_number % 30 == 0:

            print(
                f"Progress: {frame_number}/{total_frames} | "
                f"Persons: {persons_now} | "
                f"Vehicles: {vehicles_now} | "
                f"Status: {current_status}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("Processing stopped by user.")
            break


finally:

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# =====================================================
# FINAL REPORT
# =====================================================

elapsed_time = time.time() - start_time

print("\n")
print("=" * 50)
print("          SURVEILLANCE FINAL REPORT")
print("=" * 50)

print("Video frames processed:", frame_number)

print(
    "Estimated distinct persons tracked:",
    len(unique_person_ids)
)

print(
    "Maximum persons detected simultaneously:",
    max_persons_simultaneously
)

print(
    "Maximum vehicles detected simultaneously:",
    max_vehicles_simultaneously
)

print("\nObject classes detected:")

if found_classes:

    for class_name in sorted(found_classes):
        print("-", class_name)

else:

    print("No objects detected.")

print("\nFire detection sampled frames:", fire_frames)
print("Smoke detection sampled frames:", smoke_frames)

print(
    "\nProcessing time:",
    round(elapsed_time, 2),
    "seconds"
)

print("\nOutput video saved at:")
print(OUTPUT_VIDEO)

print("=" * 50)
print("\nSurveillance processing completed.")