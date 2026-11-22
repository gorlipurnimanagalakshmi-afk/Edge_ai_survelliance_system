from ultralytics import YOLO

print("Downloading general detection model...")

detector = YOLO("yolo26s.pt")

print("General detection model ready!")

print("Downloading lightweight fire/smoke base model...")

fire_model = YOLO("yolo26n.pt")

print("Base model ready!")