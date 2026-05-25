import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("C:/Users/k8254/phone_detection/runs/detect/phone_detection_model-3/weights/best.pt")

PHONE_CLASS_NAME = "phone"
CONFIDENCE_LIMIT = 0.5

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

print("Webcam started.")
print("Press q to quit.")


def draw_warning_overlay(frame):
    height, width, _ = frame.shape

    border_thickness = 18
    red = (0, 0, 255)

    cv2.rectangle(
        frame,
        (0, 0),
        (width - 1, height - 1),
        red,
        border_thickness
    )

    center_x = width // 2
    center_y = height // 2

    triangle_size = 90

    point1 = (center_x, center_y - triangle_size)
    point2 = (center_x - triangle_size, center_y + triangle_size)
    point3 = (center_x + triangle_size, center_y + triangle_size)

    triangle_points = [point1, point2, point3]

    triangle_points_np = np.array(triangle_points, np.int32)

    cv2.polylines(
        frame,
        [triangle_points_np],
        isClosed=True,
        color=red,
        thickness=8
)

    cv2.line(
        frame,
        (center_x, center_y - 30),
        (center_x, center_y + 45),
        red,
        12
    )

    cv2.circle(
        frame,
        (center_x, center_y + 80),
        8,
        red,
        -1
    )

    cv2.putText(
        frame,
        "PHONE DETECTED",
        (center_x - 190, center_y + 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        red,
        3
    )
    


while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam frame.")
        break

    results = model(frame, verbose=False)

    phone_detected = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name == PHONE_CLASS_NAME and confidence >= CONFIDENCE_LIMIT:
                phone_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), red := (0, 0, 255), 2)

                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    red,
                    2
                )

    if phone_detected:
        draw_warning_overlay(frame)

    cv2.imshow("Smartphone Detection Study Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()