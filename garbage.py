
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

classes = []

with open(r"model\obj.names") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Object list:")
print(classes)

net = cv2.dnn.readNet(r"model/custom-yolov4-tiny-detector_last.weights", r"model/custom-yolov4-tiny-detector.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# Initialize variables to track TP, FP, FN
TP = 0  # True Positive
FP = 0  # False Positive
FN = 0  # False Negative
TN = 0  # True Negative (if applicable)

# Ground truth: define ground truth objects (you can replace this with actual labels)
ground_truth = ['bottle', 'can', 'plastic', 'wrapper']  # Modify based on your needs

# Start processing the frames
while True:
    success, frame = cap.read()
    classIds, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.3)

    detected_objects = []  # Track detected objects for this frame

    if len(classIds) != 0:
        for i in range(len(classIds)):
            classId = int(classIds[i])
            confidence = scores[i]
            box = boxes[i]
            x, y, w, h = box
            className = classes[classId - 1]
            detected_objects.append(className)

            # If the object is in the ground truth (True Positive)9+
            if className in ground_truth:
                TP += 1
            else:
                FP += 1  # False Positive: Detected but not in ground truth

            # Annotate the frame
            cv2.putText(frame, className.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Count False Negatives: If ground truth object was not detected
    for gt_object in ground_truth:
        if gt_object not in detected_objects:
            FN += 1  # If an object in ground truth was not detected

    # Calculate Precision, Recall, and Accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Display the metrics on the screen
    cv2.putText(frame, f"Precision: {precision:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Recall: {recall:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Image", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close any windows
cap.release()
cv2.destroyAllWindows()




