import cv2
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

# GPIO pin setup
PROPELLER_IN1 = 17
PROPELLER_IN2 = 27
PROPELLER_EN = 18   # PWM-capable pin

CONVEYOR_IN1 = 22
CONVEYOR_IN2 = 23
CONVEYOR_EN = 19    # PWM-capable pin

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor direction control pins
GPIO.setup(PROPELLER_IN1, GPIO.OUT)
GPIO.setup(PROPELLER_IN2, GPIO.OUT)
GPIO.setup(PROPELLER_EN, GPIO.OUT)

GPIO.setup(CONVEYOR_IN1, GPIO.OUT)
GPIO.setup(CONVEYOR_IN2, GPIO.OUT)
GPIO.setup(CONVEYOR_EN, GPIO.OUT)

# PWM Setup (1000 Hz frequency)
propeller_pwm = GPIO.PWM(PROPELLER_EN, 1000)
conveyor_pwm = GPIO.PWM(CONVEYOR_EN, 1000)

propeller_pwm.start(0)
conveyor_pwm.start(0)

def start_motors(speed=70):  # speed can be 0 to 100 (percent)
    # Propeller forward
    GPIO.output(PROPELLER_IN1, GPIO.HIGH)
    GPIO.output(PROPELLER_IN2, GPIO.LOW)
    propeller_pwm.ChangeDutyCycle(speed)

    # Conveyor forward
    GPIO.output(CONVEYOR_IN1, GPIO.HIGH)
    GPIO.output(CONVEYOR_IN2, GPIO.LOW)
    conveyor_pwm.ChangeDutyCycle(speed)

    print(f"[INFO] Motors started at speed: {speed}%")

def stop_motors():
    propeller_pwm.ChangeDutyCycle(0)
    conveyor_pwm.ChangeDutyCycle(0)

    GPIO.output(PROPELLER_IN1, GPIO.LOW)
    GPIO.output(PROPELLER_IN2, GPIO.LOW)
    GPIO.output(CONVEYOR_IN1, GPIO.LOW)
    GPIO.output(CONVEYOR_IN2, GPIO.LOW)

    print("[INFO] Motors stopped")

def detect_garbage():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()

    net = cv2.dnn.readNet('model/custom-yolov4-tiny-detector_last.weights', 'model/custom-yolov4-tiny-detector.cfg')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255)

    with open('model/obj.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    garbage_classes = {"plastic", "can", "wrapper", "bottle"}

    while True:
        frame = picam2.capture_array()
        classIds, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.3)

        detected_objects = []

        if len(classIds) != 0:
            for i in range(len(classIds)):
                classId = int(classIds[i])
                box = boxes[i]
                x, y, w, h = box
                className = classes[classId - 1]
                detected_objects.append(className)

                cv2.putText(frame, className, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if any(obj in garbage_classes for obj in detected_objects):
                print("Garbage detected:", detected_objects)
                start_motors(speed=70)  # You can change speed dynamically
            else:
                stop_motors()
        else:
            stop_motors()

        resized_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Detection", resized_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    propeller_pwm.stop()
    conveyor_pwm.stop()
    GPIO.cleanup()
    print("[INFO] GPIO cleanup done")

if __name__ == "__main__":
    detect_garbage()
