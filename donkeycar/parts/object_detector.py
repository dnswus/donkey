"""
Object Detector
"""

# import the necessary packages
import numpy as np
import time
import cv2
import os

class ObjectDetector():

    def __init__(self):
        self.detections = None
        self.image = None
        print("[INFO] loading model...")
        current_path = os.path.dirname(os.path.realpath(__file__))
        prototxt = os.path.join(current_path, 'MobileNetSSD_deploy.prototxt.txt')
        model = os.path.join(current_path, 'MobileNetSSD_deploy.caffemodel')
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.running = True

    def shutdown(self):
        self.running = False
        time.sleep(0.1)

    def update(self):
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        while(self.running):
            start_time = time.time()
            if self.image is not None:
                frame = self.image
                (fH, fW) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (fW, fH), 127.5)
                self.net.setInput(blob)
                self.detections = self.net.forward()
                if self.detections is not None:
                    # loop over the detections
                    for i in np.arange(0, self.detections.shape[2]):
                        confidence = self.detections[0, 0, i, 2]
                        if confidence < 0.7:
                            continue
                        idx = int(self.detections[0, 0, i, 1])
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                            confidence * 100)

                        print(label)
                        
            print(time.time() - start_time)

            time.sleep(0.01)

    def run_threaded(self, image=None):
        self.image = image
        return self.detections
