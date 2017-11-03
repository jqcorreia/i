from darkflow.net.build import TFNet
import cv2
import json
import np

class DetectorYOLO:
    def __init__(self):
        options = {
            "model": "cfg/tiny-yolo.cfg",
            "load": "bin/tiny-yolo.weights",
            'verbalise': True,
            #"threshold": 0.1
        }
        self.tfnet = TFNet(options)

    def detect(self, image):
        result = self.tfnet.return_predict(image)

        l = [ (det['topleft']['x'], det['topleft']['y'], det['bottomright']['x'] - det['topleft']['x'], det['bottomright']['y'] - det['topleft']['y']) for det in result ]
        return l


class DetectorMobileNetSSD:
    def __init__(self, proto, model):
        self.net = cv2.dnn.readNetFromCaffe(proto, model)


    def detect(self, image):
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (255,255)), 0.007843,
            (255, 255), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            class_id = int(detections[0, 0, i, 1])


