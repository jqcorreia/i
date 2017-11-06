import cv2
import json
import numpy as np
import tensorflow as tf

class DetectorMobilenetSSD:
    def __init__(self, frozen_path):
        self.graph = self.loadGraph(frozen_path)
        self.sess = tf.Session(graph=self.graph)

        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')



    def loadGraph(self, frozen_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        # frame = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
        frame_exp = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_exp})

        return (boxes, scores, classes, num)

