import tensorflow as tf
import cv2
import numpy as np
import sys

cap = cv2.VideoCapture(sys.argv[1])


def loadGraph(file):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

detection_graph = loadGraph("head10.pb")

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    rows = open("pi-deep-learning/synset_words.txt").read().strip().split("\n")
    foo = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    while True:
        running, image = cap.read()
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

        image_exp = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_exp})

        for i in range(int(num[0])):
            # print(boxes[0][i], scores[0][i], classes[0][i])
            if scores[0][i] < 0.3:
                continue

            box = boxes[0][i] * np.array([h, w, h, w])
            (startY, startX, endY, endX) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY),
                    (255,0,0), 2)
            y = startY - 15
            cv2.putText(image, str(scores[0][i]), (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.imshow("Output", image)
        cv2.waitKey(1)
