import ujson
import argparse
import cv2
import numpy as np
import datetime

from detector import DetectorMobilenetSSD
from stream import VideoStream
from processor import OccupancyProcessor, PointsProcessor

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file location")
parser.add_argument("-s", "--stream", help="Stream to be processed")
parser.add_argument("-f", "--confidence", help="Minimum level of confidence", type=float, default=0.5)
parser.add_argument("-k", "--skip", help="Number of frames to skip", type=float, default=0)
args = parser.parse_args()

def read_config(path):
    f = open(path, "r")
    conf = ujson.load(f)

    return conf

def write_to_file(topic, data):
    f = open(topic, "a")

    f.write(ujson.dumps(data) + "\n")

def main():
    config = read_config(args.config)
    stream_urls = []
    streams = {}
    detector = DetectorMobilenetSSD(config['model'])

    # In case there is a stream URL explicitly passed via CLI
    if args.stream == None:
        stream_urls = config['streams']
    else:
        stream_urls = [ args.stream ]

    for url in stream_urls:
        stream = VideoStream(url, url) # use URL as stream name for now
        streams[url] = (stream,
            [OccupancyProcessor(args.confidence),
            PointsProcessor(args.confidence)])

    # Main processing loop
    while True:
        for _, (stream, processors) in streams.items():
            _, image = stream.read()
            (h, w) = image.shape[:2]
            (boxes, scores, classes, num) = detector.detect(image)

            for proc in processors:
                proc.update((image, boxes, scores, classes, num))

            if stream.seconds_elapsed >= config['report_interval']:
                for i, proc in enumerate(processors):
                    (topic, msg) = proc.serialize()

                    proc.reset()
                    processors[i] = proc
                    write_to_file(topic, msg)

                stream.seconds_elapsed = 0
                stream.num_frames = 0

            for i in range(int(num[0])):
                if scores[0][i] < args.confidence:
                    continue

                box = boxes[0][i] * np.array([h, w, h, w])
                (startY, startX, endY, endX) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY),
                        (255,0,0), 2)
                y = startY - 15
                cv2.putText(image, str(scores[0][i]), (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            cv2.imshow(stream.name, image)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
