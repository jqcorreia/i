from flask import Flask, render_template, Response
import cv2
import sys
import queue
from wtforms import BooleanField

from flask_wtf import Form

class Configuration(Form):
    blur = BooleanField("Blur")

class VideoServer:
    def __init__(self, queue):
        self.app = Flask(__name__, template_folder=".")
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.queue = queue
        with self.app.app_context():
            self.form = Configuration(csrf_enabled=False)

    def index(self):
        return render_template('index.html', form=self.form)

    def gen(self):
        while True:
            print(self.form.blur)

            try:
                frame = self.queue.get()
            except:
                continue
            _, jpeg = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    def video_feed(self):
        return Response(self.gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    def start(self):
        self.app.run(host='0.0.0.0')

if __name__ == '__main__':
    vs = VideoServer(sys.argv[1])
    vs.start()

