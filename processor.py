class OccupancyProcessor():
    def __init__(self, min_confidence):
        self.min_confidence = min_confidence
        self.counts = []

    def update(self, data):
        (boxes, scores, classes, num) = data
        count = 0
        for i in range(int(num[0])):
            if scores[0][i] < self.min_confidence:
                continue
            count += 1

        self.counts.append(count)

    def reset(self):
        print(max(self.counts))
        self.counts = []
