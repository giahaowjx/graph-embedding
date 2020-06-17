import cv2

class ColorPreprocessor:
    def __init__(self, color=cv2.COLOR_BGR2GRAY):
        self.color = color

    def preprocess(self, image):
        return cv2.cvtColor(image, self.color)