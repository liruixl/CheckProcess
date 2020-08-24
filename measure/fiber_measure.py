import cv2

'''
1 实验得到HSV范围、纤维丝长度、宽度等范围
2 度量Wrapper，开放接口
'''


class FiberMeasure:

    def __init__(self):

        self.lower_hsv = []
        self.high_hsv = []

        self.length = (20, 60)
        self.width = (0.0, 3.5)

        pass

    def measure(self, img):
        pass