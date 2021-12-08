import cv2
import json
import numpy as np


class LaneDetection:
    def __init__(self):
        self.currentFrame = None
        self.originalFrame = None
        self.linesDetected = None
        
        self.lastLeftLane = None
        self.lastRightLane = None
        
        self.leftCords = None
        self.rightCords = None

        self.prevLeft = []
        self.prevRight = []
        self.lane_change = ''
        self.frame = 0
    
    def __getVideoCapture(self, fileName):
        """ return the video capture object 
            if the object is already taken return None """
            
        capture = cv2.VideoCapture(fileName)
    
        if capture.isOpened() == False:
            print("video is opened")
            return None
        
        return capture
    
    def __getVideoWriter(self, sampleImage, outputFileName):
        """ return the video writer object """
        
        height, width, layers = sampleImage.shape
        videoWriter = cv2.VideoWriter(outputFileName, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
        return videoWriter
        
    def __convertToGrey(self):
        """ convert the currentFrame into grey scale """
        
        self.currentFrame = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)
    
    def __removeNoise(self, d = 5, sigma = 0):
        """ remove noise from the curretnFrame using guassin / bilateralFilter """
        
        self.currentFrame = cv2.GaussianBlur(self.currentFrame, (d, d), sigma)
        # self.currentFrame = cv2.bilateralFilter(self.currentFrame, d, sigma_r, sigma_s)
    
    def __dilateFrame(self):
        """ dilate the currentFrame """
        
        kernel = np.zeros((5,5),dtype=np.uint8)
        kernel[2,:] = 1
        kernel[:,2] = 1
        self.currentFrame = cv2.dilate(self.currentFrame, kernel, iterations = 1)
    
    def __detectEdges(self, lowThreshold = 50, highThreshold = 150): # 300, 700
        """ detect the edges of the currentFrame using Canny """
        
        self.currentFrame = cv2.Canny(self.currentFrame, lowThreshold, highThreshold)
    
    def __convertImageToBinary(self):
        """ set the image to 0 or 1 only and by that converting it to binary image color """
        
        self.currentFrame = cv2.threshold(self.currentFrame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    def __cropImage(self):
        """ set the currentFrame to the road in a triangle shape """
        
        height = self.currentFrame.shape[0]
        left_x = 440
        right_x = 1500
        topMiddlePoint = ((right_x + left_x) / 2, 250)
        buttomRightPoint = (right_x, height)
        buttomLeftPoint = (left_x, height)
        polygons = np.array([[buttomLeftPoint, buttomRightPoint, topMiddlePoint]]).astype(np.int)

        new_left_point = ((buttomLeftPoint[0] + topMiddlePoint[0]) / 2, (buttomLeftPoint[1] + topMiddlePoint[1]) / 2)
        new_right_point = ((buttomRightPoint[0] + topMiddlePoint[0]) / 2, (buttomRightPoint[1] + topMiddlePoint[1]) / 2)
        polygons2 = np.array([[new_left_point, new_right_point, topMiddlePoint]]).astype(np.int)

        mask = np.zeros_like(self.currentFrame)
        cv2.fillPoly(mask, polygons, 255)
        cv2.fillPoly(mask, polygons2, 0)
        self.currentFrame = cv2.bitwise_and(self.currentFrame, mask)
    
    def __createLines(self, maxDistance = 200, maxAngle = 1.2):
        """ draw the lines detected on the original image """
        
        if self.linesDetected is None or type(self.linesDetected) is None:
            return None
        
        lines_image = np.zeros_like(self.originalFrame)
        
        try:
            for x1, y1, x2, y2 in self.linesDetected:
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                # cv2.line(self.originalFrame, (x1, y1), (x2, y2), (255, 0, 0), 10)
        except ValueError:
            print(self.linesDetected)
            for item in self.linesDetected:
                for line in item:
                    x1, y1, x2, y2 = line
                    cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        
        self.originalFrame = cv2.addWeighted(self.originalFrame, 0.8, lines_image, 1, 1)
    
    def __getCordinates(self, parameters):
        """ return the coedinates of the slope and intercept given using linear algebra """

        if parameters is None:
            return np.array([0, 0, 0, 0])

        slope, intercept = parameters
        y1 = self.currentFrame.shape[0]
        y2 = int(y1 * (3 / 5))
        
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    
    def __detectSides(self):
        """ detect right and left lanes and set linesDetected to the avreage of each """
        
        leftLane = []
        rightLane = []

        self.frame += 1  # DEBUG

        if self.linesDetected is None or type(self.linesDetected) is None:
            return None
        
        for currentLine in self.linesDetected:
            x1, y1, x2, y2 = currentLine.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if -0.2 < slope < 0.2:
                continue
            
            if slope < 0:
                leftLane.append((slope, intercept)) 
            else:
                rightLane.append((slope, intercept))
        
        
        leftAvrage = np.average(leftLane, axis = 0)
        rightAvrage = np.average(rightLane, axis = 0)

        if self.frame % 30 == 0:
            # print(f'Frame {self.frame} - Slope: {slope}, Intercept: {intercept}')  # DEBUG
            print(f'Frame {self.frame} - {self.prevLeft}')  # DEBUG

        if type(leftAvrage) is np.float64:
            leftAvrage = self.lastLeftLane
        else:
            self.lastLeftLane = leftAvrage
            
        if type(rightAvrage) is np.float64:
            rightAvrage = self.lastRightLane
        else:
            self.lastRightLane = rightAvrage
        
        self.leftCords = self.__getCordinates(leftAvrage)
        self.rightCords = self.__getCordinates(rightAvrage)

        self.prevLeft.append(self.leftCords[0])
        if len(self.prevLeft) > 10:
            _ = self.prevLeft.pop(0)
        self.prevRight.append(self.rightCords[0])
        if len(self.prevLeft) > 10:
            _ = self.prevLeft.pop(0)

        self.linesDetected = np.array([self.leftCords, self.rightCords])
    
    def __drawMiddleLine(self):
        """ draw a line in the middle of the image, and check if the diffrence between the lines are towrad the right or left """
        
        if self.rightCords is None:
            return None
        
        middlePointTop = (int(self.originalFrame.shape[1] / 2), self.originalFrame.shape[0] - 50)
        middlePointBottom = (int(self.originalFrame.shape[1] / 2), self.originalFrame.shape[0] - 30)
        middlePointText = (int(self.originalFrame.shape[1] / 2) - 200, 200)#self.originalFrame.shape[0] - 10)
        # diffrence = self.rightCords[0] - self.leftCords[0]
        
        cv2.line(self.originalFrame, middlePointBottom, middlePointTop, (255, 0, 0), 10)

        middle_x = int(self.originalFrame.shape[1] / 2)

        left_th = 100
        count_left = 0
        for left_x in self.prevLeft:
            if abs(left_x - middle_x) < left_th:
                count_left += 1
        if count_left >= 5 and self.lane_change != 'Right':
            print('Going Left!')
            self.lane_change = 'Left'
        elif self.lane_change == 'Left':
            print('Finish Going Left!')
            self.lane_change = ''


        right_th = 100
        count_right = 0
        for right_x in self.prevRight:
            if abs(right_x - middle_x) < right_th:
                count_right += 1
        if count_right >= 5 and self.lane_change != 'Left':
            print('Going Right!')
            self.lane_change = 'right'
        elif self.lane_change == 'Right':
            print('Finish Going Right!')
            self.lane_change = ''

        # if diffrence < 640:
        #     pass # print("diffrence smaller than 640, maybe go left")
        # elif diffrence > 750:
        #     pass # print("diffrence biiger than 740, maybe go Right")

        cv2.putText(self.originalFrame, self.lane_change,
                    middlePointText, cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255  ), 10, 2)
    
        
    def __detectLines(self, threshold = 100, radiusStep = 2, angleStep = np.pi / 180.0):  # 250
        """ detect lines in the image and show it """
        
        self.linesDetected = cv2.HoughLinesP(self.currentFrame, radiusStep, angleStep, threshold, np.array([]), minLineLength = 40, maxLineGap = 5)
        # self.linesDetected = cv2.HoughLines(self.currentFrame, radiusStep, angleStep, threshold)
        self.__detectSides()
        self.__createLines()
        self.__drawMiddleLine()
    
    def __showCurrentImage(self):
        """ show the original image """
        
        cv2.imshow('Frame', self.originalFrame)
        # cv2.imshow('Frame', self.currentFrame)
    
    def __quitDetected(self):
        """ check if the user wants to quit """
        
        return cv2.waitKey(25) & 0xFF == ord('q')
    
    def __detectLinesInImage(self, image=None):
        """ detect lines in one image per time """
        
        if image is not None:
            self.originalFrame = image
            self.currentFrame = self.originalFrame.copy()
        
        self.__convertToGrey()
        self.__removeNoise()
        self.__detectEdges()
        # self.__dilateFrame()
        self.__cropImage()
        self.__detectLines()
        self.__showCurrentImage()
        
    def detect(self, videoFileName, outputFileName = "output.avi"):
        """ main function to detect lines in the video and show it """
        
        videoCapture = self.__getVideoCapture(videoFileName)
        videoWriter = None
        
        if videoCapture is not None:
            while videoCapture.isOpened():
                ret, self.originalFrame = videoCapture.read()
                
                if not ret or self.__quitDetected():
                    break
                
                if videoWriter is None:
                    videoWriter = self.__getVideoWriter(self.originalFrame, outputFileName)
                
                self.currentFrame = self.originalFrame.copy()
                
                self.__detectLinesInImage()
                
                videoWriter.write(self.originalFrame)

            videoCapture.release()
            videoWriter.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    # LaneDetection().detect("dashCam.mp4")
    # LaneDetection().detectLinesInImage(cv2.imread(config['test_image']))
    # LaneDetection().detect(config['test_video'])
    LaneDetection().detect('Tests/test_video_Trim.mp4')