import cv2
import numpy as np
from enum import Enum


class LaneDetection:
    class __Direction(Enum):
        LEFT    = 1
        RIGHT   = 2
        STRIGHT = 3


    def __init__(self):
        self.__currentFrame = None
        self.__originalFrame = None
        self.__linesDetected = None
        
        self.__lastLeftLane = None
        self.__lastRightLane = None
        
        self.__leftCords = None
        self.__rightCords = None
        self.__movementDirection = self.__Direction.STRIGHT

        self.__prevLeft = []
        self.__prevRight = []
    
    def __getVideoCapture(self, fileName):
        """ return the video capture object if the object is already taken return None """
            
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
        
        self.__currentFrame = cv2.cvtColor(self.__currentFrame, cv2.COLOR_BGR2GRAY)
    
    def __removeNoise(self, d = 7, sigma = 5):
        """ remove noise from the curretnFrame using guassin / bilateralFilter """
        
        self.__currentFrame = cv2.GaussianBlur(self.__currentFrame, (d, d), sigma)
    
    def __detectEdges(self, lowThreshold = 50, highThreshold = 150):
        """ detect the edges of the currentFrame using Canny """
        
        self.__currentFrame = cv2.Canny(self.__currentFrame, lowThreshold, highThreshold)
    
    def __convertImageToBinary(self):
        """ set the image to 0 or 1 only and by that converting it to binary image color """
        
        self.__currentFrame = cv2.threshold(self.__currentFrame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    def __cropImage(self):
        """ set the currentFrame to the road in a triangle shape """
        
        height = self.__currentFrame.shape[0]
        left_x, right_x = 50, 1200
        topMiddlePoint = ((right_x + left_x) / 2, 340)
        buttomRightPoint = (right_x, height)
        buttomLeftPoint = (left_x, height)
        polygons = np.array([[buttomLeftPoint, buttomRightPoint, topMiddlePoint]]).astype(np.int)

        mask = np.zeros_like(self.__currentFrame)
        cv2.fillPoly(mask, polygons, 255)

        self.__currentFrame = cv2.bitwise_and(self.__currentFrame, mask)
    
    def __createLines(self, maxDistance = 200, maxAngle = 1.2):
        """ draw the lines detected on the original image """
        
        if self.__linesDetected is None or type(self.__linesDetected) is None:
            return None
        
        lines_image = np.zeros_like(self.__originalFrame)
        
        try:
            for x1, y1, x2, y2 in self.__linesDetected:
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        except ValueError:
            pass
        
        self.__originalFrame = cv2.addWeighted(self.__originalFrame, 0.8, lines_image, 1, 1)
    
    def __getCordinates(self, parameters):
        """ return the coedinates of the slope and intercept given using linear algebra """

        if parameters is None:
            return np.array([0, 0, 0, 0])

        slope, intercept = parameters
        y1 = self.__currentFrame.shape[0]
        y2 = int(y1 * (3 / 5))
        
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    
    def __detectSides(self):
        """ detect right and left lanes and set linesDetected to the avreage of each """
        
        leftLane = []
        rightLane = []
        
        if self.__linesDetected is None or type(self.__linesDetected) is None:
            return None
        
        for currentLine in self.__linesDetected:
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
        
        if type(leftAvrage) is np.float64:
            leftAvrage = self.__lastLeftLane
        else:
            self.__lastLeftLane = leftAvrage
            
        if type(rightAvrage) is np.float64:
            rightAvrage = self.__lastRightLane
        else:
            self.__lastRightLane = rightAvrage

        if leftAvrage is None or rightAvrage is None:
            return None

        self.__leftCords = self.__getCordinates(leftAvrage)
        self.__rightCords = self.__getCordinates(rightAvrage)

        self.__prevLeft.append(self.__leftCords[0])
        if len(self.__prevLeft) > 10:
            _ = self.__prevLeft.pop(0)

        self.__prevRight.append(self.__rightCords[0])
        if len(self.__prevRight) > 10:
            _ = self.__prevRight.pop(0)
        
        # checks if there is a change in the direction and remove the line oposite to the direction
        if self.__movementDirection == self.__Direction.RIGHT:
            self.__linesDetected = np.array([self.__rightCords])

        elif self.__movementDirection == self.__Direction.LEFT:
            self.__linesDetected = np.array([self.__leftCords])

        else:
            self.__linesDetected = np.array([self.__leftCords, self.__rightCords])
    
    def __drawMiddleLine(self):
        """ draw a line in the middle of the image, and check if the diffrence between the lines are towrad the right or left """
        
        if self.__rightCords is None:
            return None
        
        distance_th = 100
        middle_x = int(self.__originalFrame.shape[1] / 2)
        middlePointText = (int(self.__originalFrame.shape[1] / 2) - 60 , 100)
        
        # Check left
        self.__checkDirection(self.__prevLeft, middle_x, distance_th, self.__Direction.RIGHT, self.__Direction.LEFT, self.__Direction.STRIGHT)
        
        # Check right
        self.__checkDirection(self.__prevRight, middle_x, distance_th, self.__Direction.LEFT, self.__Direction.RIGHT, self.__Direction.STRIGHT)

        display_text = ''
        if self.__movementDirection != self.__Direction.STRIGHT:
            display_text = self.__movementDirection.name

        cv2.putText(self.__originalFrame, display_text,
                    middlePointText, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, 2)
    
    def __checkDirection(self, previosDirectionLane, middle, distanceThreshold, opositeDirection, wantedDirection, defaultDirection, maxFrames = 5):
        """ check if the wanted direction was the current direction for maxFrames frames back """
        laneShiftAmount = 0

        for direction in previosDirectionLane:
            if abs(direction - middle) < distanceThreshold:
                laneShiftAmount += 1

        if laneShiftAmount >= maxFrames and self.__movementDirection != opositeDirection:
            self.__movementDirection = wantedDirection
        elif self.__movementDirection == wantedDirection:
            self.__movementDirection = defaultDirection
    
    def __detectLines(self, threshold = 100, radiusStep = 2, angleStep = np.pi / 180.0):
        """ detect lines in the image and show it """
        
        self.__linesDetected = cv2.HoughLinesP(self.__currentFrame, radiusStep, angleStep, threshold, np.array([]), minLineLength = 40, maxLineGap = 5)
        self.__detectSides()
        self.__createLines()
        self.__drawMiddleLine()
    
    def __showCurrentImage(self):
        """ show the original image """
        
        cv2.imshow('Frame', self.__originalFrame)
    
    def __quitDetected(self):
        """ check if the user wants to quit """
        
        return cv2.waitKey(25) & 0xFF == ord('q')
    
    def __detectLinesInImage(self, image=None):
        """ detect lines in one image per time """
        
        if image is not None:
            self.__originalFrame = image
            self.__currentFrame = self.__originalFrame.copy()
        
        self.__convertToGrey()
        self.__removeNoise()
        self.__detectEdges()
        self.__cropImage()
        self.__detectLines()
        self.__showCurrentImage()
        
    def detect(self, videoFileName, outputFileName = "output.avi", videoOutput = True):
        """ main function to detect lines in the video and show it """
        
        videoWriter = None
        videoCapture = self.__getVideoCapture(videoFileName)
        
        if videoCapture is not None:
            while videoCapture.isOpened():
                ret, self.__originalFrame = videoCapture.read()
                
                if not ret or self.__quitDetected():
                    break
                
                if videoOutput and videoWriter is None:
                    videoWriter = self.__getVideoWriter(self.__originalFrame, outputFileName)
                
                self.__currentFrame = self.__originalFrame.copy()
                
                self.__detectLinesInImage()
                
                if videoOutput:
                    videoWriter.write(self.__originalFrame)

            videoCapture.release()

            if videoOutput:
                videoWriter.release()
        cv2.destroyAllWindows()
