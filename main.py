import cv2
import numpy as np


def videoCapture(fileName):
    capture = cv2.VideoCapture(fileName)
    
    if capture.isOpened() == False:
        print("video is opened")
        return None
    
    while capture.isOpened():
        ret, frame = capture.read()
        # frameCanny = cv2.Canny(frame, 500, 600)
        frameCanny = cv2.Canny(frame, 300, 700)  # canny the given frame
        
        # crop the photo to the right size
        # x1, y1 = 70, 400
        # x2, y2 = 1000, 1000
        # frameCrop = frameCanny[y1:y2, x1:x2]
        
        # dilate canny
        kernel = np.zeros((5,5),dtype=np.uint8)
        kernel[2,:] = 1
        kernel[:,2] = 1
        frameDilated = cv2.dilate(frameCanny, kernel, iterations = 1)
        
        # create hugh lines
        TH = 250
        r_step = 1
        t_step = np.pi / 180.0
        lines = cv2.HoughLines(frameDilated, r_step, t_step, TH)
        
        # add the lines to the original picture
        frame = createLines(frame, lines)
        
        if ret:
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    capture.release()
    cv2.destroyAllWindows()


def createLines(img, lines):
    max_distance = 200
    
    res = img.copy()
    
    if lines is None or type(lines) is None:
        return res
    
    for r_t in lines:
        rho = r_t[0, 0]
        theta = r_t[0, 1]
        
        if theta < 1.2:         # TODO: finds only the left lane 
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + max_distance * (-b))
            y1 = int(y0 + max_distance * (a))
            x2 = int(x0 - max_distance * (-b))
            y2 = int(y0 - max_distance * (a))
        
            res = cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
    
    return res

if __name__ == '__main__':
    videoCapture("dashCam.mp4")
