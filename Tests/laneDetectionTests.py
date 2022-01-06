import sys
sys.path.append('../')

from src.laneDetection import LaneDetection


if __name__ == '__main__':
    LaneDetection().detect("testCam3.mp4", videoOutput = False)
    # LaneDetection().detect("testCam3.mp4", videoOutput = True)
