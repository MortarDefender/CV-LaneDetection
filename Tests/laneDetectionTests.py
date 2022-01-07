import sys
import json

sys.path.append('../')

from src.laneDetection import LaneDetection


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    LaneDetection().detect(config['test_video'], videoOutput = False)
    # LaneDetection().detect(config['test_video'], videoOutput = True)
