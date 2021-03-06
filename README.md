[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

# Computer vision: lane detection

## System Requirements:
- python 3.6 or heighr
- cv2 library
- numpy library

### Installation:
installation can be done using conda.

```cmd
conda activate
python setup.py install
```

## Activation:
- from main.py import LaneDetection to your project
- in LaneDetection there is only one public function named detect
- detect recives 3 arguments which the last two are optional
- the first is the video file name you went to detect the lanes in
- the output file name you want the program to write the output video into
- lastly boolean for if you want the program to output a video file or not

#### TLDR:
```python
from laneDetection import LaneDetection
LaneDetection().detect(fileName, outputName, videoOutput)
```

## Assumptions:

1. We found that in each tested video there is a somewhat triangle shaped region
that was important.
This area was formed from the starting of the lane to its end and the top of the
triangle is at the point where the lines would intersect in the horizon.
The area around the triangle would only confuse the system as there are more lines to
find from other lanes and other noise.

2. To remove any indication of color we turned the images to a grey scale format so
the resulting image will not consider color as a factor but only the color brightness.
To minimize the noise we used Gaussian function with parameters that we found
after some tests.
After we reduced noises we ran the Canny algorithm to find edges and hopefully
the lanes we want.
On top of that, in the Canny algorithm we tried a few different versions and
concluded that those parameters are best.
They reduced other edges found by accident and edges that are not interesting for
this project.

3. After we found a few different lines using a combination of Canny for the edges
and Hugh lines, we average the resulting points of each side using the slope to
differentiate between right and left. We also removed lines with slope close to
zero, aka horizontal lines because the lines of the lane are not a horizontal line.

### Here a few examples of what the system outputs are:

<img width="45%" height="250px" src="/Demo%20Assets/sideRoadEX1.png" /> <img width="45%" height="250px" src="/Demo%20Assets/sideRoadEX2.png" />


### Here are a few pictures of behind the scenes that show how the computer see the image.
The left one shows the picture in grey scale without the blur effect with canny
and the right one shows the picture in grey scale with the blur effect.


<img width="47%" height="250px" src="/Demo%20Assets/roadWithNoise.png" /> <img width="47%" height="250px" src="/Demo%20Assets/roadWithoutNoise.png" />


As we can see the addition of the blur effect reduces noise and irrelevant point of
interest that the computer can mistake for points of a line.

In addition to the lane detection we also implemented a change in lane detection,
when the car will shift lanes the system will inform the change and detect where the
change in lane is, shift to the right or to the left.


## Video Demo:
<img src="/Demo%20Assets/Lane Detection Demo.gif" />

[link]: https://github.com/MortarDefender/CV-LaneDetection
