#!/usr/bin/python

"""
This program splits the given image
into equally spaced sub-images
each of them overlapping 50 percent
For instance, if HH and WW are both 2,
and the image is the numpad,
then the resulting sub-images are
1245, 2356, 4578, 5689
"""

import cv2
import os
import sys

# split the image into HH x WW parts
HH = 3  # parts along height
WW = 6  # parts along width

file = 'jungle'
ext = '.jpg'
if len(sys.argv) is 2:
    file = sys.argv[0]
else:
    print("Using default file: " + file)

img = cv2.imread(file + ext, cv2.IMREAD_COLOR)
assert img is not None

h, w, _ = img.shape
print(h, w)

cv2.imshow('winname', img)
print("Press \'s\' to proceed")
ch = cv2.waitKey() & 0xFF
cv2.destroyAllWindows()

if ch is not ord('s'):
    print("Aborting")
    sys.exit(0)

if not os.path.exists(file):
    print("Creating directory " + file)
    try:
        os.mkdir(file)
    except OSError:
        print ("Creation of the directory %s failed" % file)
        sys.exit(1)

hh = int(h / (HH + 1))
ww = int(w / (WW + 1))

for i in range (0, HH):
    for j in range (0, WW):
        cropped_img = img[i*hh:(i+2)*hh, j*ww:(j+2)*ww]
        filename = file + '/' + str(j + WW*i) + ext
        ret = cv2.imwrite(filename, cropped_img)
        if not ret:
            print(filename + " failed")

