from __future__ import division
import cv2
import os
import time
import sys

HEIGHT = 570
WIDTH = 3260
INPUT_FOLDER = 'output_frames'
OUTPUT_FOLDER = 'video'
OUTPUT_FILENAME = 'prediction.avi'

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME), fourcc, 30.0, (WIDTH,HEIGHT))

images = sorted(os.listdir(INPUT_FOLDER))

i = 0
for im in images:
	image = cv2.imread(os.path.join(INPUT_FOLDER, im))
	video.write(image)
	i += 1
	sys.stdout.write("\r%d%%" % int(100.0 * i / len(images)))
	sys.stdout.flush()
	

cv2.destroyAllWindows()
video.release()
