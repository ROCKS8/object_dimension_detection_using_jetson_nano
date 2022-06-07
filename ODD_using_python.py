# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


import cv2

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                    
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'x'
                if keyCode == 27 or keyCode == ord('x'):
                    break
        finally:
            cv2.imwrite("image.png",frame)
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")
        
show_camera()

def midpoint(ptA, ptB):
   return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# perform edge detection, then perform a dilation + erosion to
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
   cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
   # if the contour is not sufficiently large, ignore it
   if cv2.contourArea(c) < 100:
      continue
   # compute the rotated bounding box of the contour
   orig = image.copy()
   box = cv2.minAreaRect(c)
   box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
   box = np.array(box, dtype="int")
   box = perspective.order_points(box)
   cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
   # loop over the original points and draw them
   for (x, y) in box:
      cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
   # unpack the ordered bounding box, then compute the midpoint
   # between the top-left and top-right coordinates.
   (tl, tr, br, bl) = box
   (tltrX, tltrY) = midpoint(tl, tr)
   (blbrX, blbrY) = midpoint(bl, br)
   (tlblX, tlblY) = midpoint(tl, bl)
   (trbrX, trbrY) = midpoint(tr, br)

   # draw the midpoints on the image
   cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
   cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
   cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
   cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

   # draw lines between the midpoints
   cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
      (255, 0, 255), 2)
   cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
      (255, 0, 255), 2)
   # compute the Euclidean distance between the midpoints
   dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
   dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
   # if the pixels per metric has not been initialized, then
   # compute it as the ratio of pixels to supplied metric
   # (in this case, inches)
   if pixelsPerMetric is None:
      pixelsPerMetric = dB / 2.5 #args["width"]
   # compute the size of the object
   dimA = dA / pixelsPerMetric
   dimB = dB / pixelsPerMetric
   # draw the object sizes on the image
   cv2.putText(orig, "{:.1f}in".format(dimA),
      (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
      0.65, (0,0,0), 2)
   cv2.putText(orig, "{:.1f}in".format(dimB),
      (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
      0.65, (0,0,0), 2)

   # show the output
   cv2.imshow("Image", orig)
   cv2.waitKey(0)

