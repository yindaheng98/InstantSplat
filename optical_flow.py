import numpy as np
import cv2 as cv

frame1 = cv.imread("data/N3DV/coffee_martini/3_views/images/cam01.png")
frame2 = cv.imread("data/N3DV/coffee_martini20/3_views/images/cam01.png")
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imwrite('opticalfb.png', frame2)
cv.imwrite('opticalhsv.png', bgr)
