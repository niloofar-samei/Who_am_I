import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False

while rval:
	cv2.imshow("preview", frame)
	rval, frame = vc.read()
	key = cv2.waitKey(5)

# close webcam 3000ms
#	cv2.waitKey(3000)
#	cv2.destroyAllWindows("preview")

	if key == 27 : # exit on ESC
		break
cv2.destroyWindow("preview")