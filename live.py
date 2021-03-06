# This script will detect faces via your webcam.
# Tested with OpenCV3
import cv2

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("/home/niloofar/git/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

img_count = 0

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	
	# Define the codec and create VideoWriter object.
	#out.write(frame)
	
	img_count = img_count+1
	#cv2.imwrite("me{0}.png".format(str(img_count+1)), frame)
	
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print(faces)
	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 179, 255), 2)

	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
