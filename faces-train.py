import cv2
import sys
import os
import numpy as np

subjects = ["", "benedict", "gal", "niloofar"]

def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier("/home/niloofar/git/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml")

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
	
	#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None
	
	#under the assumption that there will be only one face,
	#extract the face area
	(x, y, w, h) = faces[0]
	
	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	image_dir = os.path.join(BASE_DIR, data_folder_path)

	faces = []
	labels = []

	for root, dirs, files in os.walk(image_dir):
		for file in files:
			path = os.path.join(root, file)
			label = int(os.path.basename(root).replace("s", ""))
			image = cv2.imread(path)
			cv2.imshow("Training on image...", image)
			cv2.waitKey(100)
			 
			face, rect = detect_face(image)
			 
			if face is not None:
				faces.append(face)
				labels.append(label)
 
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	cv2.destroyAllWindows()

	return faces, labels


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

print("Preparing data...")
faces, labels = prepare_training_data("face-training-data")
print("Data prepared")
 
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
	img = test_img.copy()
	#detect face from the image
	face, rect = detect_face(img)
	 
	#predict the image using our face recognizer 
	label= face_recognizer.predict(face)
	#get name of respective label returned by face recognizer
	label_text = subjects[label[0]]
	print(label_text)
	 
	draw_rectangle(img, rect)
	draw_text(img, label_text, rect[0], rect[1]-5)
	 
	return img



print("Predicting images...")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

img_item = "me.png"
cv2.imwrite(img_item, frame)

#load test images
#test_img1 = cv2.imread("test-data/test1.jpg")

#perform a prediction
predicted_img1 = predict(frame)
print("Prediction complete")
 
#display both images
cv2.imshow(subjects[1], predicted_img1)

cv2.waitKey(2000)
cv2.destroyAllWindows()