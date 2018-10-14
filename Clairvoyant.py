from clarifai.rest import ClarifaiApp
import cv2

MY_API_KEY = 'e77c49d9b41845bb82caf94ab3f1471f'
app = ClarifaiApp(api_key = MY_API_KEY)

model = app.public_models.general_model

def take_image():
	cam = cv2.VideoCapture(0)

	while True:
		ret, frame = cam.read()
		cv2.imshow('frame', frame)
		
		key = cv2.waitKey(1)

		if key%256 == 27:
			# ESC pressed
			print("Escape hit, closing...")
			break
		elif key%256 == 32:
			# SPACE pressed
			cv2.imwrite('images\\opencv_frame.jpg', frame)
			print("Image taken")
			break

	cam.release()
	cv2.destroyAllWindows()

def predict(address):
	response = model.predict_by_filename(address)

	concepts = response['outputs'][0]['data']['concepts']

	concept_list = []
	for concept in concepts:
		concept_list.append(concept['name'])
		if len(concept_list) == 10:
			break

	for concept in concept_list:
		print(concept)


print("Hi there, my name is Clair. I am an AI image recognition agent based on Clarifai's vision API. Give me an image, and I'll give you my interpretation.\n")

#get image location
valid_response = False
while not valid_response:
	print("Enter 1 to take a picture from webcam\n")
	print("Enter 2 to choose a local image\n")
	print("Enter Q to quit")
	response = input()
	if response == str(1):
		valid_response = True
		print("Press Space to take picture")
		take_image()
		predict('images\\opencv_frame.jpg')
	elif response == str(2):
		print("Enter image location")
		address = input()
		try:
			predict(address)
			valid_response = True
		except:
			print("Invalid address\n")
	elif response == "Q":
		print("Goodbye!")
		break
	else:
		print("Invalid input\n")