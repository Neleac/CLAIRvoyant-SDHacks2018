from clarifai.rest import ClarifaiApp

app = ClarifaiApp(api_key = 'e77c49d9b41845bb82caf94ab3f1471f')

model = app.public_models.general_model

response = model.predict_by_filename('images/wallpaper.jpg')

concepts = response['outputs'][0]['data']['concepts']
for concept in concepts:
	print(concept)
	#print(concept['name'], concept['value'])