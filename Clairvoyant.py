#!pip install --upgrade pip
#!pip install clarifai --upgrade

import clarifai
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

MY_API_KEY = '9f14ea7b98e0449d8f65509d5b882896'
app = ClarifaiApp(api_key = MY_API_KEY)

model = app.models.get('general-v1.3')
#image = ClImage(file_obj=open('/Users/depressedonion/Downloads/kittenSmol.jpg', 'rb'))
print("Type location: ")
imageLocation = input()
if imageLocation.startswith('https'):
    image = ClImage(url=imageLocation)
response = model.predict([image])

outputs = response['outputs']
stuff = outputs[0]
infoEach = stuff['data']
concepts = infoEach['concepts']

listOfWords = []
for one in concepts:
    listOfWords.append(one['name'])

for word in listOfWords:
    print(word)
