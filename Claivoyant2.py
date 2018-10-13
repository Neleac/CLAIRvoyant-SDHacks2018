
# coding: utf-8

# In[4]:


get_ipython().system('pip install opencv-python')


# In[2]:


import cv2


# In[3]:


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()


# In[4]:


import clarifai
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

MY_API_KEY = '9f14ea7b98e0449d8f65509d5b882896'
app = ClarifaiApp(api_key = MY_API_KEY)


# In[6]:


i = 0
listOfWords = []
#while i < img_counter:
for i in range(0,img_counter):
    model = app.models.get('general-v1.3')
    imageLocation = "opencv_frame_{}.png".format(i)
    image = ClImage(file_obj=open(imageLocation, 'rb'))
    response = model.predict([image])
    outputs = response['outputs']
    stuff = outputs[0]
    infoEach = stuff['data']
    concepts = infoEach['concepts']
    for one in concepts:
        listOfWords.append(one['name'])
    for word in listOfWords:
        print(word)
    i+=1

