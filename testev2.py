import os
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

model = load_model('./alzheimer_model_v2.keras')

# model.summary()

class_indices = {'nonDement': 0, 'withDement': 1}
labels = {v: k for k, v in class_indices.items()}

def analisa_imagens(imageParam):
        imageParam = image.img_to_array(imageParam)
        imageParam /= 255
        imageParam = np.expand_dims(imageParam, axis = 0)
        previsao = model.predict(imageParam)
        classe_predita = labels[np.argmax(previsao)]
        print([classe_predita, float(previsao[0][class_indices[classe_predita]])])


folders = ['./images/nonDement', './images/withDement']
image_files = []

for folder in folders:
    image_files.append([f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))])


images_loaded = []

for image_temp in image_files[0]:
    images_loaded.append(image.load_img('./images/nonDement/' + image_temp, target_size=(224, 224)))

for image_temp in image_files[1]:
    images_loaded.append(image.load_img('./images/withDement/' + image_temp, target_size=(224, 224)))

print(len(images_loaded))

for img in images_loaded:
    analisa_imagens(img)
