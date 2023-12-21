from keras.preprocessing import image
from keras.models import load_model
import numpy as np

kelas_label = ['angry', 'sad', 'disgust', 'fear', 'happy', 'neutral', 'surprise']

def predict_class(image_path):
    img = image.load_img(image_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model('new_result_model.h5')
    prediksi = model.predict(img_array)
    predicted_class = np.argmax(prediksi)
    prediksi_label = kelas_label[predicted_class]

    return prediksi_label