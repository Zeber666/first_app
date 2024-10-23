import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ładowanie modelu
model = load_model('mnist_model.h5')

st.title('Rozpoznawanie cyfr - MNIST')

# Umożliwienie wgrania obrazu
uploaded_file = st.file_uploader("Załaduj obraz cyfry (28x28 pikseli)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Konwertowanie i przetwarzanie obrazu
    image = Image.open(uploaded_file).convert('L')  # Konwersja na odcienie szarości
    image = image.resize((28, 28))  # Zmiana rozmiaru do 28x28
    image_array = np.array(image).astype('float32') / 255  # Normalizacja
    image_array = image_array.reshape(1, 28, 28, 1)

    # Predykcja
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)

    # Wyświetlanie obrazu i wyniku predykcji
    st.image(image, caption=f'Predykcja: {predicted_label}', use_column_width=True)
