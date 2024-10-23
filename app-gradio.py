import gradio as gr
import pandas as pd
import keras
import numpy as np
import logging
import io
from PIL import Image
import plotly.express as px
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_class_names():
    with open('class_names.txt', 'r') as file:
        class_names = file.readlines()
    return [class_name.strip() for class_name in class_names]

def classify_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert PIL image to BytesIO object
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Load image using keras.preprocessing.image.load_img
    image = keras.preprocessing.image.load_img(img_byte_arr, target_size=(100, 100))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    logging.info('Image successfully resized')
    
    
    
    model = keras.models.load_model('model/best_model_cnn.keras')
    logging.info('Model loaded successfully')

    prediction = model.predict(image)
    logging.info('Prediction made')

    class_names = get_class_names()
    
    predicted_label = class_names[np.argmax(prediction, axis=1)[0]]
    predicted_label = str(predicted_label)
    logging.info('Label predicted')

    confidence_score = np.max(prediction)

    top_k = 5
    top_k_indices = np.argsort(prediction[0])[-top_k:][::-1]
    top_k_labels = [class_names[i] for i in top_k_indices]
    top_k_scores = prediction[0][top_k_indices]
    # Create a dictionary for the top-k predictions
    top_k_predictions = {label: float(score) for label, score in zip(top_k_labels, top_k_scores)}

    df = pd.DataFrame({
        'Class': top_k_labels,
        'Confidence': top_k_scores
    })

    df = df.sort_values(by='Confidence', ascending=False)
    fig = px.bar(df, x='Class', y='Confidence', title='Top 5 Predictions')
    
    return predicted_label, 100*float(confidence_score), fig

def about_author():
    return """
    **S. Abraham Z. KOLOBOE**

    *Data Scientist | Ingénieur en Mathématiques et Modélisation*

    Bonjour,
    Je suis Abraham, un Data Scientist et Ingénieur en Mathématiques et Modélisation.
    Mon expertise se situe dans les domaines des sciences de données et de l'intelligence artificielle.
    Avec une approche technique et concise, je m'engage à fournir des solutions efficaces et précises dans mes projets.

    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """

with gr.Blocks() as demo:
    gr.Markdown("# 🍎 Fruits Classification App 🍌")
    with gr.Accordion("A propos de l'auteur", open=False):
        gr.Markdown(about_author())

    with gr.Row():
        with gr.Column():
            image_input = gr.Image()
            label_output = gr.Label()
            confidence_output = gr.Label()

        with gr.Column():
            with gr.Accordion("A propos de l'application", open=True):
                gr.Markdown("""
                ## Bienvenue dans l'application de classification de fruits ! 🍇🍉🍍

                Cette application utilise un modèle de deep learning pour classifier des images de fruits. Vous pouvez télécharger une image de fruit, et l'application affichera la classe correspondante ainsi que le score de confiance de la prédiction. De plus, un graphique en barres montrera les scores de confiance des 5 classes les plus proches.

                ### Fonctionnalités
                - 📷 **Téléchargement d'image** : Chargez une image de fruit au format JPG, PNG ou JPEG.
                - 🧠 **Prédiction** : Le modèle de deep learning prédit la classe du fruit.
                - 📊 **Visualisation** : Affichez un graphique en barres des scores de confiance des 5 classes les plus proches.

                ### Comment utiliser l'application
                1. **Téléchargez une image** : Cliquez sur le bouton "Choose an image..." et sélectionnez une image de fruit depuis votre appareil.
                2. **Affichage de l'image** : L'image téléchargée sera affichée dans la première colonne.
                3. **Résultats de la prédiction** : La classe prédite et le score de confidence seront affichés dans la deuxième colonne.
                4. **Graphique en barres** : Un graphique en barres montrera les scores de confiance des 5 classes les plus proches.

                Profitez de l'application et amusez-vous à classifier vos fruits préférés ! 🍇🍉🍍

                N'hésitez pas à contribuer ou à signaler des problèmes via les issues du dépôt [GitHub](https://github.com/abrahamkoloboe27/Fruits-Recognition-Training) .
                """)
            
            top_k_output = gr.Plot()
    image_input.change(classify_image, inputs=image_input, outputs=[label_output, confidence_output, top_k_output])

demo.launch()
