import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import PrototypicalNet
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Define constants for Few Shot Learning    
image_size = 224
num_classes_few_shot = 14

# Data transforms setup for Few Shot Learning
transform_few_shot = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load the Few Shot Learning model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_few_shot_model(model_path):
    model = PrototypicalNet(num_classes_few_shot).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Prediction function for Few Shot Learning
def predict_few_shot(model, img, transform):
    img = Image.open(img).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()

# Constants for CNN-2D Model
class_names_cnn = ['Ampulla of vater', 'Angiectasia', 'Blood - fresh', 'Blood - hematin', 'Erosion', 'Erythema',
                   'Foreign body', 'Ileocecal valve', 'Lymphangiectasia', 'Normal clean mucosa', 'Polyp', 'Pylorus',
                   'Reduced mucosal view', 'Ulcer']

# Load the CNN-2D Model
model_cnn = tf.keras.models.load_model('my_cnn_model_augmented.h5')

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Welcome Page", "Dataset Analysis", "Predictions", "Comparative Study"])

# Welcome Page
if page == "Welcome Page":
    st.title("Medical Image Analytics for Gastro Disease Diagnosis")
    st.image("https://img.medscapestatic.com//thumbnail_library/is_150918_intestines_stomach_digestive_800x600.jpg", use_column_width=True)
    st.write("In these times, Gastro diseases can cause significant discomfort and lead to serious conditions such as gastric cancers, ulcers, stomach aches, and vomiting etc. Early detection of the disease is crucial.")
    st.write("We the team Akash Kalasagond, Kasam Rohith Reddy, Sunke Durgaprasad, Bandaru Karthik, Raghul S from Woxsen University, under the mentorship of Dr. Jaswanth Nidamanuri, Assistant Professor, introduces this Streamlit application. We showcase our work, including dataset analysis, comparative study of various deep learning models, along with predictions.")

# Dataset Analysis Page
elif page == "Dataset Analysis":
    st.title("KVASIR Capsule Endoscopy Images")
    st.markdown(" ### *Source:* *[https://osf.io/dv2ag/](https://osf.io/dv2ag/)*")
    st.write("-------------------------------------")

    # Display the number of images in each directory
    st.write("")
    st.write("### *Number of images in each directory:*")
    list = ['Ampulla of vater: 10 images', 'Angiectasia: 866 images', 'Blood - fresh: 446 images', 'Blood - hematin: 12 images',
            'Erosion: 506 images', 'Erythema: 159 images', 'Foreign body: 776 images', 'Ileocecal valve: 4189 images',
            'Lymphangiectasia: 592 images', 'Normal clean mucosa: 34338 images', 'Polyp: 55 images',
            'Pylorus: 1529 images', 'Reduced mucosal view: 2906 images', 'Ulcer: 854 images', 'Total number of images: 47238 images']

    for i in list:
        st.write(i)

    # Display the image
    script_directory = os.path.dirname(os.path.abspath(__file__))
    plot_image_path = os.path.join(script_directory, 'Images', 'Image Plot.jpg')
    plot_image = Image.open(plot_image_path)
    st.image(plot_image, use_column_width=True)

# Predictions Page
elif page == "Predictions":
    st.title("Gastro Image Multi-level Classification")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # User's choice for prediction model
    model_choice = st.sidebar.selectbox("Choose a Model", ["Few Shot Learning", "CNN-2D Model"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        if model_choice == "Few Shot Learning":
            model_few_shot = load_few_shot_model('FewshotModel_1.pth')
            predicted_class_few_shot = predict_few_shot(model_few_shot, uploaded_file, transform_few_shot)
            st.write("Predicted Class (Few Shot Learning):", class_names_cnn[predicted_class_few_shot])

        elif model_choice == "CNN-2D Model":
            img_cnn = image.load_img(uploaded_file, target_size=(336, 336))
            img_array_cnn = image.img_to_array(img_cnn)
            img_array_cnn = np.expand_dims(img_array_cnn, axis=0)

            predictions_cnn = model_cnn.predict(img_array_cnn)
            predicted_class_index_cnn = predictions_cnn.argmax(axis=1)
            st.write("Predicted Class (CNN-2D Model):", class_names_cnn[predicted_class_index_cnn[0]])

# Comparative Study Page
elif page == "Comparative Study":
    st.title("Quantitative and Qualitative Results")
    st.write("-------------------------------------")

    # Quantitative Results
    st.header("Quantitative Results")

    # Display Quantitative Results Image
    script_directory = os.path.dirname(os.path.abspath(__file__))
    quantitative_image_path = os.path.join(script_directory, 'Images', 'Table-01.jpg')
    quantitative_image = Image.open(quantitative_image_path)
    st.image(quantitative_image, use_column_width=True)

    script_directory_1 = os.path.dirname(os.path.abspath(__file__))
    quantitative_image_path_1 = os.path.join(script_directory_1, 'Images', 'Table-01.jpg')
    quantitative_image_1 = Image.open(quantitative_image_path_1)
    st.image(quantitative_image_1, use_column_width=True)

    # Qualitative Results
    st.header("Qualitative Results")

    # CNN-2D Model Results
    st.subheader("CNN-2D Model Results")

    cnn_results_plot_path = "Images\CNN-2D\Loss, Acc plot.png"
    cnn_results_plot = Image.open(cnn_results_plot_path)
    st.image(cnn_results_plot, caption="Accuracy, Loss Plot", use_column_width=True)
    
    cnn_results_matrix_path = "Images\CNN-2D\Confusion Matrix.png"
    cnn_results_matrix = Image.open(cnn_results_matrix_path)
    st.image(cnn_results_matrix, caption="Confusion Matrix", use_column_width=True)


    # Resnet Model Results
    st.subheader("Resnet Model Results")
    resnet_results_plot_path = "Images\Resnet\Loss, Accuracy Plot.jpg"
    resnet_results_plot = Image.open(resnet_results_plot_path)
    st.image(resnet_results_plot, caption="Accuracy, Loss Plot", use_column_width=True)
    
    resnet_results_matrix_path = "Images\Resnet\Confusion Matrix.jpg"
    resnet_results_matrix = Image.open(resnet_results_matrix_path)
    st.image(resnet_results_matrix, caption="Confusion Matrix", use_column_width=True)

    # Densenet121 Model Results
    st.subheader("Densenet Model Results")
    Densenet_results_plot_path = "Images\Densenet121\Loss, Acc plot.jpg"
    Densenet_results_plot = Image.open(Densenet_results_plot_path)
    st.image(Densenet_results_plot, caption="Accuracy, Loss Plot", use_column_width=True)
    
    Densenet_results_matrix_path = "Images\Densenet121\Confusion Matrix.jpg"
    Densenet_results_matrix = Image.open(Densenet_results_matrix_path)
    st.image(Densenet_results_matrix, caption="Confusion Matrix", use_column_width=True)

    # Inception V3 Model Results
    st.subheader("Inception V3 Model Results")
    Incnet_results_plot_path = "Images\Inception V3\Loss, Acc plot.jpg"
    Incnet_results_plot = Image.open(Incnet_results_plot_path)
    st.image(Incnet_results_plot, caption="Accuracy, Loss Plot", use_column_width=True)
    
    Incnet_results_matrix_path = "Images\Inception V3\Confusion Matrix.jpg"
    Incnet_results_matrix = Image.open(Incnet_results_matrix_path)
    st.image(Incnet_results_matrix, caption="Confusion Matrix", use_column_width=True)

    # VGG-19 Model Results
    vgg_results_plot_path = "Images\VGG-19\Model Loss.jpg"
    vgg_results_accuracy_path = "Images\VGG-19\Model Accuracy.jpg"
    vgg_results_matrix_path = "Images\VGG-19\Confusion Matrix.jpg"
    st.subheader("VGG-19 Model Results")
    vgg_results_matrix = Image.open(vgg_results_matrix_path)
    vgg_results_accuracy = Image.open(vgg_results_accuracy_path)
    vgg_results_plot = Image.open(vgg_results_plot_path)
    st.image(vgg_results_plot, caption="Loss Plot", use_column_width=True)
    st.image(vgg_results_accuracy, caption="Accuracy Plot", use_column_width=True)
    st.image(vgg_results_matrix, caption="Confusion Matrix", use_column_width=True)

    # Few Shot Learning Results
    few_shot_results_plot_path = "Images\Few Shot Learning\Loss Plot.png"
    few_shot_results_matrix_path = "Images\Few Shot Learning\Confusion Matrix.png"
    st.subheader("Few Shot Learning Results")
    few_shot_results_plot = Image.open(few_shot_results_plot_path)
    few_shot_results_matrix = Image.open(few_shot_results_matrix_path)
    st.image(few_shot_results_plot, caption="Loss Plot", use_column_width=True)
    st.image(few_shot_results_matrix, caption="Confusion Matrix", use_column_width=True)
