# Medical Image Analytics for Gastro Disease Diagnosis

- Gastrointestinal diseases refer to disorders affecting the digestive tract, including the stomach, intestines, and other related organs.
- The causes vary and may include infections, inflammation, genetic factors, lifestyle choices, etc.
- Leading to symptoms such as abdominal pain, Vomiting, Constipation, Feeding intolerance, diarrhea, nutritional deficiencies, impacting overall health, etc. Our focus is on addressing this challenge through:

## Problem Statement

- Proposing a technology-driven solution for precise classification of gastro diseases using Deep Learning. Traditional scanning methods fall short in describing specific disease types, necessitating manual intervention for accurate diagnosis.

## Dataset

- The dataset, sourced from [Open Science Framework (OSF)](https://osf.io/dv2ag/) known as the KVASIR-Capsule Dataset, comprises image modalities from video frames captured using Capsule Endoscopy Scans. It comprises 14 different disease classes, totaling around 47,238 images in the '.jpg' format.

![Random Images from each Class](https://drive.google.com/uc?export=download&id=1zF-Yp1HqqzWcEvtLpqzGBm-zH16cCf7Z)


*[More details are provided in the Streamlit application. The link is attached below in the file.]*

## Model Training

- We have used Deep Learning models like CNN_2D, Resnet, Densenet121, InceptionV3, VGG19, Few-Shot Learning, conducted a Comparative Analysis, and drawn the results as attached in the document.

- The main reason for selecting these models is that we are dealing with a large dataset consisting of 47,238 images spread across 14 classes. The feature extraction is critical, and these models have proven successful in image classification tasks, demonstrating valuable accuracy and robust feature extraction, according to our literature review.

## Results

![Alt Text](https://drive.google.com/uc?export=download&id=1je0fPyglTW1eW1cajH8C1_7Gr-jG0bPz)
![Alt Text](https://drive.google.com/uc?export=download&id=1-0UjG1Sqcarq-qanb_2HxpUTBu4I7Xlq)


*[Quantitative results are attached in the Streamlit application, attached below]*


## Streamlit Application
[Click Here](https://gastrodiseaseclassification.streamlit.app/)

## Future Scope

- Accurate Lesion Detection and Quantification of Lesion Area.
- Designing a Lightweight Architecture.
- Deployment on Application or Edge Devices.
- Conference Paper.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/kasamrohith02/Gastro_Disease_Classification
   ```

2. Change to the project directory:

   ```bash
   cd Gastro_Disease_Classification
   ```

3. Install the required libraries from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```
   If you already have the libraries installed, proceed to the next step.

4. Once the dependencies are installed, you can run the Streamlit app:

```bash
streamlit run app.py
```

5. The code is provided in this repository for reference purposes.
