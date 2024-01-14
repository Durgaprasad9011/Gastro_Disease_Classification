# Medical Image Analytics for Gastro Disease Diagnosis

Gastrointestinal diseases encompass a range of disorders affecting the digestive tract, involving organs such as the stomach, intestines, and related structures. Causes vary and may include infections, inflammation, genetic factors, and lifestyle choices, leading to symptoms like abdominal pain, vomiting, constipation, feeding intolerance, diarrhea, and nutritional deficiencies, significantly impacting overall health. Our focus is on addressing this challenge through:

## Problem Statement

Proposing a technology-driven solution for precise classification of gastro diseases using Deep Learning. Traditional scanning methods fall short in describing specific disease types, necessitating manual intervention for accurate diagnosis.

## Dataset

The dataset, sourced from [Open Science Framework (OSF)](https://osf.io/dv2ag/) known as KVASIR-Capsule Dataset, comprises image modalities from video frames captured using Capsule Endoscopy Scans. It consists of 14 different disease classes, totaling around 47,238 images in '.jpg' format.

![Random Images from Each Class](https://drive.google.com/file/d/1zF-Yp1HqqzWcEvtLpqzGBm-zH16cCf7Z/view?usp=sharing)

[More details are provided in the Streamlit application. Link is attached below the file.]

## Model Training

We employed Deep Learning models such as CNN_2D, Resnet, Densenet121, InceptionV3, VGG19, and Few Shot Learning, conducting a Comparative Analysis. The selection of these models is justified by their success in handling large datasets (47,238 images across 14 classes) where robust feature extraction is crucial for accurate image classification, as per our literature review.

## Results

![Result Image 1](https://drive.google.com/file/d/1je0fPyglTW1eW1cajH8C1_7Gr-jG0bPz/view?usp=sharing)

![Result Image 2](https://drive.google.com/file/d/1-0UjG1Sqcarq-qanb_2HxpUTBu4I7Xlq/view?usp=sharing)

Quantitative results are attached in the Streamlit application, accessible via the following link:

[Streamlit Application](https://gastrodiseaseclassification.streamlit.app/)

## Future Scope

- Accurate Lesion Detection and Quantification of Lesion Area.
- Designing a Lightweight Architecture.
- Deployment on Application or Edge Devices.
- Conference Paper.

Feel free to explore the Streamlit application for a detailed analysis and insights.
