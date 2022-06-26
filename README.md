# soi-hardly-human-2022

This project inteds at classifying face images based on emotions. There are 7 categories namely

1. Happy
2. Sad
3. Angry
4. Disgust
5. Neutral
6. Surprise
7. Fear

We have used a **CNN** which trains on `fer2013` dataset. Link to the dataset is [here](https://www.kaggle.com/datasets/msambare/fer2013).

#### Architecture of our CNN is described in `SOI_HH_Docs.pdf` file.

### Instructions

1. Clone the repository using the following command.
```
git clone https://github.com/shashankp28/soi-hardly-human-2022.git
```
2. Next install the required libraries.
```
pip install -r requirements.txt
```
3. Download all 3 models from google drive using this [link](https://drive.google.com/drive/folders/17p-8YiNF1Hbzt2Yl5-kQstP088qwSkNO?usp=sharing).
4. Copy these models into the `model` folder.

#### Please use a device with a camera to proceed.

### File descriptions

1. **SOI_HH_Augmentaion**: <br>
   You need to have the dataset with training and testing folders named as `data`. On running it performs necessary data sampling.
2. **SOI_HH_Training**: <br>
    Here you can see the model description, code to train the model and training logs.
3. **SOI_HH_Metric**: <br>
    Gives the evaluated accuracy for training and testing samples.
4. **SOI_HH_Camera**: <br>
    Opens webcam, detects face and shows current emotion.
5. **SOI_HH_Submission**: <br>
    Used to replicate the submission.csv. `example.csv` and `EMOTOR_TEST` folder need to be present inside the repository.
