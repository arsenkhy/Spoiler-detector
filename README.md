# Spoiler detection in movie reviews 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
## Background
Spoiler detection would be a valuable tool that enhances the overall experience of movie by identifying and flagging potential spoilers within reviews. Additionally, spoiler detection fosters a more positive and engaging environment for discussions, enabling users to engage in conversations without inadvertently revealing critical plot details to others who may not have experienced the content yet.

The project motivation was to enhance the accuracy of the model while simultaneously minimizing the requirement for extensive training data. The objective was to create a model that could identify spoilers while also addressing the resource-intensive nature of training with large datasets. This approach was driven by the aspiration to develop a solution that is not only efficient but also well-suited for real-world applications. In essence, the focus was on creating an optimal balance between model efficiency and the capacity to accurately detect spoilers.

## Data 
IMDB Spoiler Dataset available on [Kaggle](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset) by Misra (2019). This dataset consisted of two main subsets: movie details and user reviews.

Movie Details (Dataset 1):
This subset contained 1,572 movie entries. Each entry had attributes like a unique movie identifier (movie_id) and duration. These details provided context.

Review Details (Dataset 2):
The second subset included an extensive collection of 573,913 user reviews from 263,407 users. Among them, 150,924 reviews were labeled as spoilers (is_spoiler = true). Each review had the review's text (review_text) and spoiler information (is_spoiler).

Notably, during training, only reviews with plot synopses over 50 words were chosen for better training quality. We also picked movies with 200-300 reviews to reduce dataset variation. Only English reviews were used for consistent analysis language.

Text preprocessing involved cleaning raw text, lowercasing, and removing common stopwords to enhance the model's focus on informative content.

The final training data consisted of about 55,000 reviews from 216 unique movies, roughly 10% of the entire dataset.

## Approach 
In this project, our main focus was on employing a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network. The choice of the Bi-LSTM model was driven by its ability to enhance the model's comprehension through the capture of sequential information from two distinct directions: both forwards (from the past to the future) and backwards (from the future to the past).

In the context of spoiler detection, narratives often reveal themselves through the interplay of past and future events, as well as subtle cues and foreshadowing. Consider a review containing potential spoiler elements that are scattered across the text, whether through hints or allusions

## Results
Achieving a **80%** Accuracy:
Initial training on the selected dataset yielded a baseline accuracy of approximately 66%. Through the integration of techniques like data augmentation, text preprocessing, and attention mechanisms, the model's accuracy significantly improved, reaching an impressive 80%. This outcome highlights the effectiveness of our approach in enhancing spoiler detection capabilities.

## References
[1] Misra, Rishabh. "IMDB Spoiler Dataset." DOI: 10.13140/RG.2.2.11584.15362 (2019).

## Project setup and running

- Create virtual environment:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

- If you want to check the model output:

```python3 check.py``` this will print the classification report

The model outputs the prediction 0 for spoiler and 1 for spoiler to the `output/model_output.out`

It then checks values against the test set in `reference/test_reference.txt`

- If you want to run the training:

You need to firstly download the train_data.json and test_data.json.
It contains already preprocessed data (I uploaded to google drive [here](https://drive.google.com/drive/u/0/folders/1Shz1Zh6D7vFJFDEgVgkUzGdLo52x7YE7)). Then you need to put it inside the `data/` folder.
Then you can start the training by:
```python3 spoiler_detector.py```. It will also evaluate the model on the test set and as long as there is a bilstm_model
inside the `models/`, doing the same command will run the model on the test set