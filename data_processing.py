import pandas as pd
from langdetect import detect
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

# Define a function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    processed_text = " ".join(words)
    return processed_text

def preprocess_data():
    # Load movie reviews data
    df_reviews = pd.read_json('./data/raw/IMDB_reviews.json', lines=True)\
                 .drop_duplicates('review_text').sample(frac=1)
    print('Shape of movie reviews dataframe:', df_reviews.shape)

    # Load movie details data
    df_movies = pd.read_json('./data/raw/IMDB_movie_details.json', lines=True)
    print('Shape of movie details dataframe:', df_movies.shape)

    # Merge the two dataframes based on movie_id
    df_merged = df_reviews.merge(df_movies, on="movie_id", how="left", suffixes=('_review','_movie'))

    # Drop rows with missing plot synopsis
    df_merged = df_merged.dropna(subset=['plot_synopsis'])

    # Filter out plot_synopsis with more than 50 words
    df_merged = df_merged[df_merged['plot_synopsis'].apply(lambda x: len(x.split()) > 50)]

    # Create the training dataframe with required fields
    df_train = pd.DataFrame(columns=['movie_id', 'summary', 'review', 'is_spoiler'])

    df_train['movie_id'] = df_merged['movie_id']
    df_train['summary'] = df_merged['plot_synopsis']
    df_train['review'] = df_merged['review_text']
    df_train['is_spoiler'] = df_merged['is_spoiler']

    # Filter out movie IDs with review count between 200 to 300
    movie_review_counts = df_train['movie_id'].value_counts()
    valid_movie_ids = movie_review_counts[(movie_review_counts >= 200) & (movie_review_counts <= 300)].index
    df_train = df_train[df_train['movie_id'].isin(valid_movie_ids)]

    # Only english ones
    df_train = df_train[df_train['summary'].apply(is_english)]
    df_train = df_train[df_train['review'].apply(is_english)]

    # Preprocessing data
    # Remove stop words, lowercase, remove special characters, stemming
    df_train['summary'] = df_train['summary'].apply(preprocess_text)
    df_train['review'] = df_train['review'].apply(preprocess_text)

    df_train = df_train.dropna()
    print('number of rows:', df_train.shape[0])

    # Split the data into training and the test
    train_data, test_data = train_test_split(df_train, test_size=0.2, random_state=42)

    # Store the data
    train_data.to_json('./data/train_data.json', orient='records', lines=True)
    test_data.to_json('./data/test_data.json', orient='records', lines=True)

def get_data():
    df_reviews = pd.read_json('./data/raw/IMDB_reviews.json', lines=True) \
        .drop_duplicates('review_text').sample(frac=1)
    print('Shape of movie reviews dataframe:', df_reviews.shape)

    # Load movie details data
    df_movies = pd.read_json('./data/raw/IMDB_movie_details.json', lines=True)
    print('Shape of movie details dataframe:', df_movies.shape)

    df_merged = df_reviews.merge(df_movies, on="movie_id", how="left", suffixes=('_review', '_movie'))

    spoiler_counts = df_merged['is_spoiler'].value_counts()
    custom_labels = ['Non-Spoilers', 'Spoilers']

    # Create a bar chart
    plt.pie(spoiler_counts, labels=custom_labels, colors=['orange', 'blue'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


if __name__ == "__main__":
    preprocess_data()
    # get_data()
    # df_train = pd.read_json('./data/test_data.json', lines=True)
    #
    # total_records = df_train.shape[0]
    # unique_movie_ids = df_train['movie_id'].nunique()
    # print(df_train.head())
    # print(total_records)
    # print(unique_movie_ids)