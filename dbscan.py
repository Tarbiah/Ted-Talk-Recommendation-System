from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import random

nltk.download('stopwords')

app = Flask(__name__)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    imp_words = [word.lower() for word in str(text).split()
                 if word.lower() not in stop_words]
    return " ".join(imp_words)


def cleaning_punctuations(text):
    punctuations_list = string.punctuation
    signal = str.maketrans('', '', punctuations_list)
    return text.translate(signal)


# Load the dataset and perform necessary preprocessing
df = pd.read_csv('tedx_dataset.csv')
df['details'] = df['title'] + ' ' + df['details']
df['details'] = df['details'].apply(lambda text: remove_stopwords(text))
df['details'] = df['details'].apply(lambda x: cleaning_punctuations(x))

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(df['details'])

# Train a DBScan clustering model
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(X)

# Silhouette Score for DBScan
silhouette_avg = silhouette_score(X, df['cluster'])
print(f"Silhouette Score (DBScan): {silhouette_avg}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['details'], df['cluster'], test_size=0.2, random_state=42)

# Train the model using the training set
X_train_vectorized = vectorizer.transform(X_train)
dbscan.fit(X_train_vectorized)


def get_similar_talks_in_cluster(talk_content, cluster, data=df):
    talk_array1 = vectorizer.transform(talk_content).toarray()
    sim = []
    for _, row in data[data['cluster'] == cluster].iterrows():
        details = row['details']
        talk_array2 = vectorizer.transform([details]).toarray()
        cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]
        sim.append(cos_sim)
    return sim


def recommend_talks_dbscan(talk_content, data=df):
    cluster_labels = dbscan.fit_predict(vectorizer.transform(talk_content))
    cluster = cluster_labels[0]

    if cluster == -1:
        # Noisy samples are assigned the label -1 by DBScan, handle accordingly
        return pd.DataFrame(columns=['main_speaker', 'title', 'url', 'cos_sim'])

    sim_in_cluster = get_similar_talks_in_cluster(
        talk_content, cluster, data)

    recommendations_data = pd.DataFrame({
        'main_speaker': data[data['cluster'] == cluster]['main_speaker'],
        'title': data[data['cluster'] == cluster]['title'],
        'url': data[data['cluster'] == cluster]['url'],
        'cos_sim': sim_in_cluster
    })

    recommendations_data = recommendations_data.sample(
        frac=1, random_state=None)
    recommendations_data.sort_values(
        by=['cos_sim'], ascending=[False], inplace=True)
    recommendations = recommendations_data[['main_speaker', 'title', 'url']]

    return recommendations


@app.route('/')
def index():
    random_value = random.random()
    return render_template('index.html', random_value=random_value)

# Add this route to your Flask application


@app.route('/recommendations', methods=['POST'])
def get_recommendations_dbscan():
    user_input = request.form['user_input']
    talk_content = [user_input]
    recommendations = recommend_talks_dbscan(talk_content)
    return render_template('recommendations.html', recommendations=recommendations)

# Add this route to your Flask application


@app.route('/open_url', methods=['POST'])
def open_url():
    url = request.form['url']
    # You can use a library like webbrowser to open the URL in the default web browser
    # For simplicity, I'm using a simple HTML response here
    return f'<html><body><script>window.open("{url}", "_blank");</script></body></html>'


if __name__ == '__main__':
    app.run(debug=True)
