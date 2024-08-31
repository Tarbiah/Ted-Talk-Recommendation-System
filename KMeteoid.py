from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids  # Import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt

# Additional import for KMedoids
from sklearn_extra.cluster import KMedoids

nltk.download('stopwords')

app = Flask(__name__)  # Fix the typo in the Flask app initialization


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
df = pd.read_csv('cleaned_ted_data.csv')
df['details'] = df['title'] + ' ' + df['details']
df['details'] = df['details'].apply(lambda text: remove_stopwords(text))
df['details'] = df['details'].apply(lambda x: cleaning_punctuations(x))

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(df['details'])

# Dimensionality reduction using PCA
pca = PCA(n_components=100)  # You can adjust the number of components
X_pca = pca.fit_transform(X.toarray())

# Feature standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Train a K-Medoids clustering model
num_clusters = 5  # You can adjust the number of clusters based on your data
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42,
                    metric='cosine')  # Specify 'cosine' metric
df['cluster'] = kmedoids.fit_predict(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['details'], df['cluster'], test_size=0.2, random_state=42)

# Train the model using the training set
X_train_vectorized = vectorizer.transform(X_train)
X_train_pca = pca.transform(X_train_vectorized.toarray())
X_train_scaled = scaler.transform(X_train_pca)

kmedoids.fit(X_train_scaled)

silhouette_avg = silhouette_score(X_scaled, df['cluster'])
print(f"Silhouette Score: {silhouette_avg}")


def get_similar_talks_in_cluster(talk_content, cluster, data=df):
    talk_array1 = vectorizer.transform(
        [talk_content]).toarray()  # Wrap talk_content in a list
    talk_array1_pca = pca.transform(talk_array1)
    talk_array1_scaled = scaler.transform(talk_array1_pca)

    sim = []
    for _, row in data[data['cluster'] == cluster].iterrows():
        details = row['details']
        talk_array2 = vectorizer.transform(
            [details]).toarray()  # Wrap details in a list
        talk_array2_pca = pca.transform(talk_array2)
        talk_array2_scaled = scaler.transform(talk_array2_pca)

        cos_sim = cosine_similarity(
            talk_array1_scaled, talk_array2_scaled)[0][0]
        sim.append(cos_sim)
    return sim


def recommend_talks(talk_content, data=df):
    cluster = kmedoids.predict(scaler.transform(
        pca.transform(vectorizer.transform([talk_content[0]]).toarray())))  # Extract the first element of the list
    sim_in_cluster = get_similar_talks_in_cluster(
        talk_content[0], cluster[0], data)

    # Create a new DataFrame with relevant information
    recommendations_data = pd.DataFrame({
        'main_speaker': data[data['cluster'] == cluster[0]]['main_speaker'],
        'title': data[data['cluster'] == cluster[0]]['title'],
        'url': data[data['cluster'] == cluster[0]]['url'],
        'cos_sim': sim_in_cluster
    })

    # Sort by cosine similarity and shuffle the DataFrame
    recommendations_data = recommendations_data.sample(
        frac=1, random_state=None)

    # Get all recommendations
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
def get_recommendations():
    user_input = request.form['user_input']
    talk_content = [user_input]
    recommendations = recommend_talks(talk_content)
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
