from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

def get_top_tfidf_keywords(docs, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(docs)
    indices = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = feature_names[indices][:top_n]
    top_scores = np.asarray(X.sum(axis=0)).ravel()[indices][:top_n]
    return list(zip(top_features, top_scores))

def bert_cluster_job_descriptions(docs, n_clusters=6, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, embeddings

def extractive_summary_tfidf(text, n_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return text
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(sentence_vectors)
    scores = sim_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1][:n_sentences]]
    summary = ' '.join(ranked_sentences)
    return summary
