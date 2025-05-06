from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("sample_resume.txt", "r", encoding="utf-8") as f:
    resume = f.read()

with open("sample_jd.txt", "r", encoding="utf-8") as f:
    jd = f.read()

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume, jd])
score = cosine_similarity(vectors[0:1], vectors[1:2])
print(f"Match Score: {round(score[0][0]*100, 2)}%")
