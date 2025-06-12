# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import threading
import uuid
import time
import random
from collections import Counter
import spacy
from spacy.matcher import PhraseMatcher

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Shared job states
progress_dict = {}
results_dict = {}

# Age binning
def bin_age(age):
    try:
        age = int(age)
        if age < 20: return '<20'
        elif age < 30: return '20-29'
        elif age < 40: return '30-39'
        elif age < 50: return '40-49'
        else: return '50+'
    except:
        return 'Unknown'

# NLP Aspect Analysis
def analyze_aspects(df):
    nlp = spacy.load('en_core_web_sm')
    FEATURES = {
        "camera": ["camera", "photo", "picture"],
        "battery": ["battery", "charge", "charging"],
        "performance": ["performance", "speed", "lag", "slow", "smooth"],
        "display": ["screen", "display", "resolution"],
        "sound": ["sound", "speaker", "audio"],
        "design": ["design", "look", "build", "style"],
        "price": ["price", "cost", "expensive", "cheap", "value"]
    }
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    for feature, terms in FEATURES.items():
        matcher.add(feature, [nlp(term) for term in terms])

    positive_words = {"good", "great", "excellent", "amazing", "awesome", "fantastic", "positive", "smooth"}
    negative_words = {"bad", "terrible", "poor", "awful", "slow", "negative", "laggy", "issue", "problem"}

    def extract_aspect_sentiment(text):
        doc = nlp(str(text).lower())
        matches = matcher(doc)
        results = {}
        for match_id, start, end in matches:
            feature = nlp.vocab.strings[match_id]
            sent = doc[start:end].sent
            words = {token.lemma_ for token in sent}
            polarity = "NEUTRAL"
            if words & positive_words:
                polarity = "POSITIVE"
            if words & negative_words:
                polarity = "NEGATIVE"
            if polarity != "NEUTRAL":
                results[feature] = polarity
        return results

    df["aspect_sentiments"] = df["Cleaned Review"].apply(extract_aspect_sentiment)
    return df

# Group sentiment helper
def group_sentiment(df, group_col):
    all_records = []
    for _, row in df.iterrows():
        aspects = row.get("aspect_sentiments", {})
        for feature, senti in aspects.items():
            record = {
                group_col: row.get(group_col, None),
                "Product Name": row.get("Product Name", ""),
                "Month": row.get("Month", ""),
                "sentiment": senti
            }
            all_records.append(record)

    if not all_records:
        return []

    group_df = pd.DataFrame(all_records)
    group_cols = [group_col]
    if group_col != "Product Name" and "Product Name" in group_df.columns:
        group_cols.append("Product Name")
    if "Month" in group_df.columns:
        group_cols.append("Month")
    group_cols.append("sentiment")

    summary = group_df.groupby(group_cols).size().unstack(fill_value=0).reset_index()
    summary.columns = ['_'.join([str(c) for c in col]) if isinstance(col, tuple) else str(col) for col in summary.columns]
    return summary.to_dict(orient="records")

# Overall sentiment aggregation
def overall_sentiment(df):
    result = []
    for prod, group in df.groupby(["Product Name", "Month"]):
        p, m = prod
        aspects = group["aspect_sentiments"]
        pos = sum(1 for a in aspects for v in a.values() if v == "POSITIVE")
        neg = sum(1 for a in aspects for v in a.values() if v == "NEGATIVE")
        result.append({"Product Name": p, "Month": m, "sentiment": "POSITIVE", "count": pos})
        result.append({"Product Name": p, "Month": m, "sentiment": "NEGATIVE", "count": neg})
    return result

# Feature-wise summary
def feature_summary(df):
    all_records = []
    for _, row in df.iterrows():
        for feat, senti in row.get("aspect_sentiments", {}).items():
            all_records.append({
                "feature": feat,
                "sentiment": senti,
                "Product Name": row.get("Product Name", ""),
                "Month": row.get("Month", "")
            })
    if not all_records:
        return []
    df_feat = pd.DataFrame(all_records)
    summary = df_feat.groupby(["feature", "Product Name", "Month", "sentiment"]).size().unstack(fill_value=0).reset_index()
    summary.columns = ['_'.join([str(c) for c in col]) if isinstance(col, tuple) else str(col) for col in summary.columns]
    return summary.to_dict(orient="records")


def analyze_job(job_id, df):
    try:
        total_time = random.uniform(90, 160)
        progress = 0
        start = time.time()

        while progress < 90:
            elapsed = time.time() - start
            remaining = total_time - elapsed
            sleep_time = random.uniform(0.3, 1.3)
            increment = random.randint(1, max(1, int(10 - progress / 15)))
            progress = min(progress + increment, 90)
            progress_dict[job_id] = progress
            if elapsed + sleep_time > total_time:
                sleep_time = max(0.1, remaining)
            time.sleep(sleep_time)
            if elapsed >= total_time:
                break

        # Actual analysis work
        df = analyze_aspects(df)
        result = {
            'feature_summary': feature_summary(df),
            'overall_summary': overall_sentiment(df),
            'sentiment_by_brand': group_sentiment(df, 'Brand'),
            'sentiment_by_product': group_sentiment(df, 'Product Name'),
            'sentiment_by_rating': group_sentiment(df, 'Rating'),
            'sentiment_by_platform': group_sentiment(df, 'Platform'),
            'sentiment_by_gender': group_sentiment(df, 'Gender'),
            'sentiment_by_verified': group_sentiment(df, 'Verified Purchase'),
            'sentiment_by_age': group_sentiment(df, 'Age')
        }

        progress_dict[job_id] = 100
        results_dict[job_id] = result

    except Exception as e:
        progress_dict[job_id] = -1
        results_dict[job_id] = {'error': str(e)}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400

    if df.empty or 'Cleaned Review' not in df.columns:
        return jsonify({'error': 'CSV file is missing required data.'}), 400

    if 'Date' in df.columns:
        df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m')
    if 'Age' in df.columns:
        df['Age'] = df['Age'].apply(bin_age)

    job_id = str(uuid.uuid4())
    progress_dict[job_id] = 0
    thread = threading.Thread(target=analyze_job, args=(job_id, df))
    thread.start()

    return jsonify({'job_id': job_id}), 202

@app.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    progress = progress_dict.get(job_id, 0)
    return jsonify({'progress': progress})

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    result = results_dict.get(job_id)
    if result is None:
        return jsonify({'error': 'Result not ready'}), 202
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
