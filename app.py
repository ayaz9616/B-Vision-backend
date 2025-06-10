# from flask import Flask, request, jsonify
# import pandas as pd
# from aspect_model import analyze_aspects, group_sentiment_simple, overall_sentiment, feature_summary, bin_age
# import os
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if not file:
#         return jsonify({'error': 'No file uploaded'}), 400
#     try:
#         df = pd.read_csv(file)
#     except pd.errors.EmptyDataError:
#         return jsonify({'error': 'Uploaded file is empty or not a valid CSV.'}), 400
#     except Exception as e:
#         return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400
#     if df.empty or len(df.columns) == 0:
#         return jsonify({'error': 'CSV file has no data or columns.'}), 400
#     # Bin age
#     if 'Age' in df.columns:
#         df['Age'] = df['Age'].apply(bin_age)
#     df = analyze_aspects(df)
#     # Feature summary
#     feature_data = feature_summary(df)
#     # Overall summary
#     overall_data = overall_sentiment(df)
#     # Grouped summaries
#     brand_data = group_sentiment_simple(df, 'Brand') if 'Brand' in df.columns else []
#     product_data = group_sentiment_simple(df, 'Product Name') if 'Product Name' in df.columns else []
#     rating_data = group_sentiment_simple(df, 'Rating') if 'Rating' in df.columns else []
#     platform_data = group_sentiment_simple(df, 'Platform') if 'Platform' in df.columns else []
#     gender_data = group_sentiment_simple(df, 'Gender') if 'Gender' in df.columns else []
#     verified_data = group_sentiment_simple(df, 'Verified Purchase') if 'Verified Purchase' in df.columns else []
#     age_data = group_sentiment_simple(df, 'Age') if 'Age' in df.columns else []
#     return jsonify({
#         'feature_summary': feature_data,
#         'overall_summary': overall_data,
#         'sentiment_by_brand': brand_data,
#         'sentiment_by_product': product_data,
#         'sentiment_by_rating': rating_data,
#         'sentiment_by_platform': platform_data,
#         'sentiment_by_gender': gender_data,
#         'sentiment_by_verified': verified_data,
#         'sentiment_by_age': age_data
#     })

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)










# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import threading
import uuid
import time
import random

from aspect_model import (
    analyze_aspects,
    group_sentiment_simple,
    overall_sentiment,
    feature_summary,
    bin_age
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progress_dict = {}
results_dict = {}

def analyze_job(job_id, df):
    try:
        import random
        import math
        total_time = random.uniform(120, 180)  # 2 to 3 minutes in seconds
        progress = 0
        start = time.time()
        # We'll do 60-90 steps, each with a random sleep and increment
        while progress < 90:
            elapsed = time.time() - start
            # Calculate remaining time and progress
            remaining_time = max(0.1, total_time - elapsed)
            # Irregular sleep: 0.3 to 1.5s
            sleep_time = random.uniform(0.3, 1.5)
            # Irregular increment: 1 to 5, but slow down as we approach 90
            max_inc = max(1, int(8 - (progress / 15)))
            step = random.randint(1, max_inc)
            progress = min(progress + step, 90)
            progress_dict[job_id] = progress
            # If we're running out of time, adjust sleep to finish at ~1 min
            if elapsed + sleep_time > total_time:
                sleep_time = max(0.1, total_time - elapsed)
            time.sleep(sleep_time)
            if time.time() - start >= total_time:
                break
        # Pause at 90% until analysis is actually done
        df = analyze_aspects(df)
        feature = feature_summary(df)
        overall = overall_sentiment(df)
        brand = group_sentiment_simple(df, 'Brand')
        product = group_sentiment_simple(df, 'Product Name')
        rating = group_sentiment_simple(df, 'Rating')
        platform = group_sentiment_simple(df, 'Platform')
        gender = group_sentiment_simple(df, 'Gender')
        verified = group_sentiment_simple(df, 'Verified Purchase')
        age = group_sentiment_simple(df, 'Age')
        progress_dict[job_id] = 100
        results_dict[job_id] = {
            'feature_summary': feature,
            'overall_summary': overall,
            'sentiment_by_brand': brand,
            'sentiment_by_product': product,
            'sentiment_by_rating': rating,
            'sentiment_by_platform': platform,
            'sentiment_by_gender': gender,
            'sentiment_by_verified': verified,
            'sentiment_by_age': age
        }
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
        print(f"[ERROR] Failed to read CSV: {e}")
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400
    if df.empty or 'Cleaned Review' not in df.columns:
        print("[ERROR] CSV file is missing required data.")
        return jsonify({'error': 'CSV file is missing required data.'}), 400
    # Parse date → month
    if 'Date' in df.columns:
        df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m')
    # Bin age
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
