# import spacy
# import pandas as pd
# from tqdm import tqdm

# def bin_age(age):
#     try:
#         age = int(age)
#         if age < 20:
#             return '<20'
#         elif age < 30:
#             return '20-29'
#         elif age < 40:
#             return '30-39'
#         elif age < 50:
#             return '40-49'
#         else:
#             return '50+'
#     except:
#         return 'Unknown'

# def analyze_aspects(df):
#     nlp = spacy.load('en_core_web_sm')
#     FEATURES = {
#         "camera": ["camera", "photo", "picture"],
#         "battery": ["battery", "charge", "charging"],
#         "performance": ["performance", "speed", "lag", "slow", "smooth"],
#         "display": ["screen", "display", "resolution"],
#         "sound": ["sound", "speaker", "audio"],
#         "design": ["design", "look", "build", "style"],
#         "price": ["price", "cost", "expensive", "cheap", "value"]
#     }
#     from spacy.matcher import PhraseMatcher
#     matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
#     for feature, terms in FEATURES.items():
#         matcher.add(feature, [nlp(term) for term in terms])
#     positive_words = {"good", "great", "excellent", "amazing", "awesome", "fantastic", "positive", "smooth"}
#     negative_words = {"bad", "terrible", "poor", "awful", "slow", "negative", "laggy", "issue", "problem"}
#     def extract_aspect_sentiment(text):
#         doc = nlp(str(text).lower())
#         matches = matcher(doc)
#         results = {}
#         for match_id, start, end in matches:
#             span = doc[start:end]
#             feature = nlp.vocab.strings[match_id]
#             sent = span.sent
#             sent_words = {token.lemma_ for token in sent}
#             polarity = "NEUTRAL"
#             if sent_words & positive_words:
#                 polarity = "POSITIVE"
#             if sent_words & negative_words:
#                 polarity = "NEGATIVE"
#             if polarity != "NEUTRAL":
#                 results[feature] = polarity
#         return results
#     tqdm.pandas()
#     df["aspect_sentiments"] = df["Cleaned Review"].progress_apply(extract_aspect_sentiment)
#     return df

# def group_sentiment_simple(df, group_col):
#     all_records = []
#     for i, row in df.iterrows():
#         aspects = row["aspect_sentiments"]
#         for feat, senti in aspects.items():
#             all_records.append({
#                 group_col: row.get(group_col, None),
#                 "sentiment": senti
#             })
#     group_df = pd.DataFrame(all_records)
#     if group_df.empty:
#         return []
#     summary = group_df.groupby([group_col, "sentiment"]).size().unstack(fill_value=0).reset_index()
#     summary = summary.rename(columns={group_col: group_col})
#     return summary.to_dict(orient="records")

# def overall_sentiment(df):
#     all_sentiments = []
#     for aspects in df["aspect_sentiments"]:
#         for feat, senti in aspects.items():
#             all_sentiments.append(senti)
#     from collections import Counter
#     counts = Counter(all_sentiments)
#     return [{"sentiment": k, "count": v} for k, v in counts.items()]

# def feature_summary(df):
#     all_records = []
#     for i, row in df.iterrows():
#         aspects = row["aspect_sentiments"]
#         for feat, senti in aspects.items():
#             all_records.append({"feature": feat, "sentiment": senti})
#     feature_df = pd.DataFrame(all_records)
#     if feature_df.empty:
#         return []
#     summary = feature_df.groupby(["feature", "sentiment"]).size().unstack(fill_value=0).reset_index()
#     return summary.to_dict(orient="records")















# backend/aspect_model.py

import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
from collections import Counter

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

def group_sentiment_simple(df, group_col):
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
    group_df = group_df.loc[:, ~group_df.columns.duplicated()]
    print("[DEBUG] Columns before groupby:", group_df.columns.tolist())
    group_cols = [group_col]
    # Only add 'Product Name' if it's not the group_col
    if group_col != "Product Name" and "Product Name" in group_df.columns:
        group_cols.append("Product Name")
    if "Month" in group_df.columns:
        group_cols.append("Month")
    group_cols.append("sentiment")
    summary = group_df.groupby(group_cols).size().unstack(fill_value=0).reset_index()
    return summary.to_dict(orient="records")

def overall_sentiment(df):
    sentiments = []
    for aspects in df.get("aspect_sentiments", []):
        for _, senti in aspects.items():
            sentiments.append(senti)
    counts = Counter(sentiments)
    result = []
    for prod, group in df.groupby(["Product Name", "Month"]):
        p, m = prod
        aspects = group["aspect_sentiments"]
        pos = sum(1 for a in aspects for v in a.values() if v == "POSITIVE")
        neg = sum(1 for a in aspects for v in a.values() if v == "NEGATIVE")
        result.append({"Product Name": p, "Month": m, "sentiment": "POSITIVE", "count": pos})
        result.append({"Product Name": p, "Month": m, "sentiment": "NEGATIVE", "count": neg})
    return result

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
    return summary.to_dict(orient="records")
