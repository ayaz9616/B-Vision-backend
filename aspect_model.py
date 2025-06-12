import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
from collections import Counter

class AspectAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.FEATURES = {
            "camera": ["camera", "photo", "picture"],
            "battery": ["battery", "charge", "charging"],
            "performance": ["performance", "speed", "lag", "slow", "smooth"],
            "display": ["screen", "display", "resolution"],
            "sound": ["sound", "speaker", "audio"],
            "design": ["design", "look", "build", "style"],
            "price": ["price", "cost", "expensive", "cheap", "value"]
        }
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        for feature, terms in self.FEATURES.items():
            self.matcher.add(feature, [self.nlp(term) for term in terms])

        self.positive_words = {"good", "great", "excellent", "amazing", "awesome", "fantastic", "positive", "smooth"}
        self.negative_words = {"bad", "terrible", "poor", "awful", "slow", "negative", "laggy", "issue", "problem"}

    def bin_age(self, age):
        try:
            age = int(age)
            if age < 20: return '<20'
            elif age < 30: return '20-29'
            elif age < 40: return '30-39'
            elif age < 50: return '40-49'
            else: return '50+'
        except:
            return 'Unknown'

    def analyze_reviews(self, df, job_id=None, progress_dict=None):
        total = len(df)
        aspect_results = []

        for idx, row in df.iterrows():
            doc = self.nlp(str(row["Cleaned Review"]).lower())
            matches = self.matcher(doc)
            aspects = {}

            for match_id, start, end in matches:
                feature = self.nlp.vocab.strings[match_id]
                sent = doc[start:end].sent
                words = {token.lemma_ for token in sent}
                polarity = "NEUTRAL"
                if words & self.positive_words:
                    polarity = "POSITIVE"
                if words & self.negative_words:
                    polarity = "NEGATIVE"
                if polarity != "NEUTRAL":
                    aspects[feature] = polarity

            aspect_results.append(aspects)

            if job_id and progress_dict is not None:
                progress_dict[job_id] = int(((idx + 1) / total) * 80)

        df["aspect_sentiments"] = aspect_results
        return df

    def group_sentiment(self, df, group_col):
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
        group_cols = [group_col, "Product Name", "Month", "sentiment"]
        summary = group_df.groupby(group_cols).size().unstack(fill_value=0).reset_index()
        return summary.to_dict(orient="records")

    def overall_sentiment(self, df):
        result = []
        for prod, group in df.groupby(["Product Name", "Month"]):
            p, m = prod
            aspects = group["aspect_sentiments"]
            pos = sum(1 for a in aspects for v in a.values() if v == "POSITIVE")
            neg = sum(1 for a in aspects for v in a.values() if v == "NEGATIVE")
            result.append({"Product Name": p, "Month": m, "sentiment": "POSITIVE", "count": pos})
            result.append({"Product Name": p, "Month": m, "sentiment": "NEGATIVE", "count": neg})
        return result

    def feature_summary(self, df):
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
