# rumor_dynamics_utils.py
# A universal toolbox for rumor dynamics analysis, containing shared utilities for
# both Weibo and Twitter datasets.

import re
import pymysql
import jieba.analyse
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from collections import Counter
from datetime import datetime
from scipy.special import comb
from scipy.stats import skewnorm
from skfuzzy import cmeans
from sklearn.feature_extraction.text import TfidfVectorizer
import skfuzzy.control as ctrl
from typing import List, Set, Dict

# ==============================================================================
# SECTION 1: GENERIC EVOLUTIONARY GAME & COGNITION COMPONENTS
# These components are dataset-agnostic and form the core of the model.
# ==============================================================================

class FuzzySystem:
    """
    A class that encapsulates the fuzzy logic control system for calculating the
    Cognitive Bias (CB) score based on PI, SE, and IM inputs.
    """
    def __init__(self):
        """Initializes the fuzzy variables, membership functions, and rules."""
        x_PI_range = np.arange(0, 1.01, 0.1, np.float32)
        x_SE_range = np.arange(0, 1.01, 0.1, np.float32)
        x_IM_range = np.arange(0, 1.01, 0.1, np.float32)
        y_CB_range = np.arange(0, 1.01, 0.1, np.float32)

        self.x_PI = ctrl.Antecedent(x_PI_range, 'PI')
        self.x_SE = ctrl.Antecedent(x_SE_range, 'SE')
        self.x_IM = ctrl.Antecedent(x_IM_range, 'IM')
        self.y_CB = ctrl.Consequent(y_CB_range, 'CB')

        for var in [self.x_PI, self.x_SE, self.x_IM, self.y_CB]:
            var['L'] = fuzz.trapmf(var.universe, [0, 0, 0.35, 0.45])
            var['M'] = fuzz.trapmf(var.universe, [0.35, 0.45, 0.65, 0.75])
            var['H'] = fuzz.trapmf(var.universe, [0.65, 0.75, 1, 1])
        
        rule_L = ctrl.Rule(self.x_PI['L'] | self.x_SE['L'], self.y_CB['L'])
        rule_M = ctrl.Rule(self.x_PI['M'] & self.x_SE['M'], self.y_CB['M'])
        rule_H = ctrl.Rule(self.x_PI['H'] | self.x_SE['H'] | self.x_IM['H'], self.y_CB['H'])
        
        self.system = ctrl.ControlSystem([rule_L, rule_M, rule_H])
        self.sim = ctrl.ControlSystemSimulation(self.system)
        self.y_CB.defuzzify_method = 'centroid'

    def calculate_cb(self, PI: float, SE: float, IM: float) -> float:
        """Calculates the crisp CB value for a given set of inputs."""
        self.sim.input['PI'] = max(0, min(1, PI))
        self.sim.input['SE'] = max(0, min(1, SE))
        self.sim.input['IM'] = max(0, min(1, IM))
        self.sim.compute()
        return self.sim.output['CB']

def run_cognition_modeling(feature_filepath: str, optimal_clusters: int = 4) -> tuple:
    """
    Performs Fuzzy C-Means clustering and updates user cognition values.
    This function is generic and works as long as the feature file has a 'CB' column.
    """
    try:
        df = pd.read_csv(feature_filepath, sep=' ', header=0, on_bad_lines='skip', quotechar='"')
        user_cognition = df['CB'].to_numpy()
        user_list = df['id'].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        print(f"Warning: Could not load feature data from {feature_filepath}. {e}")
        return np.array([]), [], np.array([])

    if len(user_cognition) < optimal_clusters:
        print(f"Warning: Not enough data for clustering. Returning raw cognition.")
        return user_cognition, user_list, user_cognition

    data_reshaped = np.expand_dims(user_cognition, axis=0)
    centers, membership, _, _, _, _, _ = cmeans(data_reshaped, optimal_clusters, m=2, error=1e-5, maxiter=1000)
    centers = centers.flatten()
    user_group_cognition = centers[np.argmax(membership, axis=0)]
    social_cognition = np.sum(np.mean(membership, axis=1) * centers)
    
    # Simplified cognition update model
    updated_user_cognition = user_cognition * 0.9 + user_group_cognition * 0.05 + social_cognition * 0.05
    return user_cognition, user_list, np.clip(updated_user_cognition, 0, 1)

def double_exponential_decay(t: float, c1: float, c2: float, lambda1: float, lambda2: float) -> float:
    """Calculates topic popularity decay over time."""
    return c1 * np.exp(-lambda1 * t) + c2 * np.exp(-lambda2 * t)

def calculate_driving_forces(pi_R: float, pi_A: float) -> tuple:
    """Calculates driving forces for adopting rumor or anti-rumor strategy."""
    diff = pi_R - pi_A
    return (1 / (1 + np.exp(-diff)), 1 / (1 + np.exp(diff))) if abs(diff) < 700 else (1.0, 0.0) if diff > 0 else (0.0, 1.0)

# ==============================================================================
# SECTION 2: WEIBO-SPECIFIC UTILITIES
# ==============================================================================

class MysqlConn:
    """A class to handle MySQL database connections for the Weibo dataset."""
    def __init__(self, host="localhost", user="root", passwd="mysql123", database="aminer_weibo"):
        try:
            self.mydb = pymysql.connect(host=host, user=user, passwd=passwd, database=database, charset='utf8')
            self.mycursor = self.mydb.cursor()
        except pymysql.err.OperationalError as e:
            print(f"FATAL: Error connecting to MySQL Database: {e}")
            raise

    def select(self, sql: str, values: tuple = None):
        """Executes a SELECT query and fetches all results."""
        self.mycursor.execute(sql, values)
        return self.mycursor.fetchall()

    def close(self):
        """Closes the cursor and the database connection."""
        self.mycursor.close()
        self.mydb.close()

class TFIDF_Weibo:
    """A TF-IDF keyword extraction tool specifically for Chinese text using jieba."""
    def __init__(self, stopwords_path='data/stopwords_CN.txt'):
        try:
            jieba.analyse.set_stop_words(stopwords_path)
        except:
            print(f"Warning: Could not load Chinese stopwords from {stopwords_path}")
            
    def keyword(self, content: str, topK: int = 10) -> list:
        """Extracts top K keywords from Chinese text."""
        return jieba.analyse.extract_tags(content, topK=topK)

# ==============================================================================
# SECTION 3: TWITTER-SPECIFIC UTILITIES
# ==============================================================================

def calculate_twitter_resonance_scores(content_series: pd.Series, top_n_keywords: int = 15) -> pd.Series:
    """
    Calculates content resonance (Jaccard similarity of keywords) for English texts.
    """
    try:
        with open('data/stopwords_E.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().strip().split('\n'))
    except FileNotFoundError:
        print("Warning: stopwords_E.txt not found. Proceeding without stopwords.")
        stop_words = set()

    def preprocess_text(text: str) -> list:
        text = str(text)
        text = re.sub(r'http\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return [word for word in text.split() if word not in stop_words]

    def extract_keywords(texts: list, top_n: int) -> set:
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english")
        try:
            vectorizer.fit_transform(texts)
            return set(vectorizer.get_feature_names_out())
        except ValueError:
            return set()

    if content_series.empty:
        return pd.Series([], dtype=float)

    topic_keywords = extract_keywords([str(content_series.iloc[0])], top_n_keywords)
    processed_texts = content_series.apply(preprocess_text)
    user_keywords_series = processed_texts.apply(lambda x: set([word for word, _ in Counter(x).most_common(top_n_keywords)]))
    
    return user_keywords_series.apply(lambda uk: len(uk & topic_keywords) / len(uk | topic_keywords) if (uk | topic_keywords) else 0.0)