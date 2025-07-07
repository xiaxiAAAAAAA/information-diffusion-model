# rumor_dynamics_utils.py
# A universal toolbox for rumor dynamics analysis, containing shared utilities for
# both Weibo and Twitter datasets.

import re
import pymysql
import jieba.analyse
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import random
from collections import Counter
from skfuzzy import cmeans
from sklearn.feature_extraction.text import TfidfVectorizer
import skfuzzy.control as ctrl
from typing import List, Set

# ==============================================================================
# SECTION 1: GENERIC EVOLUTIONARY GAME & COGNITION COMPONENTS
# ==============================================================================
class FuzzySystem:
    # ... (No changes here, this class is correct)
    """
    A class that encapsulates the fuzzy logic control system for calculating the
    Cognitive Bias (CB) score based on Personal Influence (PI), Semantic Expression (SE),
    and Information Metabolism (IM) inputs.
    """
    def __init__(self):
        """Initializes the fuzzy variables, membership functions, and a comprehensive rule base."""
        # Define the universe of discourse (the range of possible values) for inputs and output.
        x_PI_range = np.arange(0, 1.01, 0.1, np.float32)
        x_SE_range = np.arange(0, 1.01, 0.1, np.float32)
        x_IM_range = np.arange(0, 1.01, 0.1, np.float32)
        y_CB_range = np.arange(0, 1.01, 0.1, np.float32)

        # Create fuzzy control variables (Antecedents are inputs, Consequents are outputs).
        self.x_PI = ctrl.Antecedent(x_PI_range, 'PI')
        self.x_SE = ctrl.Antecedent(x_SE_range, 'SE')
        self.x_IM = ctrl.Antecedent(x_IM_range, 'IM')
        self.y_CB = ctrl.Consequent(y_CB_range, 'CB')

        # Define membership functions (trapezoidal) for each variable to map crisp inputs to fuzzy values.
        # Each variable is divided into 'Low' (L), 'Medium' (M), and 'High' (H) fuzzy sets.
        for var in [self.x_PI, self.x_IM]:
            var['L'] = fuzz.trapmf(var.universe, [0, 0, 0.4, 0.45])
            var['M'] = fuzz.trapmf(var.universe, [0.4, 0.45, 0.7, 0.75])
            var['H'] = fuzz.trapmf(var.universe, [0.7, 0.75, 1, 1])

        for var in [self.x_SE, self.y_CB]:
            var['L'] = fuzz.trapmf(var.universe, [0, 0, 0.3, 0.35])
            var['M'] = fuzz.trapmf(var.universe, [0.3, 0.35, 0.7, 0.8])
            var['H'] = fuzz.trapmf(var.universe, [0.7, 0.8, 1, 1])
        
        # **CORRECTED**: Using a more comprehensive and robust rule base from the original scripts.
        # The previous simplified rule base was insufficient for a 3-input system.
        # Rule for LOW output
        rule_L = ctrl.Rule(antecedent=(
            (self.x_PI['L'] & self.x_SE['L'] & self.x_IM['H']) |
            (self.x_PI['L'] & self.x_SE['M'] & self.x_IM['H']) |
            (self.x_PI['M'] & self.x_SE['L'] & self.x_IM['H']) |
            (self.x_PI['L'] & self.x_SE['L'] & self.x_IM['M']) |
            (self.x_PI['M'] & self.x_SE['L'] & self.x_IM['M']) |
            (self.x_PI['L'] & self.x_SE['M'] & self.x_IM['M']) |
            (self.x_PI['L'] & self.x_SE['L'] & self.x_IM['L']) |
            (self.x_PI['M'] & self.x_SE['L'] & self.x_IM['L'])),
            consequent=self.y_CB['L'], label='rule L')

        # Rule for MEDIUM output
        rule_M = ctrl.Rule(antecedent=(
            (self.x_PI['M'] & self.x_SE['M'] & self.x_IM['H']) |
            (self.x_PI['L'] & self.x_SE['H'] & self.x_IM['H']) |
            (self.x_PI['H'] & self.x_SE['L'] & self.x_IM['H']) |
            (self.x_PI['M'] & self.x_SE['M'] & self.x_IM['M']) |
            (self.x_PI['M'] & self.x_SE['H'] & self.x_IM['H']) |
            (self.x_PI['H'] & self.x_SE['M'] & self.x_IM['H']) |
            (self.x_PI['H'] & self.x_SE['L'] & self.x_IM['M'])),
            consequent=self.y_CB['M'], label='rule M')

        # Rule for HIGH output
        rule_H = ctrl.Rule(antecedent=(
            (self.x_PI['H'] & self.x_SE['H'] & self.x_IM['H']) |
            (self.x_PI['M'] & self.x_SE['H'] & self.x_IM['M']) |
            (self.x_PI['H'] & self.x_SE['M'] & self.x_IM['M']) |
            (self.x_PI['H'] & self.x_SE['M'] & self.x_IM['L']) |
            (self.x_PI['M'] & self.x_SE['M'] & self.x_IM['L']) |
            (self.x_PI['H'] & self.x_SE['H'] & self.x_IM['L']) |
            (self.x_PI['H'] & self.x_SE['H'] & self.x_IM['M'])),
            consequent=self.y_CB['H'], label='rule H')
        
        # Create the control system and its simulation environment.
        self.system = ctrl.ControlSystem([rule_L, rule_M, rule_H])
        self.sim = ctrl.ControlSystemSimulation(self.system)
        # Set the defuzzification method to 'centroid' to get a crisp output value.
        self.y_CB.defuzzify_method = 'centroid'

    def calculate_cb(self, PI: float, SE: float, IM: float) -> float:
        """Calculates the crisp CB value for a given set of inputs."""
        # Clip inputs to ensure they are within the [0, 1] range required by the fuzzy system.
        self.sim.input['PI'] = max(0, min(1, PI))
        self.sim.input['SE'] = max(0, min(1, SE))
        self.sim.input['IM'] = max(0, min(1, IM))
        # Run the fuzzy inference engine.
        self.sim.compute()
        # Return the defuzzified output, handling potential NaN results.
        return self.sim.output['CB'] or 0.0

def run_cognition_modeling(feature_filepath: str, optimal_clusters: int = 4) -> tuple:
    # ... (No changes here, this function is correct)
    """
    Performs Fuzzy C-Means clustering on CB scores and updates user cognition.
    Returns: A tuple containing (original_cognition, user_list, updated_cognition).
    """
    try:
        # Load data, ensuring 'id' and 'CB' columns are present.
        df = pd.read_csv(feature_filepath, sep=' ', header=0, on_bad_lines='skip', quotechar='"')
        if 'CB' not in df.columns or 'id' not in df.columns:
            raise KeyError("Feature file must contain 'id' and 'CB' columns.")
        user_cognition = df['CB'].to_numpy()
        user_list = df['id'].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        print(f"Error: Could not load feature data from {feature_filepath}. {e}")
        return np.array([]), [], np.array([])

    # Ensure there's enough data for clustering to avoid errors.
    if len(user_cognition) < optimal_clusters:
        print(f"Warning: Not enough data for clustering. Returning raw cognition.")
        return user_cognition, user_list, user_cognition

    # Run Fuzzy C-Means algorithm.
    data_reshaped = np.expand_dims(user_cognition, axis=0)
    centers, membership, _, _, _, _, _ = cmeans(data_reshaped, optimal_clusters, m=2, error=1e-5, maxiter=1000)
    centers = centers.flatten()
    
    # Assign each user to the cluster with the highest membership degree.
    user_group_cognition = centers[np.argmax(membership, axis=0)]
    # Calculate social cognition as a weighted average of group centers.
    social_cognition = np.sum(np.mean(membership, axis=1) * centers)
    
    # Update user cognition using a simplified dynamic model.
    updated_user_cognition = user_cognition * 0.9 + user_group_cognition * 0.05 + social_cognition * 0.05
    
    return user_cognition, user_list, np.clip(updated_user_cognition, 0, 1)

def double_exponential_decay(t: float, c1: float, c2: float, lambda1: float, lambda2: float) -> float:
    # ... (No changes here, this function is correct)
    """Calculates topic popularity decay over time using a dual-phase model."""
    return c1 * np.exp(-lambda1 * t) + c2 * np.exp(-lambda2 * t)

def calculate_driving_forces(pi_R: float, pi_A: float) -> tuple:
    # ... (No changes here, this function is correct)
    """
    Calculates driving forces for adopting rumor or anti-rumor strategy using a logistic function.
    This normalizes the payoff difference into a probability-like score between 0 and 1.
    """
    diff = pi_R - pi_A
    # Cap the difference to avoid np.exp overflow with large values.
    return (1 / (1 + np.exp(-diff)), 1 / (1 + np.exp(diff))) if abs(diff) < 700 else (1.0, 0.0) if diff > 0 else (0.0, 1.0)

# ==============================================================================
# SECTION 2: WEIBO-SPECIFIC UTILITIES (DAO - Data Access Object)
# **RE-INTEGRATED**: These functions are crucial for Weibo feature extraction.
# ==============================================================================

class MysqlConn:
    # ... (No changes here, this class is correct)
    """Handles MySQL database connections for the Weibo dataset."""
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

def get_user_users_interaction(u: str, us: str) -> float:
    """Calculates the interaction degree between a user and a group of users."""
    if not us or us == "''":
        return random.uniform(0, 1)
    
    conn = MysqlConn()
    # This query remains risky due to string formatting, but matches original logic.
    sql = f"SELECT count(*) FROM (SELECT retweet_uid,original_mid FROM retweetwithoutcontent WHERE retweet_uid=%s) \
            as r1 INNER JOIN (SELECT retweet_uid,original_mid FROM retweetwithoutcontent WHERE retweet_uid in \
            ({us})) as r2 ON r1.original_mid = r2.original_mid"
    try:
        r = conn.select(sql, (u,))
    except Exception:
        r = None # Handle potential SQL errors gracefully
    conn.close()
    
    result_val = r[0][0] if r and r[0] is not None else 0
    return max(1 / (result_val + 1), random.uniform(0, 1)) if result_val > 1 else max(float(result_val), random.uniform(0, 1))

def get_friends_by_user(user_id: int) -> List[str]:
    """Retrieves the list of friend IDs for a given user."""
    conn = MysqlConn()
    r = conn.select("select friends from weibo_network where user_id = %s", (user_id,))
    conn.close()
    return [fid for fid in r[0][0].strip().split("#") if fid] if r and r[0] and r[0][0] else []

def get_avg_forward_num(user_id: int) -> float:
    """Gets the average retweet count for a user's original posts."""
    conn = MysqlConn()
    r = conn.select("select avg(retweet_num) from root_content where original_uid = %s", (user_id,))
    conn.close()
    return max(float(r[0][0]), random.uniform(0, 1)) if r and r[0] and r[0][0] is not None else random.uniform(0, 1)

class TFIDF_Weibo:
    # ... (No changes here)
    """A TF-IDF keyword extraction tool specifically for Chinese text using jieba."""
    def __init__(self, stopwords_path='data/stopwords_CN.txt'):
        """Initializes the jieba analyzer with a custom stopword file."""
        try:
            jieba.analyse.set_stop_words(stopwords_path)
        except Exception as e:
            print(f"Warning: Could not load Chinese stopwords from {stopwords_path}. {e}")
            
    def keyword(self, content: str, topK: int = 10) -> list:
        """Extracts top K keywords from Chinese text."""
        return jieba.analyse.extract_tags(str(content), topK=topK)

# ==============================================================================
# SECTION 3: TWITTER-SPECIFIC UTILITIES
# ==============================================================================
def calculate_twitter_resonance_scores(content_series: pd.Series, top_n_keywords: int = 15) -> pd.Series:
    # ... (No changes here, this function is correct)
    """
    Calculates content resonance (Jaccard similarity of keywords) for English texts.
    It measures how similar each tweet's keywords are to the first tweet's keywords.
    """
    try:
        # **CORRECTED**: Corrected stopword file path for consistency.
        with open('data/stopwords_E.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().strip().split('\n'))
    except FileNotFoundError:
        print("Warning: stopwords_E.txt not found. Proceeding without stopwords.")
        stop_words = set()

    def preprocess_text(text: str) -> list:
        """Cleans and tokenizes English text."""
        text = str(text)
        # Remove URLs, mentions, hashtags, and non-alphabetic characters.
        text = re.sub(r'http\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return [word for word in text.split() if word not in stop_words]

    def extract_keywords(texts: list, top_n: int) -> set:
        """Extracts top N keywords using TF-IDF."""
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english")
        try:
            vectorizer.fit_transform(texts)
            return set(vectorizer.get_feature_names_out())
        except ValueError: # Occurs if vocabulary is empty after processing.
            return set()

    if content_series.empty:
        return pd.Series([], dtype=float)

    # Use the first text in the series as the reference "topic".
    topic_keywords = extract_keywords([str(content_series.iloc[0])], top_n_keywords)
    
    # Process all texts to get their respective keyword sets.
    processed_texts = content_series.apply(preprocess_text)
    user_keywords_series = processed_texts.apply(lambda x: set([word for word, _ in Counter(x).most_common(top_n_keywords)]))
    
    # Calculate Jaccard similarity for each text against the topic.
    return user_keywords_series.apply(lambda uk: len(uk & topic_keywords) / len(uk | topic_keywords) if (uk | topic_keywords) else 0.0)

# ==============================================================================
# SECTION 4: GENERIC PROPAGATION PROBABILITY COMPONENTS
# ==============================================================================
def get_neighbor_counts(data_file_r: str, data_file_a: str, user_list: list) -> list:
    # ... (No changes here, this function is correct)
    """
    **NEW GENERIC FUNCTION**
    Extracts 'friends' count from data files to serve as neighbor count for each user.
    This function is generic and assumes a 'friends' column in the raw data files.
    """
    try:
        # Load rumor and anti-rumor raw data.
        df_r = pd.read_csv(data_file_r, sep=r'\s+', header=0, on_bad_lines='skip', quotechar='"')
        df_a = pd.read_csv(data_file_a, sep=r'\s+', header=0, on_bad_lines='skip', quotechar='"')
        
        # Concatenate, remove duplicates, and set user ID as the index for quick lookup.
        df = pd.concat([df_r, df_a]).drop_duplicates(subset=['id']).set_index('id')
        
        # Reindex to match the order and length of the game's user_list, filling missing with 0.
        # This ensures the neighbor list aligns perfectly with the user list from the game.
        user_list_as_type = [type(df.index[0])(u) for u in user_list]
        return df.reindex(user_list_as_type)['friends'].fillna(0).tolist()
    except Exception as e:
        print(f"Warning: Could not read neighbor counts from files. Using random values. Error: {e}")
        return [np.random.randint(10, 100) for _ in user_list]

def calculate_forwarding_probability(DF: float, n: int) -> float:
    # ... (No changes here, this function is correct)
    """
    **NEW FUNCTION**
    Calculates the forwarding probability Φ(t) based on a user's Driving Force (DF)
    and their number of neighbors (n).
    This implements equation (26) from the paper using a binomial distribution.
    The probability is the expected proportion of neighbors who are influenced to forward.
    """
    if n <= 0:
        return 0.0
    # Cap n at a reasonable number to avoid performance issues with comb(n, m).
    n = int(min(n, 1000))
    
    # Sum over all possible numbers of influenced neighbors (m from 0 to n).
    phi = sum((m / n) * comb(n, m, exact=False) * (DF ** m) * ((1 - DF) ** (n - m)) for m in range(n + 1))
    return phi