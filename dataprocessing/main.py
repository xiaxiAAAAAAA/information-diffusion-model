# main.py
# A universal framework for analyzing rumor dynamics on Weibo and Twitter.

import os
import json
import time
import re
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Import all necessary utilities from the toolbox
from rumor_dynamics_utils import (
    MysqlConn, TFIDF_Weibo, FuzzySystem, calculate_twitter_resonance_scores,
    run_cognition_modeling, double_exponential_decay, calculate_driving_forces
)

# ==============================================================================
# SECTION 1: WEIBO DATA PROCESSING & ANALYSIS
# ==============================================================================

def process_weibo_topic(topic_mid: int) -> pd.DataFrame:
    """
    Analyzes a specific Weibo topic by its mid.
    Extracts the main post, retweets without content (simple forwards), and
    retweets with content (sub-topics).
    
    Args:
        topic_mid (int): The original message ID (mid) of the root topic.
        
    Returns:
        pd.DataFrame: A DataFrame containing detailed information about the topic
                      and its sub-topics, or an empty DataFrame if the topic is not found.
    """
    print(f"\nAnalyzing Weibo Topic MID: {topic_mid}...")
    db = MysqlConn()
    
    # 1. Fetch main topic information
    main_topic_info = db.select("SELECT original_uid, original_time, content FROM root_content WHERE original_mid = %s", (topic_mid,))
    if not main_topic_info:
        print(f"Error: Topic with MID {topic_mid} not found in 'root_content' table.")
        db.close()
        return pd.DataFrame()
    
    main_uid, main_time, main_content = main_topic_info[0]
    
    # 2. Fetch simple retweets (without content)
    simple_retweets = db.select("SELECT retweet_uid, retweet_time FROM retweetWithoutContent WHERE original_mid = %s", (topic_mid,))
    
    # 3. Fetch retweets with content (sub-topics)
    # NOTE: Assuming a table 'retweetWithContent' exists with a similar structure.
    # If the table name is different, change it here.
    try:
        sub_topics = db.select("SELECT retweet_uid, retweet_time, content FROM retweetWithContent WHERE original_mid = %s", (topic_mid,))
    except Exception as e:
        print(f"Warning: Could not query 'retweetWithContent'. Assuming no sub-topics. Error: {e}")
        sub_topics = []
        
    db.close()

    # 4. Aggregate and structure the data
    all_participants = {str(main_uid)} | {str(r[0]) for r in simple_retweets} | {str(s[0]) for s in sub_topics}
    
    topic_data = {
        'Type': 'Main Topic',
        'UserID': main_uid,
        'Timestamp': main_time,
        'Content': main_content,
        'SubTopic_Count': len(sub_topics),
        'Simple_Retweet_Count': len(simple_retweets),
        'Total_Participant_Count': len(all_participants)
    }
    
    sub_topic_list = [{
        'Type': 'Sub-Topic',
        'UserID': st[0],
        'Timestamp': st[1],
        'Content': st[2],
        'SubTopic_Count': np.nan,
        'Simple_Retweet_Count': np.nan,
        'Total_Participant_Count': np.nan
    } for st in sub_topics]
    
    # Create a DataFrame for clear output
    df = pd.DataFrame([topic_data] + sub_topic_list)
    print(f"Analysis complete. Found {len(sub_topics)} sub-topics and {len(simple_retweets)} simple retweets.")
    return df

def extract_weibo_features(data_file: str, output_file: str, is_rumor: bool, H_topic: float):
    """Calculates and saves PI, SE, IM, CB features for Weibo data."""
    print(f"Extracting features for {'rumor' if is_rumor else 'anti-rumor'} Weibo data...")
    tfidf_model = TFIDF_Weibo()
    fuzzy_system = FuzzySystem()
    
    try:
        df = pd.read_csv(data_file, sep=' ', header=0, on_bad_lines='skip', quotechar='"')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Cannot read data from {data_file}, skipping feature extraction.")
        return
        
    # Implement PI, SE, IM, CB calculation logic here based on your model for Weibo
    # This is a placeholder logic, as the original files had complex DB lookups
    # which are simplified here for a file-based workflow.
    df['PI'] = np.random.rand(len(df)) # Placeholder
    df['SE'] = df['text'].astype(str).apply(lambda x: len(tfidf_model.keyword(x, topK=5)) / 5.0)
    df['IM'] = np.clip(H_topic * np.random.lognormal(1.2, 0.68, len(df)), 0, 1)
    df['CB'] = df.apply(lambda row: fuzzy_system.calculate_cb(row['PI'], row['SE'], row['IM']), axis=1)

    # Save features to file
    df_to_save = df[['date', 'id', 'PI', 'SE', 'IM', 'CB']]
    df_to_save.to_csv(output_file, sep=' ', index=False, float_format='%.3f')
    print(f"Weibo features saved to {output_file}")


# ==============================================================================
# SECTION 2: TWITTER DATA PROCESSING
# ==============================================================================

def extract_twitter_features(data_file: str, output_file: str, is_rumor: bool, H_topic: float):
    """Calculates and saves PI, SE, IM, CB features for Twitter data."""
    print(f"Extracting features for {'rumor' if is_rumor else 'anti-rumor'} Twitter data...")
    try:
        data = pd.read_csv(data_file, sep=' ', header=0, on_bad_lines='skip', quotechar='"')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Cannot read data from {data_file}, skipping.")
        return

    fuzzy_system = FuzzySystem()

    # PI calculation
    data.loc[data['statuses'] == 0, 'statuses'] = 1
    data.loc[data['followers'] == 0, 'followers'] = 1
    PI = 0.3 * (data['retweet'] / data['statuses']) + \
         0.2 * ((data['retweet'] + data['favourites']) / data['followers']) + \
         0.1 * (data['friends'] / data['followers'])
    PI = np.clip(PI.fillna(0), 0, 1)

    # SE calculation
    SE = calculate_twitter_resonance_scores(data['text'])
    SE = np.clip(SE, 0, 1)

    # IM calculation
    knowledge = np.random.lognormal(1.22, 0.62, len(data))
    r1, r2, T = 1.92, 3.14, 2.16
    time_factor = r1 + r2 * np.sin(np.arange(len(data)) / T * np.pi)
    IM = np.clip(H_topic * time_factor * knowledge, 0, 1)

    # CB calculation
    data['PI'] = PI
    data['SE'] = SE
    data['IM'] = IM
    data['CB'] = data.apply(lambda row: fuzzy_system.calculate_cb(row['PI'], row['SE'], row['IM']), axis=1)
    
    # Save features to file
    df_to_save = data[['date', 'id', 'PI', 'SE', 'IM', 'CB']]
    # Ensure date format is consistent
    df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d-%H:%M:%S')
    df_to_save.to_csv(output_file, sep=' ', index=False, float_format='%.3f')
    print(f"Twitter features saved to {output_file}")


# ==============================================================================
# SECTION 3: GENERIC EVOLUTIONARY GAME SIMULATION
# ==============================================================================

def run_evolutionary_game(filepath_r: str, filepath_a: str, pop_params: dict) -> pd.DataFrame:
    """
    Runs the simplified evolutionary game simulation. This function is generic.
    
    Args:
        filepath_r (str): Path to the rumor features file.
        filepath_a (str): Path to the anti-rumor features file.
        pop_params (dict): Dictionary with popularity decay parameters.
        
    Returns:
        pd.DataFrame: A DataFrame with User IDs and their final Driving Forces.
    """
    _, user_list_r, updated_cognition_r = run_cognition_modeling(filepath_r)
    _, user_list_a, updated_cognition_a = run_cognition_modeling(filepath_a)
    
    user_list = user_list_r + user_list_a
    if not user_list:
        print("Error: No users to run game. Aborting.")
        return pd.DataFrame()
        
    cognition_map = {user: cog for user, cog in zip(user_list_r, updated_cognition_r)}
    cognition_map.update({user: cog for user, cog in zip(user_list_a, updated_cognition_a)})

    p1 = 0.6 # Initial strategy proportion
    
    # Game loop to find a stable strategy proportion p1
    for t in range(50):
        pi_R_list, pi_A_list = [], []
        for user in user_list:
            user_cog = cognition_map.get(user, 0.5)
            if user in user_list_r:
                pi_R = double_exponential_decay(t, **pop_params['rumor']) * user_cog
                pi_A = 0
            else:
                pi_R = 0
                pi_A = double_exponential_decay(t, **pop_params['anti_rumor']) * user_cog
            pi_R_list.append(pi_R)
            pi_A_list.append(pi_A)
        
        # Update strategy based on average payoffs
        delta_p = np.mean(pi_R_list) - np.mean(pi_A_list)
        p1 = np.clip(p1 + delta_p * 0.06 - 0.02 * p1, 0, 1)

    print(f"Final strategy proportion (p1): {p1:.3f}")

    # Calculate final driving forces for each user using the stable p1
    driving_forces = []
    t_final = 50
    for user in user_list:
        user_cog = cognition_map.get(user, 0.5)
        if user in user_list_r:
            pi_R = double_exponential_decay(t_final, **pop_params['rumor']) * user_cog * p1
            pi_A = double_exponential_decay(t_final, **pop_params['anti_rumor']) * (1 - user_cog) * (1 - p1)
        else:
            pi_R = double_exponential_decay(t_final, **pop_params['rumor']) * (1 - user_cog) * p1
            pi_A = double_exponential_decay(t_final, **pop_params['anti_rumor']) * user_cog * (1 - p1)
            
        DF_R, DF_A = calculate_driving_forces(pi_R, pi_A)
        driving_forces.append({'UserID': user, 'DF_Rumor': DF_R, 'DF_AntiRumor': DF_A})
        
    return pd.DataFrame(driving_forces)

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- Framework Configuration ---
    # Choose which dataset to process: 'weibo' or 'twitter'
    DATASET_TO_PROCESS = 'weibo' # <-- CHANGE THIS TO 'twitter' TO RUN THE OTHER PIPELINE

    if not os.path.exists('data'):
        os.makedirs('data')

    if DATASET_TO_PROCESS == 'weibo':
        print("==========================================")
        print("=         RUNNING WEIBO PIPELINE         =")
        print("==========================================")
        
        # --- Step 1: Analyze a specific Weibo topic and its sub-topics ---
        # NOTE: This requires a populated database.
        # Replace '3485055994934997' with the topic MID you want to analyze.
        try:
            weibo_topic_df = process_weibo_topic(topic_mid=3485055994934997)
            print("\n--- Weibo Topic Analysis Result ---")
            print(weibo_topic_df)
        except Exception as e:
            print(f"\nCould not run Weibo topic analysis. Please check DB connection and table names. Error: {e}")

        # --- Step 2: Run the full evolutionary game pipeline for a pre-extracted topic ---
        # NOTE: This part assumes you have `topicA_users_r.txt` and `topicA_users_a.txt` from previous work.
        print("\n--- Running Full Weibo Evolutionary Game Pipeline ---")
        features_r_file = 'data/weibo_r_features.txt'
        features_a_file = 'data/weibo_a_features.txt'

        # This assumes raw data files are present. A real implementation would generate these from the DB.
        extract_weibo_features('data/topicA_users_r_sorted.txt', features_r_file, is_rumor=True, H_topic=0.98)
        extract_weibo_features('data/topicA_users_a_sorted.txt', features_a_file, is_rumor=False, H_topic=0.98)
        
        weibo_pop_params = {
            'rumor': {'c1': 0.8, 'c2': 0.2, 'lambda1': 0.8, 'lambda2': 0.1},
            'anti_rumor': {'c1': 0.4, 'c2': 0.6, 'lambda1': 0.3, 'lambda2': 0.05}
        }
        
        driving_force_df = run_evolutionary_game(features_r_file, features_a_file, weibo_pop_params)
        print("\n--- Final Weibo Driving Forces ---")
        print(driving_force_df.head())

    elif DATASET_TO_PROCESS == 'twitter':
        print("==========================================")
        print("=        RUNNING TWITTER PIPELINE        =")
        print("==========================================")
        
        # Define file paths
        raw_r_file = 'data/twitter_r_raw.txt'
        raw_a_file = 'data/twitter_a_raw.txt'
        sorted_r_file = 'data/twitter_r_sorted.txt'
        sorted_a_file = 'data/twitter_a_sorted.txt'
        features_r_file = 'data/twitter_r_features.txt'
        features_a_file = 'data/twitter_a_features.txt'
        
        # --- Step 1: Data Preparation ---
        # This assumes JSON data is in the specified directories.
        # process_json_directory('data/sydneysiege/rumours', raw_r_file, headers)
        # process_json_directory('data/sydneysiege/non-rumours', raw_a_file, headers)
        # sort_data_file(raw_r_file, sorted_r_file)
        # sort_data_file(raw_a_file, sorted_a_file)
        
        # --- Step 2: Feature Extraction ---
        # This logic should be adapted to the specifics of your Twitter data files
        extract_twitter_features(sorted_r_file, features_r_file, is_rumor=True, H_topic=0.95)
        extract_twitter_features(sorted_a_file, features_a_file, is_rumor=False, H_topic=0.95)
        
        # --- Step 3: Run Evolutionary Game ---
        twitter_pop_params = {
            'rumor': {'c1': 0.76, 'c2': 0.23, 'lambda1': 1.07, 'lambda2': 0.12},
            'anti_rumor': {'c1': 0.4, 'c2': 0.6, 'lambda1': 0.3, 'lambda2': 0.05}
        }
        
        driving_force_df = run_evolutionary_game(features_r_file, features_a_file, twitter_pop_params)
        print("\n--- Final Twitter Driving Forces ---")
        print(driving_force_df.head())

    else:
        print(f"Invalid dataset choice: '{DATASET_TO_PROCESS}'. Please choose 'weibo' or 'twitter'.")