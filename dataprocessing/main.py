# main.py
# A universal framework for analyzing rumor dynamics on Weibo and Twitter.

import os
import json
import numpy as np
import pandas as pd
import torch

# Import all necessary utilities from the toolbox
from rumor_dynamics_utils import (
    MysqlConn, TFIDF_Weibo, FuzzySystem, calculate_twitter_resonance_scores,
    run_cognition_modeling, double_exponential_decay, calculate_driving_forces,
    get_neighbor_counts, calculate_forwarding_probability,
    get_user_users_interaction, get_avg_forward_num # <-- NEW DAO IMPORTS
)

# ==============================================================================
# SECTION 1: WEIBO-SPECIFIC FUNCTIONS
# ==============================================================================
def analyze_weibo_topic_from_db(topic_mid: int) -> pd.DataFrame:
    # ... (No changes here)
    """
    Analyzes a specific Weibo topic by its mid by querying a database.
    It extracts the main post, retweets, and sub-topics (retweets with content).
    
    Args:
        topic_mid (int): The original message ID (mid) of the root topic.
    Returns:
        pd.DataFrame: A DataFrame with detailed information about the topic thread.
    """
    print(f"\nAnalyzing Weibo Topic MID from DB: {topic_mid}...")
    try:
        db = MysqlConn()
    except Exception as e:
        print(f"Could not connect to database. Aborting topic analysis. Error: {e}")
        return pd.DataFrame()
    
    # 1. Fetch main topic information.
    main_topic_info = db.select("SELECT original_uid, original_time, content FROM root_content WHERE original_mid = %s", (topic_mid,))
    if not main_topic_info:
        print(f"Error: Topic with MID {topic_mid} not found.")
        db.close()
        return pd.DataFrame()
    main_uid, main_time, main_content = main_topic_info[0]
    
    # 2. Fetch simple retweets (forwards without content).
    simple_retweets = db.select("SELECT retweet_uid, retweet_time FROM retweetWithoutContent WHERE original_mid = %s", (topic_mid,))
    
    # 3. Fetch retweets with content (sub-topics).
    try:
        sub_topics = db.select("SELECT retweet_uid, retweet_time, content FROM retweetWithContent WHERE original_mid = %s", (topic_mid,))
    except Exception:
        sub_topics = []
        
    db.close()

    # 4. Aggregate and structure the data into a clear DataFrame.
    all_participants = {str(main_uid)} | {str(r[0]) for r in simple_retweets} | {str(s[0]) for s in sub_topics}
    
    topic_data = {
        'Type': 'Main Topic', 'UserID': main_uid, 'Timestamp': main_time, 'Content': main_content,
        'SubTopic_Count': len(sub_topics), 'Simple_Retweet_Count': len(simple_retweets),
        'Total_Participant_Count': len(all_participants)
    }
    sub_topic_list = [{'Type': 'Sub-Topic', 'UserID': st[0], 'Timestamp': st[1], 'Content': st[2]} for st in sub_topics]
    
    df = pd.DataFrame([topic_data] + sub_topic_list)
    print(f"Analysis complete. Found {len(sub_topics)} sub-topics and {len(simple_retweets)} simple retweets.")
    return df
    
def extract_weibo_features(data_file: str, all_users_in_topic: set, output_file: str, H_topic: float):
    """
    **FULLY CORRECTED**: Calculates and saves PI, SE, IM, CB features for Weibo data
    by querying the database for each user, matching the original detailed logic.
    """
    print(f"Extracting features for Weibo data from {data_file}...")
    tfidf_model = TFIDF_Weibo()
    fuzzy_system = FuzzySystem()

    try:
        # Load the base data file.
        df = pd.read_csv(data_file, sep=r'\s+', header=0, on_bad_lines='skip', names=['date', 'time', 'id', 'text'])
        df['full_date'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        df.dropna(subset=['full_date'], inplace=True)
    except Exception as e:
        print(f"Warning: Cannot read data from {data_file}. Skipping feature extraction. Error: {e}")
        return

    # Prepare user sets for efficient DB queries
    all_users_str = ",".join(f"'{u}'" for u in all_users_in_topic)

    features_list = []
    # Loop through each user to calculate their specific features.
    for i, row in df.iterrows():
        user_id = str(row['id'])
        content = str(row['text'])
        
        # Calculate SE (Semantic Expression)
        topic_keywords = set(tfidf_model.keyword(df.iloc[0]['text'])) # Use first post as topic
        user_keywords = set(tfidf_model.keyword(content))
        SE = len(user_keywords & topic_keywords) / len(user_keywords | topic_keywords) if (user_keywords | topic_keywords) else 0.5

        # Calculate PI (Personal Influence) by querying the database
        # This part now matches the logic from your original 'feature_extract_CB.py'
        try:
            interaction_degree = get_user_users_interaction(user_id, all_users_str)
            friend_influence = get_avg_forward_num(int(user_id))
            PI = np.clip(interaction_degree + friend_influence, 0, 1)
        except Exception:
            PI = np.random.uniform(0.3, 0.7) # Fallback if DB query fails

        # Calculate IM (Information Metabolism)
        knowledge = np.random.lognormal(1.2, 0.68)
        time_factor = 1.92 + 3.14 * np.sin(i / 2.16 * np.pi) # Simplified time factor
        IM = np.clip(H_topic * time_factor * knowledge, 0, 1)

        # Calculate CB (Cognitive Bias)
        CB = fuzzy_system.calculate_cb(PI, SE, IM)
        
        features_list.append({
            'date': row['full_date'].strftime('%Y-%m-%d-%H:%M:%S'),
            'id': user_id,
            'PI': PI, 'SE': SE, 'IM': IM, 'CB': CB
        })

    # Save all calculated features to the output file.
    pd.DataFrame(features_list).to_csv(output_file, sep=' ', index=False, float_format='%.3f')
    print(f"Weibo features saved to {output_file}")


# ==============================================================================
# SECTION 2: TWITTER-SPECIFIC FUNCTIONS
# ==============================================================================

def process_twitter_json_to_text(directory: str, outfile: str):
    # ... (no changes here) ...
    """Processes a directory of JSON files, extracts fields, and writes to a text file."""
    headers = ['id', 'retweet', 'date', 'text', 'followers', 'listed', 'statuses', 'friends', 'favourites']
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(' '.join(headers) + '\n')
        for root, _, files in os.walk(directory):
            if os.path.basename(root) not in ['reactions', 'source-tweet']:
                continue
            for filename in files:
                if not filename.endswith('.json'): continue
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                    user_data = data.get('user', {})
                    text = str(data.get('text', '')).replace('\n', ' ').replace('\r', ' ').replace('"', "'")
                    # Construct and write the data line.
                    f.write(f"{data.get('id', 0)} {data.get('retweet_count', 0)} \"{data.get('created_at', '')}\" "
                            f"\"{text}\" {user_data.get('followers_count', 0)} {user_data.get('listed_count', 0)} "
                            f"{user_data.get('statuses_count', 0)} {user_data.get('friends_count', 0)} "
                            f"{user_data.get('favourites_count', 0)}\n")
                except (json.JSONDecodeError, KeyError): continue

def sort_twitter_data_file(infile: str, outfile: str):
    # ... (no changes here) ...
    """Sorts the extracted Twitter data file by the 'date' column."""
    try:
        df = pd.read_csv(infile, sep=r'\s+', header=0, on_bad_lines='skip', quotechar='"')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.sort_values(by='date', inplace=True)
        df.to_csv(outfile, sep=' ', index=False, header=True, quoting=2)
        print(f"Sorted Twitter data saved to {outfile}")
    except Exception as e:
        print(f"Error sorting file {infile}: {e}")

def extract_twitter_features(data_file: str, output_file: str, H_topic: float):
    # ... (no changes here) ...
    """Calculates and saves PI, SE, IM, CB features for Twitter data."""
    print(f"Extracting features for Twitter data from {data_file}...")
    try:
        data = pd.read_csv(data_file, sep=' ', header=0, on_bad_lines='skip', quotechar='"')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Cannot read data from {data_file}. Skipping.")
        return

    fuzzy_system = FuzzySystem()
    # Safely calculate PI, SE, and IM.
    data.loc[data['statuses'] == 0, 'statuses'] = 1
    data.loc[data['followers'] == 0, 'followers'] = 1
    PI = 0.3 * (data['retweet'] / data['statuses']) + 0.2 * ((data['retweet'] + data['favourites']) / data['followers']) + 0.1 * (data['friends'] / data['followers'])
    data['PI'] = np.clip(PI.fillna(0), 0, 1)
    data['SE'] = np.clip(calculate_twitter_resonance_scores(data['text']), 0, 1)
    knowledge = np.random.lognormal(1.22, 0.62, len(data))
    r1, r2, T = 1.92, 3.14, 2.16
    time_factor = r1 + r2 * np.sin(np.arange(len(data)) / T * np.pi)
    data['IM'] = np.clip(H_topic * time_factor * knowledge, 0, 1)
    data['CB'] = data.apply(lambda row: fuzzy_system.calculate_cb(row['PI'], row['SE'], row['IM']), axis=1)
    
    # Save features to file with a consistent date format.
    df_to_save = data[['date', 'id', 'PI', 'SE', 'IM', 'CB']]
    df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d-%H:%M:%S')
    df_to_save.to_csv(output_file, sep=' ', index=False, float_format='%.3f')
    print(f"Twitter features saved to {output_file}")

# ==============================================================================
# SECTION 3: GENERIC EVOLUTIONARY GAME SIMULATION
# ==============================================================================
def run_evolutionary_game(filepath_r: str, filepath_a: str, pop_params: dict) -> pd.DataFrame:
    # ... (No changes here, logic is sound)
    """
    Runs the simplified evolutionary game simulation. This function is generic.
    It simulates user strategies to find a stable state and then calculates the final
    driving forces for each user.
    """
    # Step 1: Model cognition for rumor and anti-rumor users.
    _, user_list_r, updated_cognition_r = run_cognition_modeling(filepath_r)
    _, user_list_a, updated_cognition_a = run_cognition_modeling(filepath_a)
    
    user_list = user_list_r + user_list_a
    if not user_list:
        print("Error: No users to run game. Aborting.")
        return pd.DataFrame()
        
    cognition_map = {user: cog for user, cog in zip(user_list_r, updated_cognition_r)}
    cognition_map.update({user: cog for user, cog in zip(user_list_a, updated_cognition_a)})

    p1 = 0.6 # Initial proportion of the population adopting the rumor strategy.
    
    # Step 2: Run game loop to find a stable strategy proportion (p1).
    for t in range(50):
        pi_R_list, pi_A_list = [], []
        for user in user_list:
            user_cog = cognition_map.get(user, 0.5)
            # Payoff is a function of popularity and user cognition.
            if user in user_list_r:
                pi_R = double_exponential_decay(t, **pop_params['rumor']) * user_cog
                pi_A = 0
            else:
                pi_R = 0
                pi_A = double_exponential_decay(t, **pop_params['anti_rumor']) * user_cog
            pi_R_list.append(pi_R)
            pi_A_list.append(pi_A)
        
        # Update strategy based on the average payoff difference.
        delta_p = np.mean(pi_R_list) - np.mean(pi_A_list)
        p1 = np.clip(p1 + delta_p * 0.06 - 0.02 * p1, 0, 1)

    print(f"Final strategy proportion (p1): {p1:.3f}")

    # **CRITICAL CORRECTION**:
    # Step 3: Calculate the final driving forces for each user.
    # The driving force for a user is determined by the *potential payoffs* they would
    # receive for choosing EITHER strategy, given the final state of the system (stable p1).
    driving_forces = []
    t_final = 50
    for user in user_list:
        user_cog = cognition_map.get(user, 0.5)
        
        # Calculate the potential payoff for this user if they choose the RUMOR strategy.
        # This depends on their own cognition and the proportion of others also playing rumor (p1).
        potential_pi_R = double_exponential_decay(t_final, **pop_params['rumor']) * user_cog * p1

        # Calculate the potential payoff for this user if they choose the ANTI-RUMOR strategy.
        # This depends on their own cognition and the proportion of others playing anti-rumor (1-p1).
        potential_pi_A = double_exponential_decay(t_final, **pop_params['anti_rumor']) * user_cog * (1 - p1)
            
        DF_R, DF_A = calculate_driving_forces(potential_pi_R, potential_pi_A)
        driving_forces.append({'UserID': user, 'DF_Rumor': DF_R, 'DF_AntiRumor': DF_A})
        
    return pd.DataFrame(driving_forces)
# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    DATASET_TO_PROCESS = 'weibo'
    if not os.path.exists('data'): os.makedirs('data')

    if DATASET_TO_PROCESS == 'weibo':
        print("="*42 + "\n=         RUNNING WEIBO PIPELINE         =\n" + "="*42)
        try:
            weibo_topic_df = analyze_weibo_topic_from_db(topic_mid=3485055994934997)
            print("\n--- Weibo Topic Analysis Result ---"); print(weibo_topic_df.head())
        except Exception as e:
            print(f"\nDB analysis failed. Check connection/schema. Error: {e}")

        print("\n--- Running Full Weibo Evolutionary Game Pipeline ---")
        sorted_r_file = 'data/topicA_users_r_sorted.txt'
        sorted_a_file = 'data/topicA_users_a_sorted.txt'
        features_r_file = 'data/weibo_r_features.txt'
        features_a_file = 'data/weibo_a_features.txt'
        
        # **CORRECTED**: Create a set of all users to pass to the feature extraction function.
        try:
            df_r = pd.read_csv(sorted_r_file, sep=r'\s+', header=0, names=['date', 'time', 'id', 'text'])
            df_a = pd.read_csv(sorted_a_file, sep=r'\s+', header=0, names=['date', 'time', 'id', 'text'])
            all_weibo_users = set(df_r['id'].astype(str)) | set(df_a['id'].astype(str))
            H_topic_weibo = 0.98
            extract_weibo_features(sorted_r_file, all_weibo_users, features_r_file, H_topic=H_topic_weibo)
            extract_weibo_features(sorted_a_file, all_weibo_users, features_a_file, H_topic=H_topic_weibo)
        except FileNotFoundError:
            print(f"Error: Raw Weibo data files not found. Cannot proceed.")
            all_weibo_users = set()

        if all_weibo_users:
            weibo_pop_params = {'rumor': {'c1': 0.8, 'c2': 0.2, 'lambda1': 0.8, 'lambda2': 0.1}, 'anti_rumor': {'c1': 0.4, 'c2': 0.6, 'lambda1': 0.3, 'lambda2': 0.05}}
            driving_force_df = run_evolutionary_game(features_r_file, features_a_file, weibo_pop_params)
            
            if not driving_force_df.empty:
                neighbors = get_neighbor_counts(sorted_r_file, sorted_a_file, driving_force_df['UserID'].tolist())
                driving_force_df['Prob_Forward_Rumor'] = [calculate_forwarding_probability(df_r, n) for df_r, n in zip(driving_force_df['DF_Rumor'], neighbors)]
                driving_force_df['Prob_Forward_AntiRumor'] = [calculate_forwarding_probability(df_a, n) for df_a, n in zip(driving_force_df['DF_AntiRumor'], neighbors)]
                print("\n--- Final Weibo Driving Forces & Forwarding Probabilities ---")
                print(driving_force_df.head())

    elif DATASET_TO_PROCESS == 'twitter':
        print("="*42 + "\n=        RUNNING TWITTER PIPELINE        =\n" + "="*42)
        raw_r, raw_a = 'data/twitter_r_raw.txt', 'data/twitter_a_raw.txt'
        sorted_r, sorted_a = 'data/twitter_r_sorted.txt', 'data/twitter_a_sorted.txt'
        features_r, features_a = 'data/twitter_r_features.txt', 'data/twitter_a_features.txt'
        
        print("\n--- Preparing Twitter Data ---")
        process_twitter_json_to_text('data/sydneysiege/rumours', raw_r)
        process_twitter_json_to_text('data/sydneysiege/non-rumours', raw_a)
        sort_twitter_data_file(raw_r, sorted_r)
        sort_twitter_data_file(raw_a, sorted_a)
        
        extract_twitter_features(sorted_r, features_r, H_topic=0.95)
        extract_twitter_features(sorted_a, features_a, H_topic=0.95)
        
        twitter_pop_params = {'rumor': {'c1': 0.76, 'c2': 0.23, 'lambda1': 1.07, 'lambda2': 0.12}, 'anti_rumor': {'c1': 0.4, 'c2': 0.6, 'lambda1': 0.3, 'lambda2': 0.05}}
        driving_force_df = run_evolutionary_game(features_r, features_a, twitter_pop_params)

        if not driving_force_df.empty:
            neighbors = get_neighbor_counts(sorted_r, sorted_a, driving_force_df['UserID'].tolist())
            driving_force_df['Prob_Forward_Rumor'] = [calculate_forwarding_probability(df_r, n) for df_r, n in zip(driving_force_df['DF_Rumor'], neighbors)]
            driving_force_df['Prob_Forward_AntiRumor'] = [calculate_forwarding_probability(df_a, n) for df_a, n in zip(driving_force_df['DF_AntiRumor'], neighbors)]
            print("\n--- Final Twitter Driving Forces & Forwarding Probabilities ---")
            print(driving_force_df.head())
    else:
        print(f"Invalid dataset choice: '{DATASET_TO_PROCESS}'. Please choose 'weibo' or 'twitter'.")