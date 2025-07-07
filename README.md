# Rumor-diffusion-model
This project provides a comprehensive and modular framework **(STIR)** for analyzing the propagation dynamics of rumors and anti-rumors on social media platforms like Weibo and Twitter. It leverages a simplified evolutionary game model, without relying on complex graph structures, to simulate user behavior and predict the driving forces behind information spread.

# Update
- :up: 2025.7: Upload all codes.

# :bulb: Overview
The core objective of this framework is to quantify and predict user engagement with rumors. It achieves this through a multi-stage pipeline:
1. Data Processing: Extracts and standardizes raw data from different social media platforms (JSON files for Twitter, MySQL database for Weibo).
2. Feature Engineering: Calculates a set of cognitive and social features for each user, including:
    - PI (Projective Identification): A measure of a user's influence based on their network and activity.
    - SE (Selective Expose): Thematic relevance of a user's content to the core topic.
    - IM (Information Mastery): A score representing a user's susceptibility to new information, based on topic entropy and individual knowledge.
    - CB (Cognitive Bias): A composite score calculated using a Fuzzy Logic System that combines PI, SE, and IM.
3. Cognition Modeling: Employs Fuzzy C-Means clustering to model group-level and social-level cognition from individual CB scores. This allows for the simulation of social influence.
4. Evolutionary Game Simulation: A simplified game-theoretic model simulates the strategic choices of users (to spread a rumor or an anti-rumor). The simulation determines a stable strategy proportion within the user population.
5. Driving Force Calculation: The final outputs are derived from the game's results:
    - Driving Force : A value between 0 and 1, calculated from the difference in potential payoffs. It represents a user's propensity or inclination to spread a rumor vs. an anti-rumor.
    - Forwarding Probability : The ultimate prediction. It translates the Driving Force into a concrete probability that a user will forward a message to their neighbors, modeled using a binomial distribution as described in academic literature.

# :bulb: Dataset
1. Weibo Dataset
    - Source: Assumed to be stored in a MySQL database. The framework requires tables for root posts (root_content), retweets with content (retweetWithContent), and retweets without content (retweetWithoutContent).
    - Schema: The code expects specific column names (e.g., original_mid, retweet_uid, content, retweet_time). Please refer to the database query functions in rumor_dynamics_utils.py for the exact schema requirements.
    - Setup: You must have a running MySQL server and populate the database with your Weibo data. The database connection credentials need to be configured in rumor_dynamics_utils.py.
2. Twitter Dataset (PHEME-like)
    - Source: Assumed to be a collection of JSON files, structured similarly to the PHEME dataset for rumor analysis.
    - Structure: The framework expects a directory structure where each event is a folder containing rumours and non-rumours subdirectories. These, in turn, contain source-tweet and reactions folders with the raw tweet JSON objects.
    - Setup: Place your Twitter dataset directories inside the data/ folder. The main.py script is pre-configured to look for this structure.

# :bulb: Data Processing
## Weibo Data Processing
1. Topic Analysis: The process_weibo_topic function in main.py connects to the MySQL database to analyze a single topic thread identified by its topic_mid. It extracts:
    - The original post.
    - All simple retweets (forwards without comments).
    - All retweets with comments, which are treated as sub-topics.
    - It aggregates and presents this information in a structured DataFrame.
2. Feature Extraction: The extract_weibo_features function processes pre-extracted text files (which should be generated from the database). It uses the jieba library for Chinese text segmentation to calculate the SE (Semantic Expression) feature. Other features are calculated based on placeholder logic that can be customized.

## Twitter Data Processing
1. JSON to Text: The process_json_directory function in main.py recursively scans the directory structure, parses each JSON file, and extracts key fields (tweet text, user stats, creation date, etc.). It writes this standardized data into intermediate text files.
2. Sorting: The intermediate files are then sorted chronologically based on the tweet's creation time.
3. Feature Extraction: The extract_twitter_features function reads the sorted text files. It calculates:
    - PI from user profile statistics (followers, friends, statuses).
    - SE by measuring the Jaccard similarity between the keywords of a user's tweet and the original topic's tweet (a "resonance score"). It uses scikit-learn's TfidfVectorizer for keyword extraction.
    - Other features (IM, CB) are calculated using the common framework logic.

# :bulb: Models
The core of the framework relies on a sequence of models to derive the final driving force.
1. Fuzzy Logic System (FuzzySystem)
    - Purpose: To translate crisp input features (PI, SE, IM) into a single, nuanced Cognitive Bias (CB) score.
    - Mechanism: It uses a set of "IF-THEN" rules defined over fuzzy sets (e.g., "IF PI is HIGH and SE is MEDIUM THEN CB is HIGH"). The skfuzzy library is used to manage the variables, membership functions, and inference engine.
    - Output: A single, defuzzified CB value for each user, representing their overall cognitive state regarding the topic.
2. Cognition Modeling (run_cognition_modeling)
    - Purpose: To simulate social influence by modeling cognition at individual, group, and societal levels.
    - Mechanism:
        1. Clustering: It takes the CB scores of all users and applies the Fuzzy C-Means algorithm to identify a predefined number of cognitive "groups" or clusters.
        2. Group & Social Cognition: For each user, it identifies their primary group's cognitive center. It also calculates a single "social cognition" score, representing the weighted average cognition of all groups.
        3. Dynamic Update: It updates each user's initial CB score using a dynamic equation that accounts for self-decay, group influence, social influence, and random external information shocks (modeled with a skew-normal distribution).
    - Output: An updated_cognition score for each user.
3. Evolutionary Game Model (run_evolutionary_game)
    - Purpose: To simulate the strategic competition between rumor-spreading and anti-rumor-spreading behaviors in the population.
    - Mechanism:
        1. Payoff Calculation: The "payoff" for a user choosing a strategy is a function of their updated_cognition and the overall topic popularity (modeled by a double_exponential_decay function).
        2. Strategy Update: The model iteratively updates the proportion of the population adopting the rumor strategy (p1). This update is based on the average payoff difference between the two strategies in each time step.
        3. Stable State: After a set number of iterations, the game reaches a relatively stable p1 value.
    - Output: The final, stable p1 and the final Driving Force (DF_Rumor, DF_AntiRumor) for each user, which is calculated using the stable p1 and the final payoffs. This driving force represents the user's final propensity to act.

# :ledger: Reference 
## Dataset
**<u>Weibo</u>**. Social influence locality
for modeling retweeting behaviors. (Zhang et al., 2013) [[paper]](https://keg.cs.tsinghua.edu.cn/jietang/publications/IJCAI13-Zhang-et-al-social-influence-locality.pdf) [[Dataset]](https://www.aminer.cn/influencelocality)

**<u>Twitter</u>**. Analysing how people orient to and spread rumours in social media by looking at conversational threads. (Zubiaga  al., 2016) [[paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150989) [[Dataset]](https://figshare.com/articles/dataset/PHEME_rumour_scheme_dataset_journalism_use_case/2068650/2)

## Other SIR based models

- **<u>SIQRS</u>**. The SIQRS propagation model with quarantine on simplicial complexes. (Chen et al., 2024)  <u>IEEE TCSS</u> [[paper]](https://ieeexplore.ieee.org/abstract/document/10418977/)

- **<u>SPNC</u>**. Optimal control for positive and negative information diffusion based on game theory in online social networks. (wan et al., 2022) <u>IEEE TNSE</u> [[paper]](https://ieeexplore.ieee.org/abstract/document/9915429/)

- **<u>STIR</u>**. An Information Dissemination Model Based on the Rumor and Antirumor and Cognitive Game. (Mou et al., 2025) <u>IEEE TCSS</u> [[paper]](https://ieeexplore.ieee.org/abstract/document/10843969/)

- **<u>SHIR</u>**. An Information Dissemination Model Based on the Rumor and Anti-Rumor and Stimulate-Rumor and Tripartite Cognitive Game. (Li et al., 2023) <u>IEEE TCDS</u> [[paper]](https://ieeexplore.ieee.org/abstract/document/9839301/)

- **<u>SEIR-UD</u>**. A SEIR-UD model with reinforcement effect for the interaction between rumors and rumor-induced behaviors. (Zhao et al., 2024) <u>Nonlinear Dynamics</u> [[paper]](https://link.springer.com/article/10.1007/s11071-024-09613-9)

- **<u>RAPR-DG</u>**. Information Propagation Dynamic Model Based on Rumors, Antirumors, Prom-Rumors, and the Dynamic Game. (Li et al., 2024) <u>IEEE TCSS</u> [[paper]](https://ieeexplore.ieee.org/abstract/document/10713453/)

- **<u>HRD-DG</u>**. Hybrid Rumor Debunking in Online Social Networks: A Differential Game Approach. (Gan et al., 2025) <u>IEEE TSMC</u> [[paper]](https://ieeexplore.ieee.org/abstract/document/10849987/)

- **<u>MD-SHC</u>**. Model-Based and Data-Driven Stochastic Hybrid Control for Rumor Propagation in Dual-Layer Network. (Zhong et al., 2024) <u>IEEE TCSS</u> [[paper]](https://ieeexplore.ieee.org/abstract/document/10746235/)




