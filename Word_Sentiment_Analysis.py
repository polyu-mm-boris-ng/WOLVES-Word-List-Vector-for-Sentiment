import os
import pickle
import itertools
import gensim
import nltk
import pandas as pd
import os
import numpy as np
import pickle
import string
import joblib
import re, string
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
from nltk.cluster import KMeansClusterer
from collections import OrderedDict, Counter
from nltk.corpus import words
from collections import Counter, defaultdict
from scipy.stats import skew, kurtosis
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from sklearn import cluster
from sklearn import metrics
from nltk.corpus import stopwords
from scipy.spatial import procrustes
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

wordnet_lemmatizer = WordNetLemmatizer()

# load pre-defined word list
stop_word_list = stopwords.words('english')
data_prefix = '/media/vol/FaceRecContest/10-X/word_vector_similarity/word_extraction/'
with open('resources/MC_positive', 'r') as f:
    positive_word_list = f.readlines()
    positive_word_list = [d.strip().lower() for d in positive_word_list]
    positive_word_dict = {d: 1 for d in positive_word_list}
with open('resources/MC_negative', 'r') as f:
    negative_word_list = f.readlines()
    negative_word_list = [d.strip().lower() for d in negative_word_list]
    negative_word_dict = {d: 1 for d in negative_word_list}

with open('resources/H4_positive', 'r') as f:
    h4_positive_word_list = f.readlines()
    h4_positive_word_list = [d.strip().lower() for d in h4_positive_word_list]
with open('resources/H4_negative', 'r') as f:
    h4_negative_word_list = f.readlines()
    h4_negative_word_list = [d.strip().lower() for d in h4_negative_word_list]

LM_all_word_list = positive_word_list + negative_word_list
H4_all_word_list = h4_positive_word_list + h4_negative_word_list
excluded_overlap_words = [d for d in positive_word_list if d in h4_negative_word_list] + [d for d in negative_word_list if d in h4_positive_word_list]
#Overlapping vocabulary lists
LM_H4_overlap_words = [d for d in LM_all_word_list if (d in H4_all_word_list) and (d not in excluded_overlap_words)]

H4_unique_pos_words = [d for d in h4_positive_word_list if (d not in LM_all_word_list)]
H4_unique_neg_words = [d for d in h4_negative_word_list if (d not in LM_all_word_list)]

# load all common words in the context of financial text
master_dictionary_pd = pd.read_csv('resources/LoughranMcDonald_MasterDictionary_2020.csv')
all_common_word_list = list(master_dictionary_pd['Word'].dropna().str.lower().values)
all_common_word_dict = {d:1 for d in all_common_word_list}
all_common_word_set = set(all_common_word_list)


# Specify a Word2Vec model and two world lists
# Compute the average distance between two groups of words
def distance_between_groups(model, word_list1, word_list2):
    embedding1 = model.wv[word_list1]
    embedding2 = model.wv[word_list2]
    norm1 = np.sqrt((embedding1*embedding1).sum(axis=1)).reshape(len(word_list1),-1)
    norm2 = np.sqrt((embedding2*embedding2).sum(axis=1)).reshape(len(word_list2),-1)
    norm_matrix = np.dot(norm1, np.transpose(norm2))
    dis_matrix = np.dot(embedding1, np.transpose(embedding2))/norm_matrix
    # Average distance between each word in word_list1 and all words in word_list2
    word_list1_distance_mean = (dis_matrix.sum(axis=1)-1) / (len(word_list2)-1)
    # Average distance between each word in word_list2 and all words in word_list1
    word_list2_distance_mean = (dis_matrix.sum(axis=0)-1) / (len(word_list1)-1)
    return word_list1_distance_mean, word_list2_distance_mean


# Detect the dynamic shift of one word
def distance_dynamic(embeddings, method='all_mean'):
    norm1 = np.sqrt((embeddings*embeddings).sum(axis=1)).reshape(len(embeddings),-1)
    norm_matrix = np.dot(norm1, np.transpose(norm1))
    dis_matrix = np.dot(embeddings, np.transpose(embeddings))/norm_matrix
    # Calculate similarity based on the selected method
    if method == 'all_mean':
        similarity_list = [(d.sum()-1)/(len(d)-1) for d in dis_matrix]
    elif method == 'last':
        similarity_list = [d[-1] for d in dis_matrix]
    elif method == 'previous':
        similarity_list = []
        for i in range(1998, 2019):
            pos = i-1997
            similarity_list.append(dis_matrix[pos][pos-1])
        similarity_list = [similarity_list[0]] + similarity_list
    return similarity_list


# Get top/bottom ranked words
def get_top_down_words(pd_data, year_cols, num=50):
    top_data_dict = {}
    down_data_dict = {}
    # Rank based on the value in each columns
    for tmp_year in year_cols + ['mean', 'std', 'maxmin']:
        pd_data = pd_data.sort_values(by=[tmp_year])
        # Select the top and bottom 'num' words
        top_words = pd_data.iloc[:num].index
        down_words = pd_data.iloc[-1*num:].index
        top_data_dict[tmp_year] = top_words
        down_data_dict[tmp_year] = down_words
    min_pd = pd.DataFrame(top_data_dict)
    max_pd = pd.DataFrame(down_data_dict)
    return min_pd, max_pd


# Use the n_similarity method of the built-in Word2Vec model to calculate the semantic distance between two sets of words.
def distance_between_groups_builtin(model, word_list1, word_list2, model2=None):
    distance_list = []
    for tmp_word in word_list1:
        tmp_distance = model.n_similarity([tmp_word], word_list2)
        distance_list.append(tmp_distance)
    distance_list = np.array(distance_list)
    return distance_list, distance_list


# Compute the cosine similarity between any two vectors
def cosine_similarity(v1, v2):
    norm1 = np.sqrt((v1*v1).sum())
    norm2 = np.sqrt((v2*v2).sum())
    dis = (v1*v2).sum()
    return dis/(norm2*norm1)


# Compute the Euclidean distance between any two vectors
def euc_distance(v1, v2, norm=True):
    norm1 = np.sqrt((v1*v1).sum())
    norm2 = np.sqrt((v2*v2).sum())
    if norm:
        dis = np.sqrt(((v1/norm1 - v2/norm2)*((v1/norm1 - v2/norm2))).sum())
    else:
        dis = np.sqrt(((v1 - v2)*((v1 - v2))).sum())
    return dis


# Calculate the distance between each word in the world list and the benchmark vector
def distance_between_benchmark_vector(model, word_list1, benchmark_vector, similarity='cos'):
    distance_list = []
    for tmp_word in word_list1:
        tmp_vector = model.get_vector(tmp_word) #Get word vectors for a given word from the word vector model
        if similarity == 'cos':       #Calculate using Cosine similarity
            tmp_distance = cosine_similarity(tmp_vector, benchmark_vector)
        else:                         #Calculate using Euclidean distance
            tmp_distance = euc_distance(tmp_vector, benchmark_vector)
        distance_list.append(tmp_distance)
    distance_list = np.array(distance_list)
    return distance_list, distance_list


# Calculate the embedding space distance between two group of words
def embedding_distance_between_groups(word_model, wordlist1, wordlist2, minus_one=False):
    embedding1 = word_model.wv[wordlist1]
    embedding2 = word_model.wv[wordlist2]
    norm1 = np.sqrt((embedding1*embedding1).sum(axis=1))
    norm2 = np.sqrt((embedding2*embedding2).sum(axis=1))
    embedding1 = embedding1/(norm1.reshape(-1,1))
    embedding2 = embedding2/(norm2.reshape(-1,1))
    similarity = np.dot(embedding1, embedding2.T).sum(axis=1)
    if minus_one:
        similarity = (similarity-1) / (len(wordlist1)-1)
    else:
        similarity = similarity/len(wordlist1)
    return similarity


# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
word_dictionary = {}
for d in words.words():
    word_dictionary[d] = 1
# Check the validity of words
def check_valid_word(tmp_token_words):
    # Lemmatize the token to verb, adjective, and noun forms
    v_tmp_token_words = wordnet_lemmatizer.lemmatize(tmp_token_words, 'v')
    a_tmp_token_words = wordnet_lemmatizer.lemmatize(tmp_token_words, 'a')
    n_tmp_token_words = wordnet_lemmatizer.lemmatize(tmp_token_words, 'n')
    # Check if any of the lemmatized forms or the original token is in the word dictionary
    if (v_tmp_token_words not in word_dictionary) and (a_tmp_token_words not in word_dictionary) and (n_tmp_token_words not in word_dictionary) and (tmp_token_words not in word_dictionary):
        return False
    return True


#Align the word vectors of the overlapping words in the two models into a common space through Procrustes analysis
def align_by_procrustes(base_model, align_model, overlap_words):
    # Get word embeddings for overlapping words
    base_embedding = base_model.wv[overlap_words]
    align_embedding = align_model.wv[overlap_words]

    # Normalize word embeddings
    norm1 = np.sqrt((base_embedding * base_embedding).sum(axis=1))
    norm2 = np.sqrt((align_embedding * align_embedding).sum(axis=1))
    base_embedding = base_embedding / (norm1.reshape(-1, 1))
    align_embedding = align_embedding / (norm2.reshape(-1, 1))

    # Calculate the rotation matrix using Procrustes analysis
    m = align_embedding.T.dot(base_embedding) # m-matrix: linear relationship between two word vector spaces
    u, _, v = np.linalg.svd(m) # Decompose the matrix m by singular value decomposition (SVD) to get two orthogonal matrices u and v
    ortho = u.dot(v)  # The product of u and v is the orthogonal matrix required for alignment
    # Adjust all word embeddings of align_model
    all_align_embeddings = align_model.wv.vectors
    all_align_embeddings = (all_align_embeddings).dot(ortho)
    return all_align_embeddings


# Load all models
all_report_wv_dict = {}
all_news_wv_dict = {}
all_cc_wv_dict = {}

# Load default model for testing
# word embedding models for annual report
model_prefix = 'model/'
report_wv_dict = {}

# Load Word2Vec models for reports for each year from 1997 to 2018
for i in range(1997, 2019):
    model = gensim.models.Word2Vec.load(str(Path(model_prefix, "{}_ini_w2v.mod".format(i))))
    report_wv_dict[i] = model
# Find the intersection of keys (overlapping words) across all Word2Vec models for reports
final_10k_set = set(report_wv_dict[1997].wv.key_to_index.keys())
for i in range(1997, 2019):
    final_10k_set = final_10k_set.intersection(set(report_wv_dict[i].wv.key_to_index.keys()))


# word embedding models for news (From 1997 to 2018)
model_prefix = 'model/'
news_wv_dict = {}
# Load Word2Vec models for news for each year from 1997 to 2018
for i in range(1997, 2019):
    model = gensim.models.Word2Vec.load(str(Path(model_prefix, "news_{}_ini_w2v.mod".format(i))))
    news_wv_dict[i] = model
# Find the intersection of keys (overlapping words) across all Word2Vec models for news
final_news_set = set(news_wv_dict[1997].wv.key_to_index.keys())
for i in range(1997, 2019):
    final_news_set = final_news_set.intersection(set(news_wv_dict[i].wv.key_to_index.keys()))


# Load Word2Vec models for conference call for each year from 2003 to 2018
model_prefix = 'model/'
cc_wv_dict = {}
for i in range(2003, 2019):
    model = gensim.models.Word2Vec.load(str(Path(model_prefix, "conference_{}_ini_w2v.mod".format(i))))
    cc_wv_dict[i] = model
# Find the intersection of keys (overlapping words) across all Word2Vec models for conference call
final_cc_set = set(cc_wv_dict[2003].wv.key_to_index.keys())
for i in range(2003, 2019):
    final_cc_set = final_cc_set.intersection(set(cc_wv_dict[i].wv.key_to_index.keys()))


all_report_wv_dict['ini'] = report_wv_dict
all_news_wv_dict['ini'] = news_wv_dict
all_cc_wv_dict['ini'] = cc_wv_dict

'''
# L2-Normalize word embeddings and apply procrustes to align the vectors
for tmp_year in report_wv_dict:
    report_wv_dict[tmp_year].wv.init_sims(replace=True)
for tmp_year in cc_wv_dict:
    cc_wv_dict[tmp_year].wv.init_sims(replace=True)
for tmp_year in news_wv_dict:
    news_wv_dict[tmp_year].wv.init_sims(replace=True)
# align the vectors to the last model
for tmp_model_type in all_report_wv_dict:
    for year in range(1997,2018):
        align_embedding = align_by_procrustes(all_report_wv_dict[tmp_model_type][2018], all_report_wv_dict[tmp_model_type][year], final_10k_set.intersection(all_common_word_set))
        all_report_wv_dict[tmp_model_type][year].wv.vectors = align_embedding
for tmp_model_type in all_news_wv_dict:
    for year in range(1997,2018):
        align_embedding = align_by_procrustes(all_news_wv_dict[tmp_model_type][2018], all_news_wv_dict[tmp_model_type][year], final_news_set.intersection(all_common_word_set))
        all_news_wv_dict[tmp_model_type][year].wv.vectors = align_embedding
for tmp_model_type in all_cc_wv_dict:
    for year in range(2003,2018):
        align_embedding = align_by_procrustes(all_cc_wv_dict[tmp_model_type][2018], all_cc_wv_dict[tmp_model_type][year], final_cc_set.intersection(all_common_word_set))
        all_cc_wv_dict[tmp_model_type][year].wv.vectors = align_embedding
'''

# Load data about term frequency and document frequency
with open('resources/annual_report_df.pk', "rb") as f:
    annual_report_counter = pickle.load(f)
with open('resources/news_df.pk', "rb") as f:
    news_txt_counter = pickle.load(f)
with open('resources/cc_df.pk', "rb") as f:
    cc_txt_counter = pickle.load(f)


# get the df tf of words in reports
year_cols = list(range(1997,2019))
words = list(final_10k_set) # List of words common across all Word2Vec models for reports

# Initialize dictionaries to store document frequencies and term frequencies
document_df_dict = {d:[] for d in year_cols}
document_df_dict['words'] = words
document_tf_dict = {d:[] for d in year_cols}
document_tf_dict['words'] = words

for tmp_word in words:
    for tmp_year in year_cols:
        normalized_df = annual_report_counter[tmp_year][tmp_word]
        normalized_tf = annual_report_counter[tmp_year][tmp_word]
        document_df_dict[tmp_year].append(normalized_df)
        document_tf_dict[tmp_year].append(normalized_tf)
# Create pandas DataFrames for document frequencies and term frequencies
report_df_pd = pd.DataFrame(document_df_dict).set_index(['words'])
report_tf_pd = pd.DataFrame(document_tf_dict).set_index(['words'])

# get the df tf of words in news
words = list(final_news_set)  # List of words common across all Word2Vec models for news
document_df_dict = {d:[] for d in year_cols}
document_df_dict['words'] = words
document_tf_dict = {d:[] for d in year_cols}
document_tf_dict['words'] = words
for tmp_word in words:
    for tmp_year in year_cols:
        normalized_df = news_txt_counter[tmp_year][tmp_word]
        normalized_tf = news_txt_counter[tmp_year][tmp_word]
        document_df_dict[tmp_year].append(normalized_df)
        document_tf_dict[tmp_year].append(normalized_tf)
news_df_pd = pd.DataFrame(document_df_dict).set_index(['words'])
news_tf_pd = pd.DataFrame(document_tf_dict).set_index(['words'])

# get the df tf of words in cc
words = list(final_cc_set) # List of words common across all Word2Vec models for cc
year_cols = list(range(2003,2019))
document_df_dict = {d:[] for d in year_cols}
document_df_dict['words'] = words
document_tf_dict = {d:[] for d in year_cols}
document_tf_dict['words'] = words
for tmp_word in words:
    for tmp_year in year_cols:
        normalized_df = cc_txt_counter[tmp_year][tmp_word]
        normalized_tf = cc_txt_counter[tmp_year][tmp_word]
        document_df_dict[tmp_year].append(normalized_df)
        document_tf_dict[tmp_year].append(normalized_tf)
cc_df_pd = pd.DataFrame(document_df_dict).set_index(['words'])
cc_tf_pd = pd.DataFrame(document_tf_dict).set_index(['words'])


# number of valid sentiment words
report_valid_positive_word_list = [d for d in positive_word_list if d in final_10k_set]
report_valid_negative_word_list = [d for d in negative_word_list if d in final_10k_set]
print(len(positive_word_list), len(negative_word_list))
print(len(report_valid_positive_word_list), len(report_valid_negative_word_list))

news_valid_positive_word_list = [d for d in positive_word_list if d in final_news_set]
news_valid_negative_word_list = [d for d in negative_word_list if d in final_news_set]
print(len(positive_word_list), len(negative_word_list))
print(len(news_valid_positive_word_list), len(news_valid_negative_word_list))

cc_valid_positive_word_list = [d for d in positive_word_list if d in final_cc_set]
cc_valid_negative_word_list = [d for d in negative_word_list if d in final_cc_set]
print(len(positive_word_list), len(negative_word_list))
print(len(cc_valid_positive_word_list), len(cc_valid_negative_word_list))


# Prepare data for Stata analysis
def get_data_for_stata(target_pd, tf_pd, df_pd, year_cols, group_distance_data, benchmark_word_set):
    year_count = len(year_cols)
    data_dict = {'word': [], 'year': [], 'score': [], 'LM_H4_word': [], 'MC_word': [],
                 'average_tf': [], 'average_df': [], 'group_distance': [], 'benchmark_word': []}
    group_distance_data_list = list(group_distance_data.loc[year_cols].values)
    for d in target_pd.index:
        # data for target
        data_dict['word'] = data_dict['word'] + [d] * year_count
        data_dict['year'] = data_dict['year'] + year_cols
        data_dict['score'] = data_dict['score'] + list(target_pd.loc[d][year_cols])
        data_dict['group_distance'] = data_dict['group_distance'] + group_distance_data_list

        tf_data_list = list(tf_pd.loc[d][year_cols].values)
        df_data_list = list(df_pd.loc[d][year_cols].values)
        data_dict['average_tf'] = data_dict['average_tf'] + tf_data_list
        data_dict['average_df'] = data_dict['average_df'] + df_data_list

        # Check if the word is in the LM_H4_overlap_words list and set the corresponding flag
        if d in LM_H4_overlap_words:  # H4_all_word_list     previously used: LM_H4_overlap_words
            data_dict['LM_H4_word'] = data_dict['LM_H4_word'] + [1] * (year_count)
        else:
            data_dict['LM_H4_word'] = data_dict['LM_H4_word'] + [0] * (year_count)

        # Check if the word is in the benchmark_word_set and set the corresponding flag
        if d in benchmark_word_set:
            data_dict['benchmark_word'] = data_dict['benchmark_word'] + [1] * (year_count)
        else:
            data_dict['benchmark_word'] = data_dict['benchmark_word'] + [0] * (year_count)

        # Check if the word is in the positive_word_dict or negative_word_dict and set the corresponding flag
        if (d in positive_word_dict) or (d in negative_word_dict):
            data_dict['MC_word'] = data_dict['MC_word'] + [1] * (year_count)
        else:
            data_dict['MC_word'] = data_dict['MC_word'] + [0] * (year_count)
    data_pd = pd.DataFrame(data_dict)
    return data_pd


# Calculate the sentiment intensity of positive and negative words based on word embeddings
# Filter positive and negative word lists to include only words present in the word embedding model
def sentiment_word_intensity(model, pos_list, neg_list, similarity='cos'):
    pos_list = [d for d in pos_list if d in model.wv]
    neg_list = [d for d in neg_list if d in model.wv]
    pos_embedding = []
    neg_embedding = []
    for tmp_word in pos_list:
        if tmp_word not in model.wv: continue
        tmp_embedding = model.wv[tmp_word]
        pos_embedding.append(tmp_embedding)
    for tmp_word in neg_list:
        if tmp_word not in model.wv: continue
        tmp_embedding = model.wv[tmp_word]
        neg_embedding.append(tmp_embedding)

    # # Normalize positive and negative embeddings
    pos_embeddings = np.array(pos_embedding)
    pos_norm_value = np.sqrt((pos_embeddings*pos_embeddings).sum(axis=1))
    pos_embeddings = pos_embeddings / pos_norm_value.reshape(-1, 1)
    neg_embeddings = np.array(neg_embedding)
    neg_norm_value = np.sqrt((neg_embeddings*neg_embeddings).sum(axis=1))
    neg_embeddings = neg_embeddings / neg_norm_value.reshape(-1, 1)
    pos_embeddings = pos_embeddings.mean(axis=0)
    neg_embeddings = neg_embeddings.mean(axis=0)

    pos_word_dict = {}
    neg_word_dict = {}
    # Calculate sentiment scores for each word in the positive list
    for tmp_word in pos_list:
        tmp_vector = model.wv.get_vector(tmp_word)
        if similarity=='cos':
            tmp_same_distance = cosine_similarity(tmp_vector, pos_embeddings)
            tmp_opposite_distance = cosine_similarity(tmp_vector, neg_embeddings)
        elif similarity=='euc':
            tmp_same_distance = -1*euc_distance(tmp_vector, pos_embeddings)
            tmp_opposite_distance = -1*euc_distance(tmp_vector, neg_embeddings)
        pos_word_dict[tmp_word] = tmp_same_distance - tmp_opposite_distance

    # Calculate sentiment scores for each word in the negative list
    for tmp_word in neg_list:
        tmp_vector = model.wv.get_vector(tmp_word)
        if similarity=='cos':
            tmp_same_distance = cosine_similarity(tmp_vector, neg_embeddings)
            tmp_opposite_distance = cosine_similarity(tmp_vector, pos_embeddings)
        elif similarity=='euc':
            tmp_same_distance = -1*euc_distance(tmp_vector, neg_embeddings)
            tmp_opposite_distance = -1*euc_distance(tmp_vector, pos_embeddings)
        neg_word_dict[tmp_word] = tmp_same_distance - tmp_opposite_distance
    return pd.Series(pos_word_dict), pd.Series(neg_word_dict)


# Calculate bias data and cluster data for each year based on word embeddings
def get_year_bias_data_dict(model_dict, test_pos_words, test_neg_words, pos_benchmark_dict, neg_benchmark_dict,
                            similarity='cos', year_cols=None):
    if similarity not in ['cos', 'euc']:
        raise
    if year_cols is None:
        year_cols = list(model_dict.keys())
        year_cols = [d for d in year_cols if d != 2020]
    report_positive_year_dict = {}
    report_negative_year_dict = {}
    for tmp_year in year_cols:
        report_positive_year_dict[tmp_year] = []
        report_negative_year_dict[tmp_year] = []
    for tmp_year in year_cols:
        for tmp_word in pos_benchmark_dict[tmp_year]:
            tmp_embedding = model_dict[tmp_year].wv.get_vector(tmp_word)
            report_positive_year_dict[tmp_year].append(tmp_embedding)
        for tmp_word in neg_benchmark_dict[tmp_year]:
            tmp_embedding = model_dict[tmp_year].wv.get_vector(tmp_word)
            report_negative_year_dict[tmp_year].append(tmp_embedding)

    report_year_positive_embedding_dict = {}
    report_year_negative_embedding_dict = {}
    average_distance_dict = {}
    for tmp_year in year_cols:
        report_year_positive_embedding_dict[tmp_year] = np.array(report_positive_year_dict[tmp_year]).mean(axis=0)
        report_year_negative_embedding_dict[tmp_year] = np.array(report_negative_year_dict[tmp_year]).mean(axis=0)

        # Calculate average distance based on the selected similarity measure
        if similarity == 'cos':
            average_distance_dict[tmp_year] = 1-cosine_similarity(report_year_positive_embedding_dict[tmp_year], report_year_negative_embedding_dict[tmp_year])
        else:
            average_distance_dict[tmp_year] = 1-euc_distance(report_year_positive_embedding_dict[tmp_year], report_year_negative_embedding_dict[tmp_year])

    mc_pos_bias_data_dict = {'words':test_pos_words}
    mc_neg_bias_data_dict = {'words':test_neg_words}
    mc_pos_cluster_data_dict = {'words':test_pos_words}
    mc_neg_cluster_data_dict = {'words':test_neg_words}

    for i in year_cols:
        tmp_report_model = model_dict[i].wv
        # Get the positive and negative reference vectors for the current year
        report_benchmark_positive_embedding = report_year_positive_embedding_dict[i]
        report_benchmark_negative_embedding = report_year_negative_embedding_dict[i]

        # Calculate distances between test positive words and negative reference vectors
        mc_report_distance_pos = distance_between_benchmark_vector(tmp_report_model, test_pos_words, report_benchmark_negative_embedding, similarity)[0]
        # Calculate distances between test negative words and positive reference vectors
        mc_report_distance_neg = distance_between_benchmark_vector(tmp_report_model, test_neg_words, report_benchmark_positive_embedding, similarity)[0]

        # Calculate distances between test positive words and positive reference vectors
        mc_report_out_distance_pos = distance_between_benchmark_vector(tmp_report_model, test_pos_words, report_benchmark_positive_embedding, similarity)[0]
        # Calculate distances between test negative words and negative reference vectors
        mc_report_out_distance_neg = distance_between_benchmark_vector(tmp_report_model, test_neg_words, report_benchmark_negative_embedding, similarity)[0]

        # Bias Data
        mc_pos_bias_data_dict[i] = mc_report_out_distance_pos - mc_report_distance_pos
        mc_neg_bias_data_dict[i] = mc_report_out_distance_neg - mc_report_distance_neg
        # Cluster Data
        mc_pos_cluster_data_dict[i] = mc_report_out_distance_pos
        mc_neg_cluster_data_dict[i] = mc_report_out_distance_neg

    # Convert bias data and cluster data to DataFrame format
    mc_pos_bias_data_pd = pd.DataFrame(mc_pos_bias_data_dict).set_index(['words'])
    mc_neg_bias_data_pd = pd.DataFrame(mc_neg_bias_data_dict).set_index(['words'])
    mc_pos_cluster_data_pd = pd.DataFrame(mc_pos_cluster_data_dict).set_index(['words'])
    mc_neg_cluster_data_pd = pd.DataFrame(mc_neg_cluster_data_dict).set_index(['words'])

    return mc_pos_cluster_data_pd, mc_neg_cluster_data_pd, mc_pos_bias_data_pd, mc_neg_bias_data_pd, average_distance_dict


### generate results for annual report and financial news
overlap_with_news_pos_words = [d for d in report_valid_positive_word_list if d in news_valid_positive_word_list]
overlap_with_news_neg_words = [d for d in report_valid_negative_word_list if d in news_valid_negative_word_list]
overlap_with_cc_pos_words = [d for d in report_valid_positive_word_list if d in cc_valid_positive_word_list] ##
overlap_with_cc_neg_words = [d for d in report_valid_negative_word_list if d in cc_valid_negative_word_list] ##

overlap_pos_benchmark = [d for d in report_valid_positive_word_list if (d in news_valid_positive_word_list) and (d in cc_valid_positive_word_list)]
overlap_neg_benchmark = [d for d in report_valid_negative_word_list if (d in news_valid_negative_word_list) and (d in cc_valid_negative_word_list)]

# get overlapping words of 10k
cc_10k_set = set(report_wv_dict[2003].wv.key_to_index.keys())
for i in range(2003, 2019):
    cc_10k_set = cc_10k_set.intersection(set(report_wv_dict[i].wv.key_to_index.keys()))
overlap_with_cc_pos_words = [d for d in cc_valid_positive_word_list if d in cc_10k_set]
overlap_with_cc_neg_words = [d for d in cc_valid_negative_word_list if d in cc_10k_set]

report_wv_dict, news_wv_dict, cc_wv_dict = all_report_wv_dict['ini'], all_news_wv_dict['ini'], all_cc_wv_dict['ini']

## Get benchmark words per year
threshold=0
similarity='cos'
report_pos_benchmark_dict, report_neg_benchmark_dict = {}, {}
news_pos_benchmark_dict, news_neg_benchmark_dict = {}, {}
cc_pos_benchmark_dict, cc_neg_benchmark_dict = {}, {}

# Sentiment intensity calculation for annual reports
for year in range(1997, 2019):
    report_pos_overall_sentiment, report_neg_overall_sentiment = sentiment_word_intensity(report_wv_dict[year], report_valid_positive_word_list, report_valid_negative_word_list, similarity=similarity)
    tmp_pos_benchmark_words = report_pos_overall_sentiment[report_pos_overall_sentiment>threshold].index
    tmp_neg_benchmark_words = report_neg_overall_sentiment[report_neg_overall_sentiment>threshold].index
    report_pos_benchmark_dict[year] = list(tmp_pos_benchmark_words)
    report_neg_benchmark_dict[year] = list(tmp_neg_benchmark_words)
# Sentiment intensity calculation for news
for year in range(1997, 2019):
    news_pos_overall_sentiment, news_neg_overall_sentiment = sentiment_word_intensity(news_wv_dict[year], news_valid_positive_word_list, news_valid_negative_word_list, similarity=similarity)
    tmp_pos_benchmark_words = news_pos_overall_sentiment[news_pos_overall_sentiment>threshold].index
    tmp_neg_benchmark_words = news_neg_overall_sentiment[news_neg_overall_sentiment>threshold].index
    news_pos_benchmark_dict[year] = list(tmp_pos_benchmark_words)
    news_neg_benchmark_dict[year] = list(tmp_neg_benchmark_words)
# Sentiment intensity calculation for conference calls
for year in range(2003, 2019):
    cc_pos_overall_sentiment, cc_neg_overall_sentiment = sentiment_word_intensity(cc_wv_dict[year], cc_valid_positive_word_list, cc_valid_negative_word_list, similarity=similarity)
    tmp_pos_benchmark_words = cc_pos_overall_sentiment[cc_pos_overall_sentiment>threshold].index
    tmp_neg_benchmark_words = cc_neg_overall_sentiment[cc_neg_overall_sentiment>threshold].index
    cc_pos_benchmark_dict[year] = list(tmp_pos_benchmark_words)
    cc_neg_benchmark_dict[year] = list(tmp_neg_benchmark_words)


report_news_pos_dict, report_news_neg_dict = {}, {}
for year in report_pos_benchmark_dict:
    report_news_pos_dict[year] = [d for d in report_pos_benchmark_dict[year] if d in news_pos_benchmark_dict[year]]
    report_news_neg_dict[year] = [d for d in report_neg_benchmark_dict[year] if d in news_neg_benchmark_dict[year]]
report_cc_pos_dict, report_cc_neg_dict = {}, {}
for year in cc_pos_benchmark_dict:
    report_cc_pos_dict[year] = [d for d in report_pos_benchmark_dict[year] if d in cc_pos_benchmark_dict[year]]
    report_cc_neg_dict[year] = [d for d in report_neg_benchmark_dict[year] if d in cc_neg_benchmark_dict[year]]


similarity='cos'
_, _, report_news_pos_bias_data_pd, report_news_neg_bias_data_pd, report_news_average_distance_dict \
            = get_year_bias_data_dict(report_wv_dict, overlap_with_news_pos_words, overlap_with_news_neg_words,
                                 report_news_pos_dict, report_news_neg_dict, similarity=similarity)
_, _, news_news_pos_bias_data_pd, news_news_neg_bias_data_pd, news_news_average_distance_dict \
            = get_year_bias_data_dict(news_wv_dict, overlap_with_news_pos_words, overlap_with_news_neg_words,
                                 report_news_pos_dict, report_news_neg_dict, similarity=similarity)

_, _, report_cc_pos_bias_data_pd, report_cc_neg_bias_data_pd, report_cc_average_distance_dict \
        = get_year_bias_data_dict(report_wv_dict, overlap_with_cc_pos_words, overlap_with_cc_neg_words,
                             report_cc_pos_dict, report_cc_neg_dict, similarity=similarity, year_cols=list(range(2003, 2019)))
_, _, cc_cc_pos_bias_data_pd, cc_cc_neg_bias_data_pd, cc_cc_average_distance_dict \
        = get_year_bias_data_dict(cc_wv_dict, overlap_with_cc_pos_words, overlap_with_cc_neg_words,
                                  report_cc_pos_dict, report_cc_neg_dict, similarity=similarity, year_cols=list(range(2003, 2019)))


model_type = 'ini'
similarity='cos'
threshold=0.0

report_wv_dict = all_report_wv_dict[model_type]
report_pos_benchmark_dict, report_neg_benchmark_dict = {}, {}
for year in range(1997, 2019):
    # Calculating the sentiment intensity of positive and negative words in annual reports for the current year
    report_pos_overall_sentiment, report_neg_overall_sentiment = sentiment_word_intensity(report_wv_dict[year], report_valid_positive_word_list, report_valid_negative_word_list,
                                                                                          similarity=similarity)
    # Filtering out words with sentiment intensity above the threshold as benchmark words
    tmp_pos_benchmark_words = report_pos_overall_sentiment[report_pos_overall_sentiment>threshold].index
    tmp_neg_benchmark_words = report_neg_overall_sentiment[report_neg_overall_sentiment>threshold].index
    # Storing the positive and negative benchmark words for the current year
    report_pos_benchmark_dict[year] = list(tmp_pos_benchmark_words)
    report_neg_benchmark_dict[year] = list(tmp_neg_benchmark_words)
# Calculating bias data for positive and negative words in annual reports
_, _, report_pos_bias_data_pd, report_neg_bias_data_pd, _ \
            = get_year_bias_data_dict(report_wv_dict, report_valid_positive_word_list, report_valid_negative_word_list,
                                 report_pos_benchmark_dict, report_neg_benchmark_dict, similarity=similarity)


news_wv_dict = all_news_wv_dict[model_type]
news_pos_benchmark_dict, news_neg_benchmark_dict = {}, {}
for year in range(1997, 2019):
    news_pos_overall_sentiment, news_neg_overall_sentiment = sentiment_word_intensity(news_wv_dict[year], news_valid_positive_word_list, news_valid_negative_word_list,
                                                                                          similarity='cos')
    tmp_pos_benchmark_words = news_pos_overall_sentiment[news_pos_overall_sentiment>threshold].index
    tmp_neg_benchmark_words = news_neg_overall_sentiment[news_neg_overall_sentiment>threshold].index
    news_pos_benchmark_dict[year] = list(tmp_pos_benchmark_words)
    news_neg_benchmark_dict[year] = list(tmp_neg_benchmark_words)

_, _, news_pos_bias_data_pd, news_neg_bias_data_pd, _ \
            = get_year_bias_data_dict(news_wv_dict, news_valid_positive_word_list, news_valid_negative_word_list,
                                 news_pos_benchmark_dict, news_neg_benchmark_dict, similarity=similarity)



cc_wv_dict = all_cc_wv_dict[model_type]
cc_pos_benchmark_dict, cc_neg_benchmark_dict = {}, {}
for year in range(2003, 2019):
    cc_pos_overall_sentiment, cc_neg_overall_sentiment = sentiment_word_intensity(cc_wv_dict[year], cc_valid_positive_word_list, cc_valid_negative_word_list,
                                                                                          similarity='cos')
    tmp_pos_benchmark_words = cc_pos_overall_sentiment[cc_pos_overall_sentiment>threshold].index
    tmp_neg_benchmark_words = cc_neg_overall_sentiment[cc_neg_overall_sentiment>threshold].index
    cc_pos_benchmark_dict[year] = list(tmp_pos_benchmark_words)
    cc_neg_benchmark_dict[year] = list(tmp_neg_benchmark_words)

_, _, cc_pos_bias_data_pd, cc_neg_bias_data_pd, _ \
            = get_year_bias_data_dict(cc_wv_dict, cc_valid_positive_word_list, cc_valid_negative_word_list,
                                 cc_pos_benchmark_dict, cc_neg_benchmark_dict, similarity=similarity)


# The average distance between the two groups of words
report_news_average_distance_dict = pd.Series(report_news_average_distance_dict)
news_news_average_distance_dict = pd.Series(news_news_average_distance_dict)
report_cc_average_distance_dict = pd.Series(report_cc_average_distance_dict)
cc_cc_average_distance_dict = pd.Series(cc_cc_average_distance_dict)

print(report_news_average_distance_dict)
# get the df tf of words in reports
year_cols = list(range(2003,2019))
words = overlap_with_cc_pos_words + overlap_with_cc_neg_words
document_df_dict = {d:[] for d in year_cols}
document_df_dict['words'] = words
document_tf_dict = {d:[] for d in year_cols}
document_tf_dict['words'] = words
for tmp_word in words:
    for tmp_year in year_cols:
        normalized_df = annual_report_counter[tmp_year][tmp_word]
        normalized_tf = annual_report_counter[tmp_year][tmp_word]
        document_df_dict[tmp_year].append(normalized_df)
        document_tf_dict[tmp_year].append(normalized_tf)
report_cc_df_pd = pd.DataFrame(document_df_dict).set_index(['words'])
report_cc_tf_pd = pd.DataFrame(document_tf_dict).set_index(['words'])


# generate stata data for annual report and financial news
report_mc_positive_result = get_data_for_stata(report_cc_pos_bias_data_pd, report_cc_tf_pd, report_cc_df_pd, list(range(2003,2019)), report_cc_average_distance_dict, [])
report_mc_negative_result = get_data_for_stata(report_cc_neg_bias_data_pd, report_cc_tf_pd, report_cc_df_pd, list(range(2003,2019)), report_cc_average_distance_dict, [])
cc_mc_positive_result = get_data_for_stata(cc_cc_pos_bias_data_pd, cc_tf_pd, cc_df_pd, list(range(2003,2019)), cc_cc_average_distance_dict, [])
cc_mc_negative_result = get_data_for_stata(cc_cc_neg_bias_data_pd, cc_tf_pd, cc_df_pd, list(range(2003,2019)), cc_cc_average_distance_dict, [])

report_mc_positive_result['sentiment'] = 1
report_mc_negative_result['sentiment'] = 0
cc_mc_positive_result['sentiment'] = 1
cc_mc_negative_result['sentiment'] = 0

report_mc_positive_result['source'] = 'report'
report_mc_negative_result['source'] = 'report'
cc_mc_positive_result['source'] = 'cc'
cc_mc_negative_result['source'] = 'cc'

report_mc_positive_result['word_source'] = report_mc_positive_result['word'] + '_report'
report_mc_negative_result['word_source'] = report_mc_negative_result['word'] + '_report'
cc_mc_positive_result['word_source'] = cc_mc_positive_result['word'] + '_cc'
cc_mc_negative_result['word_source'] = cc_mc_negative_result['word'] + '_cc'

report_cc_data = pd.concat([report_mc_positive_result, report_mc_negative_result, cc_mc_positive_result, cc_mc_negative_result])
report_cc_data.to_csv('result/cc.csv')



# generate stata data for annual report and financial news
report_mc_positive_result = get_data_for_stata(report_news_pos_bias_data_pd, report_tf_pd, report_df_pd, list(range(1997,2019)), report_news_average_distance_dict, [])
report_mc_negative_result = get_data_for_stata(report_news_neg_bias_data_pd, report_tf_pd, report_df_pd, list(range(1997,2019)), report_news_average_distance_dict, [])
news_mc_positive_result = get_data_for_stata(news_news_pos_bias_data_pd, news_tf_pd, news_df_pd, list(range(1997,2019)), news_news_average_distance_dict, [])
news_mc_negative_result = get_data_for_stata(news_news_neg_bias_data_pd, news_tf_pd, news_df_pd, list(range(1997,2019)), news_news_average_distance_dict, [])

report_mc_positive_result['sentiment'] = 1
report_mc_negative_result['sentiment'] = 0
news_mc_positive_result['sentiment'] = 1
news_mc_negative_result['sentiment'] = 0

report_mc_positive_result['source'] = 'report'
report_mc_negative_result['source'] = 'report'
news_mc_positive_result['source'] = 'news'
news_mc_negative_result['source'] = 'news'

report_mc_positive_result['word_source'] = report_mc_positive_result['word'] + '_report'
report_mc_negative_result['word_source'] = report_mc_negative_result['word'] + '_report'
news_mc_positive_result['word_source'] = news_mc_positive_result['word'] + '_news'
news_mc_negative_result['word_source'] = news_mc_negative_result['word'] + '_news'

report_news_data = pd.concat([report_mc_positive_result, report_mc_negative_result, news_mc_positive_result, news_mc_negative_result])
report_news_data.to_csv('result/news.csv')