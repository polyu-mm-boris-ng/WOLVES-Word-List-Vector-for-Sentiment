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

# load all common words in the context of financial text from Loughran and McDonald (LM) dictionary
master_dictionary_pd = pd.read_csv('resources/LoughranMcDonald_MasterDictionary_2020.csv')
all_common_word_list = list(master_dictionary_pd['Word'].dropna().str.lower().values)
all_common_word_dict = {d:1 for d in all_common_word_list}
all_common_word_set = set(all_common_word_list)


#Specify a Word2Vec model and two lists of words. Compute the distance between these two sets of words
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


# detect the dynamic shift of one words
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


# Get the trend of each wrod on the given year columns
def get_change_trend(target_pd, year_cols, with_year=True):
    target_pd = target_pd.copy()
    if with_year:
        iv = pd.DataFrame({'year':year_cols, 'const':[1]*len(year_cols)})
        iv['year'] = iv['year']-min(year_cols)
    else:
        iv = pd.DataFrame({'const':[1]*len(year_cols)})
    year_coeff_list = []
    year_pvalue_list = []
    const_coeff_list = []
    const_pvalue_list = []
    # Fit OLS model
    for words in target_pd.index:
        target = target_pd.loc[words].loc[year_cols]
        ols_model = sm.OLS(target.values, iv).fit()

        # Store coefficient and p-value
        if with_year:
            year_coeff_list.append(ols_model.params.loc['year'])
            year_pvalue_list.append(ols_model.pvalues.loc['year'])

        # Store coefficient and p-value for constant term
        const_coeff_list.append(ols_model.params.loc['const'])
        const_pvalue_list.append(ols_model.pvalues.loc['const'])
    target_pd = target_pd[year_cols]
    if with_year:
        target_pd['year_coeff'] = year_coeff_list
        target_pd['year_pvalue'] = year_pvalue_list
    target_pd['const_coeff'] = const_coeff_list
    target_pd['const_pvalue'] = const_pvalue_list
    return target_pd


# Specify a word2vec model and two word lists. Compute the distance between words.
def distance_between_groups_builtin(model, word_list1, word_list2, model2=None):
    distance_list = []
    ##Use the n_similarity function from the Gensim library to calculate the similarity between the current word and the second set of words
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


#Calculate the distance (or similarity) between each word in a given list of words and a given benchmark vector
def distance_between_benchmark_vector(model, word_list1, benchmark_vector, similarity='cos'):
    distance_list = []
    for tmp_word in word_list1:
        tmp_vector = model.get_vector(tmp_word) #Get word vectors for a given word from the word vector model
        if similarity == 'cos':       #Calculate using cosine similarity
            tmp_distance = cosine_similarity(tmp_vector, benchmark_vector)
        else:                         #Calculate using Euclidean distance
            tmp_distance = euc_distance(tmp_vector, benchmark_vector)
        distance_list.append(tmp_distance)
    distance_list = np.array(distance_list)
    return distance_list, distance_list


#Calculate the embedding space distance between two group of words
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


#Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
word_dictionary = {}
for d in words.words():
    word_dictionary[d] = 1
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


#Calculate the sentiment intensity of positive and negative words based on word embeddings
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

    # normalize the vectors before calculating the centroid
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
    for tmp_word in pos_list:
        tmp_vector = model.wv.get_vector(tmp_word)
        if similarity=='cos':
            tmp_same_distance = cosine_similarity(tmp_vector, pos_embeddings)
            tmp_opposite_distance = cosine_similarity(tmp_vector, neg_embeddings)
        elif similarity=='euc':
            tmp_same_distance = -1*euc_distance(tmp_vector, pos_embeddings)
            tmp_opposite_distance = -1*euc_distance(tmp_vector, neg_embeddings)
        pos_word_dict[tmp_word] = tmp_same_distance - tmp_opposite_distance
    for tmp_word in neg_list:
        tmp_vector = model.wv.get_vector(tmp_word)
        if similarity=='cos':
            tmp_same_distance = cosine_similarity(tmp_vector, neg_embeddings)
            tmp_opposite_distance = cosine_similarity(tmp_vector, pos_embeddings)
        elif similarity=='euc':
            tmp_same_distance = -1*euc_distance(tmp_vector, neg_embeddings)
            tmp_opposite_distance = -1*euc_distance(tmp_vector, pos_embeddings)
        #Calculate sentiment intensity
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


# Load all models
all_report_wv_dict = {}

# Word embedding models for annual report
model_prefix = 'model/'
report_wv_dict = {}
for i in range(1997, 2019):
    model = gensim.models.Word2Vec.load(str(Path(model_prefix, "{}_ini_w2v.mod".format(i))))
    report_wv_dict[i] = model
# get overlapping words of 10k
final_10k_set = set(report_wv_dict[1997].wv.key_to_index.keys())
for i in range(1997, 2019):
    final_10k_set = final_10k_set.intersection(set(report_wv_dict[i].wv.key_to_index.keys()))

all_report_wv_dict['ini'] = report_wv_dict

# number of valid sentiment words
report_valid_positive_word_list = [d for d in positive_word_list if d in final_10k_set]
report_valid_negative_word_list = [d for d in negative_word_list if d in final_10k_set]
print(len(positive_word_list), len(negative_word_list))
print(len(report_valid_positive_word_list), len(report_valid_negative_word_list))

# Load MD&A Data
item7_tf_data_pd = pd.read_csv('resources/mda_tf.csv', index_col=['fila_name'])

market_variable = pd.read_csv('resources/index.csv')
# get ratio across years
year_index_pd = market_variable[['calendar_year', 'fila_name']].set_index(['fila_name'])
year_word_list = [[n,d] for n,d in year_index_pd.groupby(['calendar_year'])]


# Application
model_type = 'ini'
report_wv_dict = all_report_wv_dict['ini']
threshold = 0.0
report_pos_benchmark_dict, report_neg_benchmark_dict = {}, {}
for year in range(1997, 2019):
    # Calculate overall sentiment intensity for positive and negative words in the specified year
    report_pos_overall_sentiment, report_neg_overall_sentiment = sentiment_word_intensity(report_wv_dict[year],
                                                                                          report_valid_positive_word_list,
                                                                                          report_valid_negative_word_list, similarity='cos')
    # Identify positive and negative benchmark words based on sentiment intensity exceeding threshold
    tmp_pos_benchmark_words = report_pos_overall_sentiment[report_pos_overall_sentiment > threshold].index
    tmp_neg_benchmark_words = report_neg_overall_sentiment[report_neg_overall_sentiment > threshold].index
    report_pos_benchmark_dict[year] = list(tmp_pos_benchmark_words)
    report_neg_benchmark_dict[year] = list(tmp_neg_benchmark_words)

# Apply bias analysis
report_wv_dict = all_report_wv_dict[model_type]
similarity = 'cos'
_, _, report_pos_bias_data_pd, report_neg_bias_data_pd, _ \
    = get_year_bias_data_dict(report_wv_dict, report_valid_positive_word_list, report_valid_negative_word_list,
                              report_pos_benchmark_dict, report_neg_benchmark_dict, similarity=similarity)
positive_word_trend_bias_pd = report_pos_bias_data_pd.copy()
negative_word_trend_bias_pd = report_neg_bias_data_pd.copy()
split_quantile = 3 # Number of quantiles to split negative word trend bias into
model_name = model_type
top_col = 'top_neg_' + model_name
middle_col = 'middle_neg_' + model_name
buttom_col = 'buttom_neg_' + model_name
result_pd_list = []
top_word_list = []
for tmp_year, pd_data in year_word_list:
    tmp_year = int(tmp_year[0])
    if tmp_year < 1997:
        continue
    tmp_year = str(tmp_year)
    tmp_negative_word_trend_bias_pd = negative_word_trend_bias_pd[int(tmp_year)].sort_values(ascending=False)
    # based on rank
    neg_count = len(tmp_negative_word_trend_bias_pd) // split_quantile
    left_count = len(tmp_negative_word_trend_bias_pd) % split_quantile

    # Split words into top, middle, and bottom categories based on rank
    tmp_top_neg_bias_words = tmp_negative_word_trend_bias_pd.iloc[:neg_count + left_count].index
    tmp_middle_neg_bias_words = tmp_negative_word_trend_bias_pd.iloc[neg_count + left_count:-1 * neg_count].index
    tmp_buttom_neg_bias_words = tmp_negative_word_trend_bias_pd.iloc[-1 * neg_count:].index

    top_word_list.append(list(tmp_top_neg_bias_words))

    year_item7_tf = item7_tf_data_pd.loc[pd_data.index]

    pd_data[top_col] = year_item7_tf[year_item7_tf.columns.intersection(tmp_top_neg_bias_words)].sum(axis=1)
    pd_data[middle_col] = year_item7_tf[year_item7_tf.columns.intersection(tmp_middle_neg_bias_words)].sum(axis=1)
    pd_data[buttom_col] = year_item7_tf[year_item7_tf.columns.intersection(tmp_buttom_neg_bias_words)].sum(axis=1)

    result_pd_list.append(pd_data)

result_pd = pd.concat(result_pd_list)[[top_col, middle_col, buttom_col]]
result_pd[[top_col, middle_col, buttom_col]] = (result_pd[[top_col, middle_col, buttom_col]] - result_pd[[top_col, middle_col, buttom_col]].mean()) / result_pd[[top_col, middle_col, buttom_col]].std()
result_pd.to_csv('result/mda_sentiment.csv')