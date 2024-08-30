# WOLVES: Word List Vector for Sentiment

The **WOLVES** (Word List Vector for Sentiment) is an innovative sentiment analysis algorithm that combines manually defined lists of sentiment words with word embedding techniques to quantify the change in text sentiment over time. WOLVES dynamically update the sentiment lexicon according to changing semantics and sentiment intensity, thus solving the limitations of static sentiment analysis models. In addition, WOLVES can help extract valuable business intelligence from financial texts, which provides powerful support for business decision-making.
</br>
The official implementation of paper "[The Effects of Sentiment Evolution in Financial Texts: A Word Embedding Approach](https://www.tandfonline.com/doi/full/10.1080/07421222.2023.2301176)". 
 

## Dataset Information
The dataset is avaliable at `resources` directory, which contains the following data files:

| Dataset                               | Description                              |
|---------------------------------------|------------------------------------------|
| LM_positive                           |Positive words in the Loughran and McDonald dictionary|
| LM_negative                           |Negative words in the Loughran and McDonald dictionary|
| H4_positive                           |Positive words in the H4 dictionary|
| H4_negative                           |Negative words in the H4 dictionary|
| LoughranMcDonald_MasterDictionary_2020.csv |Loughran-McDonald Dictionary|
| mda_tf.csv                           |Sentiment words data in the Form 10Ks|
| index.csv                            |Index and year information for Form 10Ks|
| cc_df.pk                             |Word frequency data in Conference Calls|
| news_df.pk                           |Word frequency data in Financial News |
| annual_report_df.pk                  |Word frequency data in Annual Report|
| mda_tf.csv                           |Word data in MD&A|



## Code Structure

```
WOLVES-Word-List-Vector-for-Sentiment-main/
  - model                 # Word2Vec model directory
  - resources             # Dataset directory
    - annual_report_df.pk
    - cc_df.pk
    - news_df.pk
    - H4_negative
    - H4_positive
    - MC_negative
    - MC_positive
    - index.csv
    - LoughranMcDonald_MasterDictionary_2020.csv
    - mda_tf.csv
  - stata        # Stata analysis program directory
    - stage1_cc
    - stage1_news
  - result       # Saved results directory
  - Word_Sentiment_Analysis.py    # Word sentiment analysis program
  - MDA_Analysis.py      #MD&A sentiment analysis program
```


## Installation
### Clone the repository to your local machine:
```bash
git clone https://github.com/Mingze0111/WOLVES-Word-List-Vector-for-Sentiment.git
```
### Dependencies
To install the required dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

## Usage of `Word_Sentiment_Analysis.py`
### Key Functions:
The `get_data_for_stata` is used to generate a dataset suitable for Stata analysis. The dataset contains a variety of statistical information and features of the financial text data that can be used for further data analysis and modeling.
```python
def get_data_for_stata(target_pd, tf_pd, df_pd, year_cols, group_distance_data, benchmark_word_set):
    """
    Generate Stata-compatible dataset from financial text data and related statistics

    Parameters:
        target_pd (pandas.DataFrame): DataFrame containing target word data.
        tf_pd (pandas.DataFrame): DataFrame containing term frequency data.
        df_pd (pandas.DataFrame): DataFrame containing document frequency data.
        year_cols (list): List of years for which data is available.
        group_distance_data (pandas.DataFrame): DataFrame containing similarity data.
        benchmark_word_set (set): Set of benchmark words.

    Returns:
        pandas.DataFrame: Stata-compatible dataset containing various statistics and flags.
"""
```
### Application:
You can run the code to apply word sentiment analysis to financial text data and get corresponding statistics:

```bash
python Word_Sentiment_Analysis.py
```
Output: `cc.csv` and `news.csv`. Output example: 

|         |   word  | year |    score    | LM_H4_word | MC_word | average_tf | average_df | group_distance | benchmark_word | sentiment | source | word_source |
|:-------:|:-------:|:----:|:-----------:|:----------:|:-------:|:----------:|:----------:|:--------------:|:--------------:|:---------:|:------:|:-----------:|
|    0    |   able  | 1997 | 0.246407986 |     1      |    1    | 2.094902306| 2.094902306|    0.826524228 |       0        |     1     | report | able_report |
|    1    |   able  | 1998 |  0.21979484 |     1      |    1    | 2.458366319| 2.458366319|    0.830164492 |       0        |     1     | report | able_report |
|    2    |   able  | 1999 | 0.236650333 |     1      |    1    | 2.874019678| 2.874019678|    0.844888464 |       0        |     1     | report | able_report |
|   ...   |   ...   | ...  | ...         |    ...     |  ...    |    ...     | ...        |    ...         |     ...        | ...       | ...    |     ...     |


You can use `Stage1_cc` and `Stage1_news` for further analysis.

## Usage of `MDA_Analysis.py`
### Key Functions:
The `sentiment_word_intensity` function calculates the sentiment intensity of positive and negative words based on their embeddings in the word embedding model. By comparing the distances between word embeddings and centroids of positive and negative word clusters, this function provides insights into the strength of sentiment conveyed by each word.  
The function returns the results as Pandas Series, with words as index and sentiment intensity scores as values.
```python
def sentiment_word_intensity(model, pos_list, neg_list, similarity='cos'):
    """
    Calculate the sentiment intensity of positive and negative words based on word embeddings

    Parameters:
    - model (Word2Vec): Word embedding model.
    - pos_list (list): List of positive words.
    - neg_list (list): List of negative words.
    - similarity (str, optional): Similarity measure to be used ('cos' for cosine similarity or 'euc' for Euclidean distance). Defaults to 'cos'.

    Returns:
    - pos_intensity (Series): Series containing sentiment intensity scores for positive words.
    - neg_intensity (Series): Series containing sentiment intensity scores for negative words.
    """
```

</br>

The `get_year_bias_data_dict` function is designed to calculate bias and cluster data for a given word embedding model across different years. This function leverages word embeddings to quantify semantic relationships between words and provides insights into how word meanings evolve over time.  
The function returns the results as DataFrames and dictionaries for further analysis and visualization.  

```python

def get_year_bias_data_dict(model_dict, test_pos_words, test_neg_words, pos_benchmark_dict, neg_benchmark_dict,
                            similarity='cos', year_cols=None):
    """
    Calculate bias data and cluster data for a given word embedding model across different years

    Parameters:
    - model_dict (dict): A dictionary containing word embedding models for different years.
    - test_pos_words (list): List of positive test words.
    - test_neg_words (list): List of negative test words.
    - pos_benchmark_dict (dict): A dictionary containing positive benchmark words for each year.
    - neg_benchmark_dict (dict): A dictionary containing negative benchmark words for each year.
    - similarity (str, optional): Similarity measure to be used ('cos' for cosine similarity or 'euc' for Euclidean distance). Defaults to 'cos'.
    - year_cols (list, optional): List of years to consider. Defaults to None.

    Returns:
    - mc_pos_cluster_data_pd (DataFrame): DataFrame containing cluster data for positive test words.
    - mc_neg_cluster_data_pd (DataFrame): DataFrame containing cluster data for negative test words.
    - mc_pos_bias_data_pd (DataFrame): DataFrame containing bias data for positive test words.
    - mc_neg_bias_data_pd (DataFrame): DataFrame containing bias data for negative test words.
    - average_distance_dict (dict): Dictionary containing average distances between positive and negative reference vectors for each year.
    """
```

### Application:
You can run the code to apply MD&A analysis to financial text data and get corresponding result:

```bash
python MDA_Analysis.py
```

Output: `mda_sentiment.csv`. Output example: 
| fila_name                                         | top_neg_ini | middle_neg_ini | buttom_neg_ini |
|:-------:|:-------:|:-------:|:-------:|
| 19970103_10-k_edgar_data_68330_0000950146-97-000011_1.txt | -1.011716575 | -0.891375949   | -1.095006192   |
| 19970109_10-k_edgar_data_882835_0000931763-97-000022_1.txt | -1.446545355 | -0.63005235    | 1.154805075    |
| 19970110_10-k405_edgar_data_729533_0000950135-97-000077_1.txt | 0.264839489 | 1.710236305    | -0.33126256    |
| ... | ... | ...   | ...    |


**After executing the WOLVES algorithm, you can use the `get_change_trend` function to implement it in real practice:**

This function `get_change_trend` calculates the change trend using Ordinary Least Squares (OLS) regression. It fits an OLS regression model to estimate the change trend of the target variable over time.
```python
def get_change_trend(target_pd, year_cols, with_year=True):
    """
    Calculate the change trend using Ordinary Least Squares (OLS) regression.

    Parameters:
    - target_pd (DataFrame): DataFrame containing the target variable over different years.
    - year_cols (list): List of years to include in the analysis.
    - with_year (bool): Flag indicating whether to include years as independent variables. Default is True.

    Returns:
    - DataFrame: DataFrame containing the change trend analysis results, including coefficient and p-value.
    """
```


## Citation
```
@article{https://doi.org/10.1080/07421222.2023.2301176,  
    author = {Jiexin Zheng, Ka Chung Ng, Rong Zheng and Kar Yan Tam},  
    title = {The Effects of Sentiment Evolution in Financial Texts: A Word Embedding Approach},  
    journal = {Journal of Management Information Systems},  
    volume = {41, 2024},  
    number = {n/a},  
    pages = {178-205},  
    keywords = {word embedding, word list, sentiment evolution, textual analysis, strategic communication},  
    doi = {https://doi.org/10.1080/07421222.2023.2301176},  
    url = {https://doi.org/10.1080/07421222.2023.2301176},  
    eprint = {https://doi.org/10.1080/07421222.2023.2301176},  
}
```

## Contact
```

kc-boris.ng@polyu.edu.hk  

```

## License

[MIT](https://choosealicense.com/licenses/mit/)
