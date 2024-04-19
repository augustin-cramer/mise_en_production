import json
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from Processing.text_cleaning import *
from GloVe.weights import *
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from Axes.projection_functions import *
from Axes.models import *
from Axes.filter_words import *
from Processing.preprocess_parliament import *
import os 
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


events_keywordskeywords = list(set(clean(events_keywords, 'unigram')))
new_topics = list(set(clean(new_topics, 'unigram')))

def process_year_data(year, model_words_year, with_parliament=True):
    
    if with_parliament:
        with open(f'data/with parliament/words/Finalwords_{year}.json') as f:
            words_year = json.load(f)
    if not with_parliament:
        with open(f'data/without parliament/words/Finalwords_{year}_WP.json') as f:
            words_year = json.load(f)
        
    weights_year = get_weights_word2vec(words_year, a=1e-3)

    if with_parliament:
        with open(f'data/with parliament/vocabs/vocab_{year}.json') as f:
            vocab_year = json.load(f)
    if not with_parliament:
        with open(f'data/without parliament/vocabs/vocab_{year}_WP.json') as f:
            vocab_year = json.load(f)
        
    vocab_embed_year = [weights_year[i] * model_words_year[i] for i in vocab_year]

    df_words_year = pd.DataFrame(zip(vocab_year, vocab_embed_year), columns=['text', 'embedding'])

    axis_v1 = axis_vector(pos_1, neg_1, model_words_year)
    axis_v2 = axis_vector(pos_2, neg_2, model_words_year)

    df_words_year['cos axe 1'] = df_words_year['text'].apply(cosine_with_axis, axis_v=axis_v1, model_sentences=model_words_year)
    df_words_year['cos axe 2'] = df_words_year['text'].apply(cosine_with_axis, axis_v=axis_v2, model_sentences=model_words_year)

    df_words_year['year'] = year if year <= 2019 else year - 18090  # Adjust year for 20110 and beyond
    
    return df_words_year

def var_embed_real(word:str, df1, df2, cos_axe:str):
    try :
        return(df2.loc[df2['text'] == word][cos_axe].values[0] - df1.loc[df1['text'] == word][cos_axe].values[0])
    except :
        return None
    
def is_in_keywords(word):
    if word in new_topics:
        return True
    if word in events_keywords:
        return True
    return False

def process_yearly_data(df, year, with_parliament=True):
     # Load the words from the file
    if with_parliament:
        with open(f'data/with parliament/words/Finalwords_{year}.json') as f:
            words = json.load(f)
    if not with_parliament:
        with open(f'data/without parliament/words/Finalwords_{year}_WP.json') as f:
            words = json.load(f)
    
    # Calculate word counts
    word_counts = Counter(words)
    
    # Apply the word count to the dataframe
    df['word count'] = df['text'].apply(lambda word: word_counts.get(word, 0))
    
    # Filter rows where 'word count' is greater than 100
    df_filtered = df[df['word count'] > 100]
    
    # Apply the check for 'in keywords'
    df_filtered['in keywords'] = df_filtered['text'].apply(is_in_keywords)
    
    # Filter by 'in keywords'
    df_keywords = df_filtered[df_filtered['in keywords']]
    
    return df_keywords

def get_top_variations(df_keywords, axis, number):
    """Sorts the dataframe by the specified axis and gets the top number variations."""
    var_up = df_keywords.sort_values(by=[f'var cos axe {axis}'], ascending=False).head(number)[['text', 'year', f'var cos axe {axis}']]
    var_down = df_keywords.sort_values(by=[f'var cos axe {axis}'], ascending=True).head(number)[['text', 'year', f'var cos axe {axis}']]
    return var_up, var_down

def vizualize_top_variations(df_keywords, axis_1, axis_2=None, variation_1 = 'up', variation_2 = 'down', number=20, with_parliament=True):

    var_up_1, var_down_1 = get_top_variations(df_keywords, axis_1, number)

    if axis_2:
        var_up_2, var_down_2 = get_top_variations(df_keywords, axis_2, number)
    else:
        var_up_2, var_down_2 = var_up_1, var_down_1
        axis_2 = axis_1

    if variation_1 == 'down':
        var_up_1 = var_down_1
    if variation_2 == 'up':
        var_down_2 = var_up_2

    '''fig = make_subplots(rows=2, cols=1)

    # Add bar plot for increasing variations
    fig.add_trace(go.Bar(x=var_up_1['text'], y=var_up_1[f'var cos axe {axis_1}'], name='Increasing'), row=1, col=1)

    # Add bar plot for decreasing variations
    fig.add_trace(go.Bar(x=var_down_2['text'], y=var_down_2[f'var cos axe {axis_2}'], name='Decreasing'), row=2, col=1)

    fig.update_layout(title_text= f"Extreme embedding variation on axis {axis_1} and {axis_2}")
    fig.update_layout(autosize=False, width=1000, height=800)
    fig.write_html(f"plots/Word_analysis/Extreme embedding variation on axis {axis_1} and {axis_2} ; variation_1 = {variation_1}, variation_2 = {variation_2}, with_parliament = {with_parliament}.png")

    fig.show()'''

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # Create two subplots vertically

    # Plot for increasing variations
    axs[0].bar(var_up_1['text'], var_up_1[f'var cos axe {axis_1}'], color='skyblue', label='Increasing')
    axs[0].set_title(f"Increasing Variation on Axis {axis_1}")
    axs[0].legend()
    axs[0].set_xticklabels(var_up_1['text'], rotation=45, ha="right")

    # Plot for decreasing variations
    axs[1].bar(var_down_2['text'], var_down_2[f'var cos axe {axis_2}'], color='lightgreen', label='Decreasing')
    axs[1].set_title(f"Decreasing Variation on Axis {axis_2}")
    axs[1].legend()
    axs[1].set_xticklabels(var_down_2['text'], rotation=45, ha="right")

    # Overall figure title and layout adjustments
    plt.suptitle(f"Extreme Embedding Variation on Axis {axis_1} and {axis_2}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Saving the figure
    plt.savefig(f"plots/Word_analysis/Extreme embedding variation on axis {axis_1} and {axis_2} ; variation_1 = {variation_1}, variation_2 = {variation_2}, with_parliament = {with_parliament}.png")

    plt.show()


def word_variations(year, axis_1 = 1, axis_2 = 1, variation_1 = 'up', variation_2 = 'down', with_parliament=True):

    if os.path.exists(f"plots/Word_analysis/Extreme embedding variation on axis {axis_1} and {axis_2} ; variation_1 = {variation_1}, variation_2 = {variation_2}, with_parliament = {with_parliament}.png"):
        print('already computed')
        img = mpimg.imread(f"plots/Word_analysis/Extreme embedding variation on axis {axis_1} and {axis_2} ; variation_1 = {variation_1}, variation_2 = {variation_2}, with_parliament = {with_parliament}.png")
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    else:

        print('computing')

        if year > 2019 :
            year = year + 18090
        i = eval(str(year)[-1:])
        
        # Assuming you have a dictionary `model_words` with keys as years and values as the corresponding model for that year
        current_df = process_year_data(year, models_w[i], with_parliament)
        previous_df = process_year_data(year-1, models_w[i-1], with_parliament)

        for cos_axe in ['cos axe 1', 'cos axe 2']:
            var_column_name = f'var {cos_axe}'
            current_df[var_column_name] = current_df['text'].apply(var_embed_real, 
                                                                df1=previous_df, 
                                                                df2=current_df, 
                                                                cos_axe=cos_axe)

        current_df = process_yearly_data(current_df, year, with_parliament)

        vizualize_top_variations(current_df, axis_1, axis_2, variation_1, variation_2, with_parliament=with_parliament)
        