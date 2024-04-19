from __future__ import division
from collections import Counter
import json
from scipy import sparse
import numpy as np
from scipy.sparse import dok_matrix
from mittens import Mittens
import json
import numpy as np
import os
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os, psutil
from tqdm import tqdm
from Processing.text_cleaning import *


def vocab_dic(fichier) :
    '''  Returns
    vocab : the entire list of words that are used, without doubles
    word2idx : dictionary created with this vocabulary
    
    Parameters:
    -----------
    fichier : the words
    '''
    with open(fichier) as f:
        vocab_inter = json.load(f)
    vocab = list(set(vocab_inter))
    word2idx = {v: i for i, v in enumerate(vocab)}
    return vocab, word2idx

def inter_coocc(items, word2idx):
        '''
        Creates the cooccurence matrix for the words in the items

        Parameters:
        -----------
        items : text to process
        '''
        print('dans inter_coocc')
        coocc = dok_matrix((len(word2idx), len(word2idx))) 
        #loop on subItems
        for (j,t) in items:
          word_counts = Counter(t)
          window = list(word_counts.items())
          for i, (word, count1) in enumerate(window):
            for (context, count2) in window[i:i+10]:
                try :
                    coocc[word2idx[word], word2idx[context]] += count1 * count2
                    if context != word:
                        coocc[word2idx[context], word2idx[word]] += count1 * count2
                except : 
                     coocc = coocc
          if j % 10000 == 0:
                print(t)
                print(j)
        return(coocc)

def split(a, n):
    ''' Function to split a list in n evenly subpart'''
    k, m = divmod(len(a), n)
    return ([a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)])

def glove2dict(glove_filename):
    ''' transforms a txt file of embeddings into a dictionary
    Parameters:
    -----------
    glove_filename : embeddings txt file
    '''
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        words = []
        mats = []
        for line in reader :
            if len(clean(line[0], gram='unigram'))>0:
                words.append(clean(line[0], gram='unigram')[0])
                mats.append(np.array(list(map(float, line[1:]))))
    embed = {words[i]: mats[i] for i in range(len(words))}
    return embed

def process_iteration(i, original_embedding):
    start_time = datetime.now()
    print(f"Starting iteration {i} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Memory usage before operation
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"Memory usage before operation: {mem_before:.2f} MB")
    
    vocab_path = f'data/without parliament/vocabs/vocab_201{i}_WP.json'
    cooccurrence_path = f'data/without parliament/glove_cooccurences/glove_cooccurences_201{i}_WP.npz'
    output_path = f'data/without parliament/embeddings/embeddings_201{i}_WP.txt'
    
    with open(vocab_path) as f:
        vocab = json.load(f)
        
    cooccurrence = sparse.load_npz(cooccurrence_path).toarray()
    
    mittens_model = Mittens(n=50, max_iter=1000)
    new_embeddings = mittens_model.fit(cooccurrence, vocab=vocab, initial_embedding_dict=original_embedding)
    
    #a = np.array(list(vocab.keys()))
    a = np.array(vocab)
    b = new_embeddings
    c = np.column_stack((a, b))
    np.savetxt(output_path, c, fmt='%s')
    
    # Memory usage after operation
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"Memory usage after operation: {mem_after:.2f} MB, Difference: {mem_after - mem_before:.2f} MB")
    
    end_time = datetime.now()
    print(f"Completed iteration {i} at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, Duration: {end_time - start_time}")
    
    # Return the path to the new embeddings to be used as the next original_embedding
    return output_path

def parallel_process():
    original_embedding = glove2dict('data/glove.6B/glove.6B.50d.txt')
    #original_embedding = glove2dict('data/without parliament/embeddings/embeddings_201'+str(3)+'_WP.txt')
    for i in tqdm(range(14)):
        #future = executor.submit(process_iteration(i, original_embedding)
        #futures.append(future)
        process_iteration(i, original_embedding)
        print(os.getcwd())
        # Wait for the current iteration to complete before moving to the next
        # This is necessary because each iteration's output is the input for the next
        original_embedding = glove2dict('data/without parliament/embeddings/embeddings_201'+str(i)+'_WP.txt')