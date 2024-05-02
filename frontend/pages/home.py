import streamlit as st


def display_home():
    st.title("Tools for the anaylis of public opinion with a GloVe model")

    st.markdown(
        """
    ## Project Overview

    This project explores techniques described in recent scholarly papers and adapts them to analyze public opinion about BigTech companies in the UK. Our aim is to develop robust analysis tools tailored to this specific context.

    ## Objective

    The primary goal is to devise methods that can:
    - Track positions of newspapers and political parties on specific issues.
    - Identify words and phrases that are most indicative of their stance on BigTechs.

    ## Data Sources

    Our analysis tools are built upon a dense and balanced database of relevant texts, which includes:
    - Speeches from the House of Commons related to BigTechs, spanning from 2010 to 2019.
    - Articles from five major British newspapers covering the same theme from 2010 to 2023:
        - *The Guardian*
        - *The Telegraph*
        - *The Daily Mail*
        - *The Daily Express*
        - *Metro*

    ## Approach

    We adapt methodologies from referenced papers to suit the topic of public opinion on BigTechs in the UK, enhancing our ability to derive insightful analytics from textual data.
                
    We trained a GloVe model on the database, and we defined simple and relevant axes in the embeddings space. The axes defined aim to have a global feel of the opinion from the media and the politicians regarding tech companies. 
                
    The two axes we work with are :
    - Axis 1 : a positive pole formed of words describing economic technologic laisser-faire, and a negative pole formed of words describing more regulation. 
    - Axis 2 : a positive pole with positive words, and a negative pole with negative words.
                
    ## The two main parts

    #### Curves analysis 

    In this part we can visualize, given some filtering arguments, 2 types of curves :
    - The curves of cosine similarity of the embeddings of different corpuses with the axes.
    - The curves of polarization between two different corpuses. 

    We can also visualize them both on the same graphs, for the same sources and axis.
                
    #### Words analysis

    In this part we can visualize, given some filtering arguments, 3 types of measures :
    - The words with projected embeddings on an axis that are varying the most between two years.
    - The words that define the poles of the axes that are the most responsible for attracting the corpus towards them between two years.
    - A spectral clustering of the words embeddings of words in filtered corpuses.
    
                
    """,
        unsafe_allow_html=True,
    )
