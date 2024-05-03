# [The app](https://mep.kub.sspcloud.fr/)

# Mise en production

This project explores techniques described in recent scholarly papers and adapts them to analyze public opinion about BigTech companies in the UK. Our aim is to develop robust analysis tools tailored to this specific context. Specifically, the tools of the project are inspired by :

- [Gentzkow, Matthew, Jesse M. Shapiro, and Matt Taddy](https://scholar.harvard.edu/shapiro/publications/measuring-group-differences-high-dimensional-choices-method-and-application)
- [Dorottya Demszky, Nikhil Garg, Rob Voigt, James Zou, Matthew Gentzkow, Jesse Shapiro, Dan Jurafsky](https://arxiv.org/abs/1904.01596)
- [Austin C. Kozlowski, Matt Taddy, James A. Evans](https://arxiv.org/abs/1803.09288)
- [Gennaro G. and Ash E.](https://www.research-collection.ethz.ch/handle/20.500.11850/468192)

## Overview

<details>
<summary><strong>Objective</strong></summary>
<br>

The primary goal is to devise methods that can:
- Track positions of newspapers and political parties on specific issues.
- Identify words and phrases that are most indicative of their stance on BigTechs.

</details>

<details>
<summary><strong>Data Sources</strong></summary>
<br>

Our analysis tools are built upon a dense and balanced database of relevant texts, which includes:
- Speeches from the House of Commons related to BigTechs, spanning from 2010 to 2019.
- Articles from five major British newspapers covering the same theme from 2010 to 2023:
    - *The Guardian*
    - *The Telegraph*
    - *The Daily Mail*
    - *The Daily Express*
    - *Metro*

</details>

<details>
<summary><strong>Model</strong></summary>
<br>

We adapt methodologies from referenced papers to suit the topic of public opinion on BigTechs in the UK, enhancing our ability to derive insightful analytics from textual data.
                
We trained a GloVe model on the database, and we defined simple and relevant axes in the embeddings space. The axes defined aim to have a global feel of the opinion from the media and the politicians regarding tech companies. 
            
The two axes we work with are :
- Axis 1 : a positive pole formed of words describing economic technologic laisser-faire, and a negative pole formed of words describing more regulation. 
- Axis 2 : a positive pole with positive words, and a negative pole with negative words.

The goal is to project texts and corpuses on those axes to see how they are positioned compared to the poles. 

</details>

<details>
<summary><strong>The two main parts</strong></summary>
<br>

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
</details>

<br>

## Approach

In this project, we focus on what comes after dataprocessing, model training and computation of cosine embeddings with thes axes. The axes are already defined, and we have stored dataframes with the projection of each text embedding on both axes, and also to which tech company each text is related.

This repository is therefore a subpart of a large NLP project, and the focus was to make the tools developped easily accessible and manipulable. We did our best to clean all the scripts and build the most deployable structure. Because of the size of the repository, some unknown errors may still arise, but the tools should nevertheless be usable. The app is here :

<div align="center">

[https://mep.kub.sspcloud.fr/](https://mep.kub.sspcloud.fr/)

<div align="left">

The app contains a description of the tools and an explanation of all the different parameters that can be used to generate analysis. The app employs the continuous integration / continuous deployment using Docker, ArgoCD and Kubernetes. It also uses caching in order not to compute multiple times the data necessary to generate the same graph. 

## Project Structure

This section provides a detailed overview of the project's directory and file structure, explaining the purpose and contents of each part. We also give a particular focus to the `src` folder.


```plaintext
.
├── Dockerfile                - Dockerfile for building the project's container image.
├── LICENSE                   - Contains the licensing information for the project.
├── README.md                 - The top-level guide to the project.
the project's capabilities and explaining the parameters.
├── deployment                - Holds Kubernetes deployment configurations.
│   ├── deployment.yaml       - Kubernetes deployment configuration.
│   ├── ingress.yaml          - Kubernetes ingress settings for external access.
│   └── service.yaml          - Kubernetes service definition.
├── frontend                  - Frontend application files.
│   ├── pages                 - Web pages for the frontend application.
│   └── static                - Static assets like CSS, JavaScript, and images.
├── requirements.txt          - Lists all Python library dependencies for the project.
├── src                       - Source code for the project's Python modules.
│   ├── Axes                  - Module for managing graphical axes properties.
│   ├── Clustering            - Algorithms and utilities for data clustering.
│   ├── GloVe                 - Implementation and utilities for the GloVe word embeddings and document embeddings.
│   ├── Polarization          - Tools for analyzing and visualizing corpus polarization.
│   ├── Processing            - General data processing utilities.
│   ├── S3_Bucket_utils       - Utilities for interacting with AWS S3 buckets.
│   ├── Word_analysis         - Tools for performing linguistic analysis on corpuses.
└── streamlit_app.py          - Streamlit application entry point.
```

<details>
<summary><strong>Detailed structure and scripts utility of the `src` source</strong></summary>

### `src/`

- The source code for the core functionality of the project.
  
  - [`src/processing/`](src/processing/): All the functions linked use to filter texts on the theme of BigTechs, and text cleaning functions. The `clean()` function is called many times in the project in order to clean uniformally newcoming texts.

  - [`src/GloVe/`](src/GloVe/): 
    - [`glove_functs.py`](src/GloVe/glove_functs.py) contains the main functions to perform the computation of the cooccurrence matric then the training of the GloVe model.
    - [`weights.py`](src/GloVe/weights.py) contains the functions to compute the weighting of embeddings inside a document in order to get the document general embedding.

  - [`src/Axes/`](src/Axes/): This folder builds all the functions relative to the definition of the axes we want to look at and the projection of embeddings on them.
    - [`axes_definition.py`](src/Axes/axes_definition.py) : Here you can find and modify the list of words defining the poles of the axes.
    - [`models.py`](src/Axes/models.py): This script loads the embeddings from text format into word2vec format, which is much more manipulable.
    - [`projection_functions.py`](src/Axes/projection_functions.py): This scripts contains all the functions performing the projection of texts on the axes. It also contains the important functions used to convert the embeddings files into word2vec objects, more easily manipulable.
    - [`curves_plots.py`](src/Axes/curves_plots.py): Builds the main function to plot the cosine between selected sources and axis, with multple parameters available.

  - [`src/Polarization/`](src/Polarization/):
    - [`polarization_functions.py`](src/Polarization/polarization_functions.py): This script contains the functions to compute the polarization of a corpus given two parties, following the method of Gentzkow and al.
    - [`polarization_plots.py`](src/Polarization/polarization_plots.py): This script defines the important function computing and plotting polarization values given parties and multiple variables, and storing the values and the plots in the `data` folder.
    - [`cos_pol.py`](src/Polarization/cos_pol.py): Contains the function plotting the polarization long with the cosine similarity when restricted to an axis.
    - [`word_partisanship.py`](src/Polarization/word_partisanship.py): Functions to compute the partizanship of words or bigrams.

  - [`src/S3_Bucket_utils`](src/S3_Bucket_utils):
    - [`read_data.py`](src/S3_Bucket_utils/read_data.py): Contains the function necessary for the interaction with the S3 bucket, and also the dataloader allowing caching. 

  - [`src/Word_analysis/`](src/Word_analysis/): This folder contains all the functions to proceed to the different linguistic analysis we built to explain the variations we observed on the different curves. 
    - [`words_variation.py`](src/Word_analysis/words_variation.py): Functions to look at the biggest variations for words in embedding between two years. 
    - [`axis_variation.py`](src/Word_analysis/axis_variation.py): Functions to look at the words in the poles which are the most responsible for the movement of the corpus towards their respective pole. 
    - [`cluster_words.py`](src/Word_analysis/cluster_words.py): main function to perform the spectral clustering of a selectd corpus, also using the functions in [`src/Clustering/`](src/Clustering/).



</details>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

All packages are listed in [the requirements file](requirements.txt). To install these packages, you can use the following command if you are using `pip`:

```bash
pip install -r requirements.txt
```
