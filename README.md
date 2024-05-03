# Mise en production

This project explores techniques described in recent scholarly papers and adapts them to analyze public opinion about BigTech companies in the UK. Our aim is to develop robust analysis tools tailored to this specific context. Specifically, the tools of the project are inspired by :

- [Gentzkow, Matthew, Jesse M. Shapiro, and Matt Taddy](https://scholar.harvard.edu/shapiro/publications/measuring-group-differences-high-dimensional-choices-method-and-application)
- [Dorottya Demszky, Nikhil Garg, Rob Voigt, James Zou, Matthew Gentzkow, Jesse Shapiro, Dan Jurafsky](https://arxiv.org/abs/1904.01596)
- [Austin C. Kozlowski, Matt Taddy, James A. Evans](https://arxiv.org/abs/1803.09288)

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
    


## Data

For now, the data can be found in https://drive.google.com/drive/folders/1OG0NaPqlbzNlvG83L0LQMMVsw8jftEsz
Download the zipfile and create a `data` folder with its elements. 
Then add this folder and the `plots` folder to the project folder.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

All packages are listed in [the requirements file](requirements.txt). To install these packages, you can use the following command if you are using `pip`:

```bash
pip install -r requirements.txt
```

## Project Structure

This section provides a detailed overview of the project's directory and file structure, explaining the purpose and contents of each part.

<details>
<summary><strong>src</strong></summary>

### `src/`

- The source code for the core functionality of the project.
  
  - [`src/Processing/`](src/Processing/): All the functions linked use to filter texts on the theme of BigTechs, and text cleaning functions. The `clean()` function is called many times in the project in order to clean uniformally newcoming texts.

  - [`src/GloVe/`](src/GloVe/): 
    - [`glove_functs.py`](src/GloVe/glove_functs.py) contains the main functions to perform the computation of the cooccurrence matric then the training of the GloVe model.
    - [`weights.py`](src/GloVe/weights.py) contains the functions to compute the weighting of embeddings inside a document in order to get the document general embedding.

  - [`src/Axes/`](src/Axes/): This folder builds all the functions relative to the definition of the axes we want to look at and the projection of embeddings on them.
    - [`axes_definition.py`](src/Axes/axes_definition.py) : Here you can find and modify the list of words defining the poles of the axes.
    - [`models.py`](src/Axes/models.py): This script loads the embeddings from text format into word2vec format, which is much more manipulable.
    - [`models_dataframes.py`](src/Axes/models_dataframes.py): This scripts takes the word2vec embeddings format and computes the cosine of each document with the 2 axes defined. It also filters the texts speaking of specific companies thanks to the filtering words defined in [`filter_words.py`](src/Axes/filter_words.py). Then, it stores two dataframes in the `data` folder : a dataframe with all the texts and their cosines with the axes, and a similar one but with information relative to the company of which each text speaks of.The computations and bootstrapping are done with functions in [`projection_functions.py`](src/Axes/projection_functions.py) and [`bootstraping.py`](src/Axes/bootstraping.py).
    - [`curves_plots.py`](src/Axes/curves_plots.py): Builds the main function to plot the cosine between selected sources and axis, with multple parameters available.

  - [`src/Polarization/`](src/Polarization/):
    - [`polarization_functions.py`](src/Polarization/polarization_functions.py): This script contains the functions to compute the polarization of a corpus given two parties, following the method of Gentzkow and al.
    - [`polarization_plots.py`](src/Polarization/polarization_plots.py): This script defines the important function computing and plotting polarization values given parties and multiple variables, and storing the values and the plots in the `data` folder.
    - [`cos_pol.py`](src/Polarization/cos_pol.py): Contains the function plotting the polarization long with the cosine similarity when restricted to an axis.
    - [`word_partisanship.py`](src/Polarization/word_partisanship.py): Functions to compute the partizanship of words or bigrams.

  - [`src/Word_analysis/`](src/Word_analysis/): This folder contains all the functions to proceed to the different linguistic analysis we built to explain the variations we observed on the different curves. 
    - [`words_variation.py`](src/Word_analysis/words_variation.py): Functions to look at the biggest variations for words in embedding between two years. 
    - [`axis_variation.py`](src/Word_analysis/axis_variation.py): Functions to look at the words in the poles which are the most responsible for the movement of the corpus towards their respective pole. 
    - [`cluster_words.py`](src/Word_analysis/cluster_words.py): main function to perform the spectral clustering of a selectd corpus, also using the functions in [`src/Clustering/`](src/Clustering/).

  - [`src/S3_Bucket_utils`](src/S3_Bucket_utils):
    - [`read_data.py`](src/S3_Bucket_utils/read_data.py): 


</details>

<details>
<summary><strong>main notebook</strong></summary>

### [`main.ipynb`](main.ipynb)

The main notebook to use in order to easily access all the different analysis at the place, and play with the different parameters. This is where all the parameters of the main functions are explained. 

</details>
