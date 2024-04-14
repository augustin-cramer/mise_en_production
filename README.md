# Mise en production

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
<summary><strong>webscraping</strong></summary>
<br>

### `webscraping/`

- Scripts designed to extract data from the web, facilitating the data collection process for analysis. They can be viewed as an inspiration for future webscrapping, as they will be very hard to use again. 

</details>
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
    - [`models_dataframes.py`](src/Axes/models_dataframes.py): This scripts takes the word2vec embeddings format and computes the cosine of each document with the 2 axes defined. It also filters the texts speaking of specific companies thanks to the filtering words defined in [`filter_words.py`](src/Axes/filter_words.py). Then, it stores two dataframes in the `data` folder : a dataframe with all the texts and their cosines with the axes, and a similar one but with information relative to the company of which each text speaks of. The computations and bootstrapping are done with functions in [`projection_functions.py`](src/Axes/projection_functions.py) and [`bootstraping.py`](src/Axes/bootstraping.py).
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


</details>
<details>
<summary><strong>processing</strong></summary>

### `processing/`

This folder contains the three notebooks that we use to clean and filter our corpus, and also get the entire vocabulary of our corpus. 

</details>
<details>
<summary><strong>notebooks</strong></summary>

### `notebooks/`

This folder is where we can visualize all our results and do our manipulations. 

- [`glove model/`](notebooks/glove%20model/):
  - [`__main__.ipynb`](notebooks/glove%20model/__main__.ipynb): The notebook is where we can launch the computation of the cooccurence matrix, the training of the GloVe model and the formation of document embeddings. 

- [`define axes/`](notebooks/define%20axes/):
  - [`axes_definition.ipynb`](notebooks/define%20axes/axes_definition.ipynb): This notebook launches the definition of axes, the computation of cosines between the corpus and the axes and the filtering with respect to the BigTechs. 

- [`cosine similarity curves/`](notebooks/cosine%20similarity%20curves/):
  - [`curves.ipynb`](notebooks/cosine%20similarity%20curves/curves.ipynb): The main notebook where we can visualize the evolution of cosine similarity between a corpus and an axis, given multiple variables. 

- [`polarization/`](notebooks/polarization/):
  - [`curves.ipynb`](notebooks/polarization/curves.ipynb): The main notebook to visualize the evolution of the polarization between two sources, given multiple variables, and also compared to the evolution of the cosine similarity on an axis during the same period. 
  - [`polarized_words.ipyn`](notebooks/polarization/polarized_words.ipynb): The notebook to vizualize the most partisan words every year. 

- [`word analysis/`](notebooks/word%20analysis/): Here you can perform all the linguistic analysis built in [`src/Word_analysis/`](src/Word_analysis/) in order to explain the variation on the preceeding curves. 

</details>
<details>
<summary><strong>plots</strong></summary>

### `plots/`

- This directory houses all graphical outputs generated by the project.

</details>
<details>
<summary><strong>data</strong></summary>

### `data/`

- This directory houses all the inputs used by the project.

</details>
<details>
<summary><strong>main notebook</strong></summary>

### [`main.ipynb`](main.ipynb)

The main notebook to use in order to easily access all the different analysis at the place, and play with the different parameters. This is where all the parameters of the main functions are explained. 

</details>

## How to proceed to a new GloVe training ? 

- Start from your dataframes.
- Clean them, filter them and convert them to vocabulary with the notebooks in [`processing`](processing).
- Compute the cooccurence matrix, train your model and form document embeddings in [`glove model/`](notebooks/glove%20model/). 