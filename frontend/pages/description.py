import streamlit as st


def display_description():
    st.title("Description of the tools")

    st.markdown(
        """
        We provide a description of the different tools we developped, as well as a guide of the different parameters that are used in the tools. 

        #### `with_parliament`
        There are actually two GloVe models that have been trained. One with a corpus of texts from the House of Commons and the texts from th media, and another one only on the corpus from the media.
        Therefore, before using a tool, you must indicate if you want to take the model trained on the corpus with the parliament, or not. 
        - The `with_parliament` parameter indicates if you want to work with the model trained on the corpus with the speeches from the parliament (ending in 2019), or only the corpus of newspapers (from 2010 to 2023). The parameter is either `True` or `False`.

        #### Abreviations :
        - `GUA` for *The Guardian*, `TE` for *The Telegraph*, `DM` for *The Daily Mail*, `DE` for The *Daily Express*, `par` for the whole Parliament, `Lab` for only the Labour party, `Con` for only the Conservative party.
        - `am` for Amazon, `fb` for Facebook, `ap` for Apple, `go` for Google, `mi` for Microsoft.
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Curves analysis", "Words analysis"])

    with tab1:
        st.markdown(
            """
        
        <details>
        <summary><strong>Cosine similarity curves</strong></summary>
        <br>
                    
        #### `axis` 
        - The `axis` parameter designates the currentlty defined axes of study. For now, `axis=1` corresponds to a positive pole of laisser-faire, and a negative pole of regulation. `axis=2` is a simple positive and negative axis. 
        For example :
        
                    axis = 1

        #### `sources` 
        - The `sources` parameter is a list of the newpapers, or parties considered in the computations. The liste must contain a subset of `['par', 'Lab', 'Con', 'GUA', 'TE', 'DM', 'DE', 'MET']`. It's a list of the sources we decided to aggregate the texts by, and display on the graph. 
        For example :
                    
                    sources = ["GUA", "TE", "DM"]

        #### `focus_on_companies` 
        - The `focus_on_companies` parameter allows to consider the curves from the different defined sources but restricting their corpuses to the companies given. The parameter is a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. Leave it to `None` if you want to consider all the companies.
        For example :
                    
                    focus_on_companies = ['am', 'fb']

        #### `curves_by_company`
        - The `curves_by_company` method aggregates the entire corpus by the company mentionned in each text, then proceeds with these aggregates. This parameter is also a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. You will thus have one curve by company selected. Leave it to `None` if you want to consider all the companies. For example :
                    
                    curves_by_company = ['am', 'fb']

        </details>
                    
        <details>
        <summary><strong>Polarization curves</strong></summary>
        <br>
                    
        #### `left` party and `right` party
        - To compute polarization in a corpus you need to split it into two separate parties, a `left` party and a `right` party. Each of these parameters is a sublist of `['par', 'Lab', 'Con', 'GUA', 'TE', 'DM', 'DE', 'MET']`. They cannot overlap. For example :
                    
                left = ["Lab"]
            and 
                    
                right = ["Con"]
                    
        #### `axis` 
        - The `axis` parameter allows to compute the polarization while considering texts close to the poles of the axis selected. This parameter goes hand in hand with the `percentiles` parameter. Given these parameters, we only keep the texts with a cosine with the `axis` outside of the quantiles given by the `percentiles`. Doing so, we only compute polarization on the texts relevant to the subjects defined by the axis. We leave them both to `None` if we don't want to consider this method. 
        For example :
        
                    axis = 1

            and       
                    
                    percentiles = [10, 90]


        #### `curves_by_company`
        - The `curves_by_company` method aggregates the entire corpus by the company mentionned in each text, then proceeds with these aggregates. This parameter is also a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. You will thus have one curve by company selected. Leave it to `None` if you want to consider all the companies. For example :
                    
                    curves_by_company = ['am', 'fb']

        #### `print_random_pol`
        -The `print_random_pol` parameter is a boolean indicating if you want to plot the polarization computed on random labels. (It should be relatively close to 0.5). It is set to `True` by default.
                    
                    print_random_pol = True

        </details>
        
        """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            """
        
        <details>
        <summary><strong>Words variations</strong></summary>
        <br>
                    
        This tool allows to visualize, given different parameters, the words with the biggest embedding variations, when projected on an axis, between two years. It gives a first example of the terms driving the cosine similarity toward a pole of this axis.
                    
    #### `year` 
        - The year considered. For example :
                    
                    year = 2016

        #### Axes
        - The axes considered for each of the subplots. For example :
                    
                    first_axis = 1

            and 
                    
                    second_axis = 1

        #### Variations
        - For each subplot, towards which pole we are looking at the variations. Either positive or negative. 
                
        #### Number
        - The number of words you want to see :
                    
                    number = 20

        </details>
                    
        <details>
        <summary><strong>Axis variations</strong></summary>
        <br>
                    
        This tool allows to visualize, given different parameters, the words defining the axes poles that are the most responsible for attracting the corpus towards their respective pole. It gives another example of the terms driving the cosine similarity toward a pole of this axis.
                    
        #### `year` 
        - The year considered. For example :
                    
                    year = 2016

        #### `source`
        - The `source` parameter allows us to restrict our corpus to a specific source. The `source` must be one of `['par', 'GUA', 'TE', 'DM', 'DE', 'MET']`, or leave it to `None` if you want to consider all sources. For example :
                    
                    source = "GUA"

        #### Number
        - The number of words you want to see :
                    
                    number = 20

        </details>
                    
        <details>
        <summary><strong>Spectral clustering</strong></summary>
        <br>
                    
        This tool first selects texts positionned close to the axes poles and then performs spectral clustering on the words embeddings. This allows to build clusters of similar notions, and try to decipher explanations for the variations of the curves. 
                    
        #### `year` 
        - The year considered. For example :
                    
                    year = 2016

        #### `axis`
        - The `axis` considered :
                    
                    axis = 1
                        
        #### 1 - Percentiles method
        - We can select the relevant texts on this axis by taking texts with cosine values with the axis under and above two percentiles. Default is [10, 90] :
                    
                    percentiles = [10, 90]

        #### 2 - Thresholding method 
        - We can select the relevant texts on this axis by taking texts with cosine values with the axis under and above two thresholds. Default is [-0.3, 0.3]. For example :
                    
                    left_threshold = -0.3

            and 
                    

                    right_threshold = 0.3

        #### 3 - Head-tail method 
        - We can select the relevant texts on this axis by taking texts with the n highest and lowest cosine values with the axis. Default is 100 highest values and 100 lowest values. For example :
                    
                    head = 100

            and 
                    tail = 200

        #### **⚠️ For now, these 3 methods cannot be combined !**
                    
        #### `company`
        - If you want to focus on a company. The parameter is a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. Leave it to `None` if you want to consider all the companies. For example :
                    
                    company = 'am'

        </details>
        
        """,
            unsafe_allow_html=True,
        )
