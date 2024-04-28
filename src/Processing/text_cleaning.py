import pandas as pd
import string
import nltk

nltk.download("stopwords")
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords

stemmer = SnowballStemmer(language="english")
english_stopwords = set(stopwords.words("english"))

########################
# Parliament stopwords #
########################

simple_britain_stopwords = [
    "since ",
    "gentlelady",
    "house lord",
    "supplementary",
    "questioned",
    " united kingdom",
    "please",
    "section",
    "prime minister",
    "questions",
    "ladies",
    "hon memb",
    "parliament",
    "like ",
    "hope wil",
    "iii",
    "years ago",
    "Answers To Questions",
    " home secretari",
    "vote",
    "member ",
    "republican",
    "mps",
    "friend",
    "congresswoman",
    " secretary state",
    "Speaker",
    "friend memb",
    "chair",
    "senator",
    "congresswomen",
    "Commons",
    " member",
    "secretary st",
    "billion",
    "Council",
    "right hon",
    " will",
    "opposition",
    "motions",
    "asked",
    "opposite",
    "chairman",
    "uk",
    "congressman",
    "parties",
    "asks",
    "governments",
    "leader",
    "ii",
    "exminister",
    "government",
    "country",
    "senate",
    "yield",
    "pursuant",
    "sections",
    "year",
    "minister",
    "hon.",
    "prime",
    "Gentleman",
    "floor",
    "yes",
    "years",
    "order",
    "bill",
    "gentleman",
    " think",
    "hon friend",
    "local author",
    "congressmen",
    "s",
    "i",
    "deputy",
    "last year",
    "mr",
    "ladi",
    "mp",
    "colleagues",
    "congress",
    " ladi",
    "madam",
    "memb",
    " group",
    "committee",
    "british",
    "£ million",
    "prime minist",
    "supply",
    "united kingdom",
    "votes",
    "colleague",
    " look",
    "per cent",
    "Minister",
    "chairmanship",
    "secretary",
    "clause",
    "state",
    "hon",
    "secretary state",
    "ministers",
    "party",
    "britain",
    "demo- crat",
    "speaker",
    " per cent",
    "£ billion",
    "members",
    "motion",
    "House",
    "question",
    "million",
    "ask",
    "bills",
    " house lord",
    "lady",
    "member",
    "sir",
    "amendment",
    " although",
    "gentlemen",
    "Oral",
    "hon. Gentleman",
    "house",
]


def read_input(path, encod, **kwargs):
    dtype_values = kwargs.get("dtype_values", None)
    df = pd.read_csv(path, sep=";", encoding=encod, dtype=dtype_values)
    return df


dtypes = {
    "party.facts.id": str,
    "date": object,
    "agenda": object,
    "speechnumber": int,
    "speaker": object,
    "party": object,
    "party.facts.id": object,
    "chair": bool,
    "terms": int,
    "text": object,
}


def read_HouseOfCommons(keep_date: bool, rd_lines: bool, size: int):
    """
    Read the parlementary database, and returns the dataframe preprocessed

    Parameters:
    -----------
    keep_date: determines if we delete the keep_date column
    rd_lines : if we want to keep a number of random lines
    size : number of random lines we want to keep
    """
    df = read_input(
        "data/raw_corpuses/Corp_HouseOfCommons_V2_2010.csv",
        encod="ISO-8859-1",
        dtype_values=dtypes,
    )
    if keep_date:
        df.drop(
            columns=[
                "Unnamed: 0",
                "iso3country",
                "parliament",
                "party.facts.id",
                "speechnumber",
                "chair",
                "terms",
            ],
            inplace=True,
        )
    else:
        df.drop(
            columns=[
                "Unnamed: 0",
                "iso3country",
                "parliament",
                "party.facts.id",
                "speechnumber",
                "chair",
                "terms",
                "date",
            ],
            inplace=True,
        )
    df.rename(columns={"speaker": "Speaker"}, inplace=True)
    if rd_lines:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(size)
        return df
    return df


"""df_par = read_HouseOfCommons(keep_date=True, rd_lines=False, size=100)
speaker_names = list(df_par['Speaker'].drop_duplicates())
speakers = [i.split()for i in speaker_names if len(str(i))>3 and len(i.split())>1 ][1:]
speakers = [i[1] for i in speakers]"""

stemmer = PorterStemmer()

english_stopwords_stem = [stemmer.stem(word.lower()) for word in english_stopwords]
simple_britain_stopwords_stem = [
    stemmer.stem(word.lower()) for word in simple_britain_stopwords
]

"""speakers_stem = [stemmer.stem(word.lower()) for word in speakers]"""

stopwords_stem = [word for word in english_stopwords_stem] + [
    word for word in simple_britain_stopwords_stem
]  # + [word for word in speakers_stem]

#################
# Text cleaning #
#################


def extract_bigrams(n_grams: str):
    """extracts the list of bigrams from a list of words

    Parameters:
    -----------
    ngrams : the text
    """
    bigrams = []
    for i in range(len(n_grams) - 1):
        bigram = f"{n_grams[i]} {n_grams[i+1]}"
        bigrams.append(bigram)
    return bigrams


def clean(text: str, gram: str):
    """
    This function does the main cleaning of the text, from characters removal to stemming and stopwords removal

    Parameters:
    -----------
    text : the text to clean
    """
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.translate(str.maketrans("", "", string.digits))
    # tokenization
    tokens = word_tokenize(text)
    # Enlever les caractères qui ne sont pas des lettres
    tokens = [re.sub("[^a-zA-Z]", "", token) for token in tokens]
    # Stemming
    tokens_stemmed = [stemmer.stem(token) for token in tokens]
    filtered_words = [
        word
        for word in tokens_stemmed
        if not word.lower() in stopwords_stem and len(word) > 3
    ]
    if gram == "bigram":
        filtered_words = extract_bigrams(filtered_words)
    return filtered_words
