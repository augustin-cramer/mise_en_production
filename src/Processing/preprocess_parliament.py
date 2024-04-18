import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from Processing.text_cleaning import *
stemmer = SnowballStemmer(language='english')
english_stopwords = set(stopwords.words('english'))



###################################
###fonctions to read speakers df### 
###################################

def read_input(path, encod, **kwargs):
    dtype_values = kwargs.get('dtype_values', None)
    df = pd.read_csv(path, sep=';', encoding=encod, dtype=dtype_values)
    return df

dtypes = {
    'party.facts.id' : str,
    'date': object,
    'agenda': object,
    'speechnumber': int,
    'speaker': object,
    'party': object,
    'party.facts.id': object,
    'chair': bool,
    'terms': int,
    'text': object,
}

def read_HouseOfCommons(keep_date: bool, rd_lines: bool, size:int):
    '''
    Read the parlementary database, and returns the dataframe preprocessed

    Parameters:
    -----------
    keep_date: determines if we delete the keep_date column
    rd_lines : if we want to keep a number of random lines
    size : number of random lines we want to keep
    '''
    df = read_input('data/raw_corpuses/Corp_HouseOfCommons_V2_2010.csv', encod='ISO-8859-1', dtype_values=dtypes)
    if keep_date:
        df.drop(columns=['Unnamed: 0', 'iso3country', 'parliament', 'party.facts.id', 'speechnumber', 'chair', 'terms'], inplace=True)
    else:
        df.drop(columns=['Unnamed: 0', 'iso3country', 'parliament', 'party.facts.id', 'speechnumber', 'chair', 'terms', 'date'], inplace=True)
    df.rename(columns=
        {'speaker': 'Speaker'},
        inplace=True
    )
    if rd_lines:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(size)
        return df
    return df 

#################################
# Topics linked to the Bigtechs #
#################################

technology=['technology','innovation','computer','high tech|high-tech','science','engineering']

consumer_protection=['privacy','data leak','leak','fake news',' safety','decept','defective','hack']

firms=['google','alphabet','apple','facebook','meta','amazon','microsoft']

products=['chrome','incognito','youtube','nexus','pixel','google drive','gmail','glass','street view','buzz','fitbit',
'maps', 'doodle','play','translate','search', 'google news','nest hub','xl','nest','chromecast','stadia','hub',
'marshmallow','lollipop','cloud','waymo','earth','engine',

'apple pay','apple watch','iphone','ipad','ipod','iwatch','macbook','macbook pro', 'macbook air','mac',
'imac','airpods','ios','siri','icloud','apple tv','apple music','app store', 'safari','x','app','apple store',
'xr', 'xs', 'se','iphones','itunes','ibook','plus','pro','max','mini','os','airtag','airtags','arcade','homepod',
'keynote','ipados','id','foxconn','facetime','beat','stalk',

'messenger','instagram', 'whatsapp','page','feed','oculus',

'prime', 'kindle',
'publishing','amazon prime','amazon drive','amazon video','amazon business','amazon web service',
'amazon cloud', 'alexa','echo dot','echo','dot', 'delivery', 'amazon uk','unlimited', 'episode','foods','grocery', 
'grand tour','grand','tour','viking', 'vikings','argo','argos','macmillan', 'dvd','clarkson','lord','ring','hair','skin','vacuum',
'pre','beer','drake','spark','kart','dog','twitch','cat','xo','matthew','stafford','ratchet','clank',
'swagway','album','mouse','showbiz','beauty','guardian','batman','arkham','gc','hair','skin','shirt',
'lovefilm','mirzapur','cast','audio','drama','movie','jack ryan','actor','character','lucifer','outlander',
'premier','super mario','sky','channel','voyage',


'windows','window','xp','surface','xbox','studio','microsoft office', 'office','word','cortana', 'surface pro','teams',  'playstation',
'microsoft edge', 'edge', 'gear','outlook','halo','skype','kinect','internet explorer','explorer','ie','bing','xcloud','hololens',
'forza','ori','scarlett','scorpio','wordperfect','valhalla','onedrive','games gold','lumia','azure',
'assassin creed','assassin','creed','minecraft','yammer','warcraft','tay']



ceos=['sundar','pichai','eric','schmidt','steve jobs','tim cook','mark zuckerberg','andy jassy','jeff','bezos','satya', 'nadella','bill gates',
'gates','steve job','steve','tim', 'cook','zuckerberg','ceo','tim cook ','steve ballmer','ballmer','elop',
'schiller','fadell','phil spencer','spencer','mcspirit','sandberg','paul','allen','larry hryb','hryb']


types=['tablet','mobile', 'laptop', 'pc', 'computer', 'desktop','smartphone', 'smartwatch', 'search engine', 'software','hardware',
'machine', 'browser','ebook', 'book',  'reader',  'console', 'headphone', 'earbud','bud','store','music',
'gaming', 'operating','streaming','title','chatbot']



competitor=['samsung', 'galaxy',  'twitter','tiktok', 'switch','sony', 'asos', 'activision blizzard', 'activision','blizzard',
'nintendo','snes', 'netflix','android','yahoo','nokia','huawei','motorola','htc','blackberry','oppo','oneplus','rim','symbian','bbc','morrison','spotify'] 


configue=['device','feature','battery','screen','sound','gb','g','k','mm','chip','processor','design','display','touch','ram',
'inch','keyboard','camera','handset','speaker','button','touchscreen','storage', 'data']


celebrity=['dubost','neymar','amanda','beyonce','blur','richard','hammond','ranj','jeremy clarkson', 'jeremy','momoa',
'jared','aniston','smith','kim','tony','tom','sophie','oasis','trio','sharon','betty','raoul','moat','lauren','andrew',
'samuel gibbs','samuel','gibbs','van','gaal']


# Cela nous donne les topics : 

topics = celebrity + configue + competitor + types + ceos + products + firms + consumer_protection + technology

def process_list_BigTech_words(topics: list):
    '''
  Output is a clean list of BigTech topics

  Parameters:
  -----------
  topics : list of topics
  '''
    list_stem_topics = []
    for word in topics:
        list_stem_topics.append(stemmer.stem(word))
    return list_stem_topics


def lines_to_keep(text: str, liste_big_tech: list):
    '''
  Takes the text of the speech as an input, and returns a bool saying if the speech is within the BigTech topic

  Parameters:
  -----------
  text : text of the speech
  liste_big_tech : words in link with BigTechs
  '''
    try :
        if len(set(clean(text, 'unigram')) & set(liste_big_tech)) > 0:
            return True
    except :
        return False
    return False


def keep_Bigtech_speeches(df: pd.DataFrame, list_stem_topics: list):
  '''
  Takes a dataframe of speeches as an input, and returns only the speeches that have a link with the topics

  Parameters:
  -----------
  df : Dataframe of inputs
  list_stem_topics : list of BigTech words
  '''
  set_stem_topics = set(list_stem_topics)
  df['lines_to_keep'] = df['text'].apply(lines_to_keep, args=(set_stem_topics,))
  df = df.loc[df['lines_to_keep']]
  df.drop(columns=['agenda', 'lines_to_keep'], inplace=True)
  return df 