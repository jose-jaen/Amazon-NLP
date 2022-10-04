# Required packages
import gzip
import string
import pandas as pd
from langid.langid import LanguageIdentifier, model

def parse(path):
    """ Opens gzip file for specified path

    - Parameters:
        - path = Path where user has gzip file

    - Output: gzip file with data
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    """ Creates Pandas dataframe from gzip file

    - Parameters:
        - path = Path where user has gzip file

    - Output:
        - Pandas dataframe with data
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def remove_punctuation(text):
    """ Removes punctuation from reviews

    - Parameters:
        - text = Data column with stored comments

    - Output: Cleaned text
    """
    try:
        text = text.translate(None, string.punctuation)
    except:
        translator = text.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    return text

def languages(comments):
    """ Identifies language in reviews

    - Parameters:
        - comments = Reviews array to be analyzed

    - Output:
        - languages_list = List with languages
    """
    languages_list = []
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    for i in range(len(comments)):
        language = identifier.classify(comments[i])[0]
        languages_list.append(language)
    return languages_list

def english_identifier(languages):
    """ Identifies if there are non-english comments

    - Parameters:
        - languages = List with languages

    - Output: Non-english comments or confirming message
    """
    non_english = []
    for i in languages:
        if i != 'en':
            lang = i
            non_english.append(lang)
            return non_english
        else:
            return 'Every comment was made in English'