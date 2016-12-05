## imports
from bs4 import BeautifulSoup
import requests
import pandas as pd
# import jellyfish
# import lxml
# import html5lib
import json
import os
import string
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import difflib
import numpy as np
from pprint import pprint


"""
@todo:
- Join Character table with the big table (script table)
- Check how to join planet with dialogue tables.
- Download https://en.wikipedia.org/wiki/List_of_Star_Trek:_The_Next_Generation_episodes
- Merge the episode table with the script df
"""

# NOTE: We decided to have more of a scripting approach to this project, it being more exploratory in nature.

## POSSIBLE QUESTIONS TO ANSWER ABOUT THE STAR TREK TNG UNIVERSE #
# 1: % of screen time over each episode by [character, empire (UFD, Romulan, Borg), species, gender, planet]
# 2: Count of weapons fire over each episode by type (photon torpedo, phaser, etc.)
# 3: Count of off-ship excursions over time (beam outs/beam ins, shuttle craft activity)
# 4: Topic modeling. What is generally discussed in TNG? Do these topics change over time? What is topic sentiment?
# 5: Sentiment analysis. How positive/neutral/negative is the show over each episode?
# 6: Count of how many times Picard says "Make it so" or "engage"
# 7: How fast does the ship travel over time? What is the specified warp speed over each episode?
# 8: Which planets does the ship travel to? Can we put this on a galactic map?

## scripts from "Star Trek: The Next Generation" were downloaded from the following link:
# www.st-minutiae.com/resources/scripts/

## download any other relevant data, such as the planet table
# print('Fetching planetary table...')
# # use requests library to get the main page content
# r = requests.get('http://www.startrekmap.com/library/maplist.html').text
# # use BS4 to locate the div with class
# r_soup = BeautifulSoup(r, 'html.parser')
# planet_table = str((r_soup.find_all('table'))[0])
# planet_table_df_list = pd.read_html(planet_table)
# planet_table_df = pd.DataFrame(planet_table_df_list[5])
# planet_table_df = planet_table_df.rename(index=str, columns={0: "Designation",
#                                                              1: "Quadrant",
#                                                              2: "Affiliation",
#                                                              3: "Status"})
# # convert beta character to "b"
# planet_table_df['Quadrant'] = planet_table_df['Quadrant'].str.replace('ß', 'b')
# # write to csv
# planet_table_df.to_csv('../data/planets_df.csv', sep=',')
# print('Retrieved planetary table:')
# print(planet_table_df.head(10))

# --------

## download the table of characters, which provides species, actor, rank, posting, and position information
# print('Fetching character table...')
# # use requests library to get the main page content
# r = requests.get('https://en.wikipedia.org/wiki/List_of_Star_Trek_characters').text
# # use BS4 to locate the div with class wikitable
# r_soup = BeautifulSoup(r, 'html.parser')
# wiki_tables = r_soup.find_all('table', {"class": "wikitable"})
# # the first table is a list of star trek series, so the second table is the table we want to work with
# # print(wiki_tables[1])
#
# # create the dataframe we will add our data to
# character_df = pd.DataFrame(columns=[
#     'character_name',
#     'actor_name',
#     'rank',
#     'posting',
#     'position',
#     'species',
#     ]
# )
#
#
# for entry in tqdm(str(wiki_tables[1]).split('<tr>')):
#     character = str(BeautifulSoup(entry, 'html.parser')).split('<td>')
#     # characters with lengths of 4 represent secondary actors for a given role
#     # for example, Spock is played by Leonard Nimoy in the original series and TNG, but is played by
#     # Zachary Quinto in the J.J. Abrahams reboot films.
#     # handle this by having an list of actors in the data frame
#
#     if len(character) == 8:
#         try:
#             character_name = re.findall('<a.*?>(.+?)</a>', character[1])[0]
#         except IndexError:
#             character_name = None
#         try:
#             actor_name = re.findall('<a.*?>(.+?)</a>', character[2])[0]
#         except IndexError:
#             actor_name = None
#         try:
#             rank = [re.sub('<[^<]+?>', '', re.findall('<a.*?>(.+?)</a>', character[4])[0])]
#         except IndexError:
#             rank = None
#         appearances = re.findall('<i.*?>(.+?)</i>', character[3])
#
#         try:
#             posting = re.sub('<[^<]+?>', '', re.findall('<a.*?>(.+?)</a>', character[5])[0])
#         except IndexError:
#             posting = None
#         # posting = re.findall('<a.*?>(.+?)</a>', character[5])
#         try:
#             if 'href' not in re.findall('(.+?)</td>', character[6])[0]:
#                 position = re.sub('<[^<]+?>','', re.findall('(.+?)</td>', character[6])[0])
#             else:
#                 position = re.findall('<a.*?>(.+?)</a>', character[6])[0]
#         except IndexError:
#             position = None
#         # need to clean up position
#         species = re.findall('(.+?)</td>', character[7])
#         if 'href' not in re.findall('(.+?)</td>', character[7])[0]:
#             species = re.findall('(.+?)</td>', character[7])[0]
#         else:
#             species = re.findall('<a.*?>(.+?)</a>', character[7])[0]
#
#         temp_df = pd.DataFrame(
#                 [[character_name, actor_name, rank, appearances, posting, position, species]],
#                 columns=[
#                     'character_name',
#                     'actor_name',
#                     'rank',
#                     'appearances',
#                     'posting',
#                     'position',
#                     'species',
#                 ]
#             )
#         character_df = pd.concat([character_df, temp_df], ignore_index=True)
#
#     # NOTE: does not support multi actor characters right now, which excludes Spock and Montgomery Scott
#     # if len(character) == 2:
#     #     print(character)
#
# # examine the dataframe
# print(character_df.head(20))
#
# # write the df to a csv
# character_df.to_csv('../data/character_df.csv', sep=',')
#



## download the table of episode names and related information
# print('Fetching episode table...')
# # use requests library to get the main page content
# r = requests.get('https://en.wikipedia.org/wiki/List_of_Star_Trek:_The_Next_Generation_episodes').text
# # use BS4 to locate the div with class wikitable
# r_soup = BeautifulSoup(r, 'html.parser')
# wiki_tables = r_soup.find_all('table', {"class": "wikitable"})
# num_seasons = 7
# episode_df = pd.DataFrame()
# for i in range(1, num_seasons + 1):
#     print(i)
#     season = pd.read_html(str(wiki_tables[i]))
#     season_df = pd.DataFrame(season[0])
#     season_df.columns = season_df.iloc[0]
#     season_df = season_df.ix[1:]
#     episode_df = pd.concat([episode_df, season_df])
#
# episode_df = pd.DataFrame(episode_df)
# episode_df['merge_index'] = range(1, len(episode_df.index) + 1)
# print(episode_df.head(10))
# print(episode_df.tail(10))
# print(episode_df.columns)
# print(len(episode_df.index))
#
# # write the table to a csv
# episode_df.to_csv('../data/episode_df.csv', sep=',')


# # load any nltk tools we will use
# porter_stemmer = PorterStemmer()
# wordnet_lemmatizer = WordNetLemmatizer()
# sid = SentimentIntensityAnalyzer()
#
# # create the dataframe we will add our text data to
# script_df = pd.DataFrame(columns=[
#     'sentence',
#     'focus',
#     'episode',
#     'filename',
#     'tokens',
#     'stemmed_tokens',
#     'norm_tokens',
#     'token_count',
#     'sentiment'
#     ]
# )
#
#
# episode_counter = 1
# for file in os.listdir("../data/scripts"):
#     # if the file is a text file
#     if file.endswith(".txt"):
#         # open it and do all the things!
#         print('Processing episode # ' + str(episode_counter))
#         with open('../data/scripts/' + file, mode='r', encoding='utf-8', errors='ignore') as f:
#             # read the content and replace new lines and tabs
#             content = f.read().replace('\n', '').replace('\t', ' ').replace('-', '').replace('--', '')
#             # here we define a focus (i.e. a person, scene, or other current focus point
#             focus = "TNG"
#             # define a base index
#             b_index = 0
#             # for each entry in the content
#             for entry in tqdm(re.split('(?<=[.!?]) +', content)):
#
#                 # split by space and remove any blank entries, as well as single and double dashes
#                 split_entry = list(filter(None, entry.split(' ')))
#
#                 # if the first character is uppercase, it signifies a character or scene change
#                 change = split_entry[0].isupper() and split_entry[0].isalpha()
#
#                 # if there is a scene of character change
#                 if change is True:
#                     # extract the change token and remove stray "I" characters
#                     change_token = ' '.join(token for token in split_entry if token.isupper() and token is not "I")
#                     # set the current focus to this token
#                     focus = change_token
#
#                 # compile the original sentence from the list of tokens
#                 sentence = [' '.join(token for token in split_entry if not token.isupper())]
#
#                 # remove two and three digit that are indicators of scene changes
#                 # we lose the full date, but retain publish month and year, and stardates
#                 sentence = re.sub('([^\d]|^)\d{2,3}([^\d]|$)', ' ', sentence[0])
#
#                 # remove any and all punctuation and obtain a list of tokens and retain tokens that are not stopwords
#                 translator = sentence.maketrans({key: None for key in string.punctuation})
#                 tokens = [token for token in sentence.translate(translator).split() if token not in stopwords.words('english')]
#
#                 # use nltk to stem words using the Porter Stemmer
#                 stemmed_tokens = []
#                 [stemmed_tokens.append(porter_stemmer.stem(token)) for token in tokens]
#
#                 # NOTE: lemmatize takes a part of speech parameter, "pos." If not supplied, the default is "noun."
#                 # NOTE: it may be beneficial to do POS tagging with nltk prior to lemmatization to avoid this
#                 # SEE: http://textminingonline.com/dive-into-nltk-part-iii-part-of-speech-tagging-and-pos-tagger
#
#                 # use nltk to normalize the stemmed tokens
#                 norm_tokens = []
#                 [norm_tokens.append(wordnet_lemmatizer.lemmatize(token)) for token in stemmed_tokens]
#                 '''WordNet® is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets
#                 of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of
#                 conceptual-semantic and lexical relations. The resulting network of meaningfully related words and concepts
#                 can be navigated with the browser. WordNet is also freely and publicly available for download. WordNet’s
#                 structure makes it a useful tool for computational linguistics and natural language processing.
#                 -----
#                 WordNet superficially resembles a thesaurus, in that it groups words together based on their meanings.
#                 However, there are some important distinctions. First, WordNet interlinks not just word forms—strings of
#                 letters—but specific senses of words. As a result, words that are found in close proximity to one another in
#                 the network are semantically disambiguated. Second, WordNet labels the semantic relations among words, whereas
#                 the groupings of words in a thesaurus does not follow any explicit pattern other than meaning similarity'''
#
#                 # get the token count
#                 token_count = len(tokens)
#
#                 # score the sentiment of each sentence using the VADER sentiment analysis tool in nltk on the normalized tokens
#                 ss = sid.polarity_scores(' '.join(token for token in norm_tokens))
#                 '''VADER Sentiment Analysis. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and
#                 rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and
#                 generally works well on text from other domains.
#                 ------
#                 Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social
#                 Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
#                 ------
#                 Link to Paper: http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf
#                 '''
#
#                 # add the following terms as a row to the dataframe
#                 # sentence, focus, episode, filename, tokens, stemmed tokens, norm_tokens, token_count, sentiment
#                 temp_df = pd.DataFrame(
#                     [[sentence, focus, episode_counter, file, tokens, stemmed_tokens, norm_tokens, token_count, ss['compound']]],
#                     columns=[
#                         'sentence',
#                         'focus',
#                         'episode',
#                         'filename',
#                         'tokens',
#                         'stemmed_tokens',
#                         'norm_tokens',
#                         'token_count',
#                         'sentiment'
#                     ]
#                 )
#                 script_df = pd.concat([script_df, temp_df], ignore_index=True)
#     episode_counter += 1
#     print('\n')
#
#
# # examine the df
# print(script_df.head(20))
#
# # write the master df to a csv
# script_df.to_csv('../data/script_df.csv', sep=',')

# other ideas for dataframe
# get the episode name
# get the season

# doing the processing this way (for looping each file) stores the df in memory as we add to it,
# severely slowing down the processing speed.

## merge the two dataframes using a fuzzy match
# scripts_df = pd.read_csv('../data/scripts_df.csv')
# character_df = pd.read_csv('../data/character_df.csv')
# scripts_df.index = scripts_df['focus']
# character_df.index = character_df['character_name']

# print(scripts_df.head(5))
# print(character_df.head(5))
#
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# from numpy import NaN
#
#
# def get_closest_match(x, list_strings):
#     best_match = None
#     highest_jw = 0
#     for current_string in list_strings:
#         # print(type(current_string))
#         # print(current_string)
#         # print(type(x))
#         # print(x)
#         try:
#             current_score = jellyfish.levenshtein_distance(x, current_string)
#             if current_score > highest_jw:
#                 highest_jw = current_score
#                 best_match = current_string
#         except TypeError:
#             continue
#     return best_match
#
#
# def fuzzy_match(a, b):
#     left = '1' if pd.isnull(a) else a
#     right = b.fillna('2')
#     print(left, right)
#     out = difflib.get_close_matches(left, right)
#     print(out)
#     return out[0] if out else None
#
#
#



# scripts_df.index = tqdm(scripts_df.index.map(lambda x: get_closest_match(x, character_df.index)))

# print(scripts_df.head(20))
#
# from pprint import pprint
# pprint(scripts_df.head(100))
#





#########################
##### New Matching ######
#########################

##### @UNCOMMENT this if don't want to read csv with split text #####

# pprint(scripts_df)
# print(character_df.head())
# print(scripts_df.columns)
# print(character_df.columns)
#
#
# ## Split the columns and remove the stop words.
# # ALso replace np.nan with N/A string
# # scripts_df = scripts_df.iloc[1:5]
#
# ## Why where there nas in the beginning???
# # Remove NAs in scripts df
# scripts_df = scripts_df[pd.notnull(scripts_df['focus'])]
# scripts_df = scripts_df[pd.notnull(scripts_df['sentence'])]
# scripts_df = scripts_df[(scripts_df.astype(str)['tokens'] != '[]')]
# # print(scripts_df[scripts_df.isnull().any(axis=1)])
# print(scripts_df.shape[0])
#
# scripts_df.focus = scripts_df.focus.fillna(value="NA")
# character_df.character_name = character_df.character_name.fillna(value="NA")
#
# ## Create all character text - use this to filter out text that id not in character text
# allCharacter_text = ' '.join(character_df.character_name.tolist()).split(' ')
# allCharacter_text = set(map(lambda x: x.lower(), allCharacter_text))
#
# ## Set lower case script focus and character name (in char df)
# scripts_df.focus = scripts_df.focus.str.lower()
# character_df.character_name = character_df.character_name.str.lower()


# print(allCharacter_text)
# print(scripts_df.focus.iloc[1:100])
# print(character_df.character_name[1:100])

# pprint(scripts_df)
# pprint(character_df)

# # Split
# scripts_df['focus_split'] = scripts_df['focus'].apply(lambda x: [item for item in x.split(' ') if (item not in stopwords.words('english')) and (item in allCharacter_text)])
# character_df['character_name_split'] = character_df.character_name.apply(lambda x: [item for item in x.split(' ') if item not in stopwords.words('english')])
#
# ### Remove focus splits that are empty
# scripts_df = scripts_df[(scripts_df.astype(str)['focus_split'] != '[]')]
# print(scripts_df[scripts_df.isnull().any(axis=1)])
# print(scripts_df.shape[0])
#
# print("SPLIT AND REMOVED STOP WORDS")
#
#
# # Write to csv
# scripts_df.to_csv("../data/scripts_df.csv")
# character_df.to_csv("../data/character_df.csv")
# #
# # Read from cv
# # scripts_df = pd.read_csv("scripts_df.csv")
# # character_df = pd.read_csv("character_df.csv")
#
# """
# @Note:
# - Save the files to avoid having to read
# - Remove mom troi (@Valentino)
# - Remove from script focus all words that are not in character name @IMPORTANT!!!!!
# - For things like star trek the next generation - add them in character_df text.
# """
#
# # Create new dataframe
# merged_data = scripts_df.copy() #.iloc[0:3] # @ Note make sure proper copy
# merged_data['characterName'] = np.nan
# print(merged_data.shape[0])
#
# # For every word in script focus, compare to every word in character df charac name
# # If have a match, assign character name to merged_data char name
# print(merged_data[merged_data.isnull().any(axis=1)])
# pprint(merged_data)
#
# for index_script, row_script in merged_data.iterrows():
#     if index_script % 250 == 0:
#         print(index_script)
#     for index_char, row_char in character_df.iterrows():
#         try:
#             matchedChar = bool(set(row_script.focus_split) & set(row_char.character_name_split))
#             if matchedChar: # @note change because this only one (first???) match
#                 merged_data.loc[index_script, "characterName"] = row_char.character_name
#                 # print("MATCH WITH FOCUS: {} and CHAR: {}".format(row_script.focus_split, row_char.character_name_split ))
#                 # print(merged_data.loc[index_script, "characterName"])
#         except TypeError:
#             continue
#
# merged_data.to_csv("../data/script_df_withCharName.csv")
#
# pprint(merged_data)

# read in the script data with the character names

# scripts_df_withCharName = pd.read_csv('../data/script_df_withCharName.csv')
# print(scripts_df_withCharName.head(10))
# print(scripts_df_withCharName.columns)
#
# # merge the scripts df with the character df
# character_df = pd.read_csv('../data/character_df.csv')
# scripts_character_df = pd.merge(scripts_df_withCharName, character_df, left_on='characterName', right_on='character_name')
# # drop columns that are fucked up
# scripts_character_df = scripts_character_df.drop(scripts_character_df.columns[[0, 1, 2, 3, 15, 16, 17, 18]], axis=1)
# print(scripts_character_df)
# print(scripts_character_df.shape[0])
# print(scripts_character_df.columns)
#
# # write this to a csv file
# scripts_character_df.to_csv('../data/scripts_character_df.csv', sep=',')


## merge the episode dataframe with the scripts_character_df dataframe
# scripts_character_df = pd.read_csv('../data/scripts_character_df.csv')
# episode_df = pd.read_csv('../data/episode_df.csv')
# scripts_character_episode_df = pd.merge(scripts_character_df, episode_df, left_on='episode', right_on='merge_index')
# scripts_character_episode_df.to_csv('../data/scripts_character_episode_df.csv', sep=',')


## then do some count stuff
## count the occurrences of make it so in each episode, similarly engage
## avg warp speed over time
## % share of speaking time by character
##
