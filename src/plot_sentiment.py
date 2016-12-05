import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in('vc1492a', 'g3c3nky1sj')



# let's plot the sentiment from the scripts df for now and then used the merge one later

# load the main dataframe
main = pd.read_csv('../data/scripts_character_df.csv')
main = main.drop(main.columns[[0]], axis=1)
# print(main.head(10))

# NOTE: right now our naive algorithm mis-classifies deanna troi as lwaxana troi. Recode for now, but deal with later.
main = main.replace('lwaxana troi', 'deanna troi')
# NOTE: number one is the same as william riker (context)
main = main.replace('number one', 'william riker')


# average the sentiment across episodes
episode_groupby = main.copy()
# print(character_groupby.head(100))
episode_groupby = episode_groupby.groupby(['episode'], sort=True).mean()
# order by the highest average token count
# print(episode_groupby)


# plot the average token count of each episode
# plot the average sentiment for each episode
def plotly_line_dual_ax(y1, y2, x, y1_title, y2_title, main_title, file_title):
    trace1 = go.Scatter(
        x=x,
        y=y1,
        name=y1_title
    )
    trace2 = go.Scatter(
        x=x,
        y=y2,
        name=y2_title,
        yaxis='y2'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title=main_title,
        yaxis=dict(
            title=y1_title
        ),
        yaxis2=dict(
            title=y2_title,
            overlaying='y',
            side='right'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return py.plot(fig, filename=file_title)


# mean_token_sentiment_plot = plotly_line_dual_ax(
#     episode_groupby['token_count'],
#     episode_groupby['sentiment'],
#     episode_groupby.index,
#     'Mean Token Count',
#     'Mean Sentiment',
#     'Mean Token Count and Sentiment by Episode',
#     'Mean Token Count and Sentiment by Episode'
# )

# the graph someone shows that the average token count and average sentiment follow roughly the same path
# that's interesting, as it suggests episodes with more spoken word are also more positive
# will have to see if there is something in the way the VADER algorithm calculates sentiment that would add more
# positive weight towards sentences with higher token counts.

# NOTE: Since we supplied lowercase tokens into the VADER analyzer, we suppressed it's ability to pick up on
# capital letters used within sentences, which increases the intensity of the sentiment.
# NOTE: VADER's ability to detect and adjust for contrastive conjunctions such as "but" makes it ideal
# for human language. Words after the contrastive conjunction dictate the overall sentiment class.
# NOTE: Vader has the ability to detect negated sentences 90% of the time.


# smoothing the mean token counts and sentiment so that the trend becomes more clear
# rol_mean_episode_groupby = pd.DataFrame.rolling(episode_groupby, 5).mean()

# rol_mean_token_sentiment_plot = plotly_line_dual_ax(
#     rol_mean_episode_groupby['token_count'],
#     rol_mean_episode_groupby['sentiment'],
#     rol_mean_episode_groupby.index,
#     'Rolling Avg. (5) Mean Token Count',
#     'Rolling Avg. (5) Mean Sentiment',
#     'Rolling Avg. (5) Mean of Token Count and Sentiment by Episode',
#     'Rolling Avg. (5) Mean of Token Count and Sentiment by Episode'
# )


# group by character and show the mean character sentiment by episode
# average the sentiment across episodes
episode_char_groupby = main.copy()
# print(character_groupby.head(100))
episode_char_groupby = episode_char_groupby.groupby(['episode', 'character_name'], sort=True).mean().reset_index()
episode_char_groupby = pd.DataFrame(episode_char_groupby)


def plotly_line_multi(dataset, x, y, group, main_title, file_title, num_episodes):
    # get list of sentiment values by character name
    cat_list = episode_char_groupby['character_name'].unique()
    data = []
    for entry in cat_list:
        if len(dataset[dataset['character_name'] == entry]['episode'].values) > num_episodes:
            y = dataset[dataset['character_name'] == entry]['sentiment'].values
            x = dataset[dataset['character_name'] == entry]['episode'].values
            entry = (go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=entry
            ))
            data.append(entry)
        else:
            continue
    layout = go.Layout(
        title=main_title
    )
    fig = go.Figure(data=data, layout=layout)
    return py.plot(fig, filename=file_title)


mean_sentiment_by_character_plot = plotly_line_multi(
    episode_char_groupby,
    '',
    '',
    '',
    'Mean Character Sentiment by Episode',
    'Mean Character Sentiment by Episode',
    150
)

# do a rolling mean by character
# rol_mean_episode_char_groupby = pd.DataFrame.rolling(episode_char_groupby, 5).mean()
# print(episode_char_groupby.head(5))
# print(rol_mean_episode_char_groupby.head(5))
#
#
# rol_mean_token_sentiment_plot = plotly_line_multi(
#     rol_mean_episode_char_groupby,
#     '',
#     '',
#     '',
#     'Rolling Avg. (5) Mean Character Sentiment by Episode',
#     'Rolling Avg. (5) Mean Character Sentiment by Episode',
#     150
# )


## TO DO
# group by species and show the mean species sentiment by episode
# group by rank and show sentiment by episode
# group characters and show token counts by episode









