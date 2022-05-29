import pandas as pd
import Recommenders as Recommenders
import streamlit as st

song_df_1 = pd.read_csv('triplets_file.csv')

song_df_2 = pd.read_csv('song_data.csv')

st.title('Recommending Songs')

selected_user_id = st.text_input('Enter User Id', 'b80344d063b5ccb3212f76538f3d9e43d87dca9e')

# combine both data
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')

# creating new feature combining title and artist name
song_df['song'] = song_df['title']+' - '+song_df['artist_name']

# taking top 10k samples for quick results
song_df = song_df.head(10000)

# cummulative sum of listen count of the songs
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()

grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum ) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])

pr = Recommenders.popularity_recommender_py()
pr.create(song_df, 'user_id', 'song')
# display the top 10 popular songs
precommendations = pr.recommend(selected_user_id)

ir = Recommenders.item_similarity_recommender_py()
ir.create(song_df, 'user_id', 'song')

user_items = ir.get_user_items(selected_user_id)

# give song recommendation for that user
irecommendations = ir.recommend(selected_user_id)

st.header("Recommendations for you!")
recommendations = []

recommendations = precommendations['song'].head(5).tolist() + irecommendations['song'].head(5).tolist()

for recommendation in recommendations:
    st.write(recommendation)