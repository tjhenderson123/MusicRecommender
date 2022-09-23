import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

#Download music data on desktop and create approraite file path
music_df = pd.read_csv("Enter File Path Here")

scaled_col =["acousticness", "danceability", "duration_ms", "energy", "instrumentalness",
"liveness", "loudness", "speechiness", "tempo", "valence",]
scaler = MinMaxScaler()
norm_df = scaler.fit_transform(music_df[scaled_col])
indices = pd.Series(music_df.index, index=music_df["artists"])
cosine = cosine_similarity(norm_df)
def create_recommendeation(artist, model_type=cosine):
    index = indices[artist]
    score = list(enumerate(model_type[indices[artist]]))
    similarity_score = sorted(score, key = lambda x:x[1], reverse = True)
    similarity_score = similarity_score[1:11]
    top_artist_index = [i[0] for i in similarity_score]
    top_artist = music_df["artists"].iloc[top_artist_index]
    return top_artist
artist_to_lookup = input("Please enter an artist that you enjoy listening to:")
print("Based on the artist that you entered, you may also enjoy these 10 artist:")
lookup = create_recommendeation(artist_to_lookup).values
print(lookup)


left = [1, 2, 3]
height = [10, 9, 8]
tick_label = [lookup[0], lookup[1], lookup[2]]
plt.bar(left, height, tick_label=tick_label,
        width=0.8, color=['red', 'green'])
plt.xlabel('Artists')
plt.ylabel('Recommendation Level')
plt.title('Artist Recommendations (higher value = higher recommendation)')
plt.show()
x = [lookup[0], lookup[1], lookup[2]]
y = [music_df[music_df["artists"]==lookup[0]]["popularity"].values[0],
     music_df[music_df["artists"]==lookup[1]]["popularity"].values[0],
     music_df[music_df["artists"]==lookup[2]]["popularity"].values[0]]
plt.plot(x, y)
plt.xlabel('Artist')
plt.ylabel('Popularity')
plt.title('Levels of Popularity (higher value = higher popularity)')
plt.show()


x = [lookup[0], lookup[1], lookup[2]]
y = [music_df[music_df["artists"]==lookup[0]]["danceability"].values[0],
     music_df[music_df["artists"]==lookup[1]]["danceability"].values[0],
     music_df[music_df["artists"]==lookup[2]]["danceability"].values[0]]
plt.scatter(x, y, label= "= danceability level", color="green",
            marker="*", s=30)
plt.xlabel('Artist')
plt.ylabel('Danceability')
plt.title('Levels of Danceability (higher value = higher danceability)')
plt.legend()

plt.show()



