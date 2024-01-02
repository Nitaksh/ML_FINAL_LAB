import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import SpotifyAPI as sp
from tensorflow.keras import layers
import joblib
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler,RobustScaler
import warnings
import plotly.express as px
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')
# Function to display the homepage

#targets = tf.keras.utils.to_categorical(train['Class'], num_classes=11, dtype='float32')

sc = StandardScaler()
rb = RobustScaler()

def display_homepage():
    st.markdown("""<h1 style="text-shadow: 4px 4px 4px #000000;font-family:Copperplate, Papyrus, fantasy; color:#000000; font-size: 40px;text-align:center;</h1>""", unsafe_allow_html=True)
    page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80");
        background-size: 100%;
        background-position: top left;
        background-attachment: local;
        }}
        </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
# Replace 'https://example.com/your_image.jpg' with the actual URL of your image.

    st.title("Music Genre Classification")
    st.write("Welcome to the Music Genre Classification app!")

# Function to display the visualization page
def display_visualization():
    st.title("Visualization Page")
    sc = StandardScaler()
    rb = RobustScaler()
    train = pd.read_csv('train.csv')
    test = pd.read_csv("test.csv")

    music_genres = {
    1: "Blues",
    2: "Classical",
    3: "Country",
    4: "Disco",
    5: "Hip-hop",
    6: "Jazz",
    7: "Metal",
    8: "Pop",
    9: "Reggae",
    10: "Rock"
    }
    train['Class'] = train['Class'].map(music_genres)
    targets = train['Class']
    # Load your dataset (replace 'path_to_your_dataset.csv' with your actual dataset path)
    # Assuming your dataset has a column 'Class' with genre names
    data = pd.read_csv('train.csv')
    # Group by 'Class' and count the occurrences
    genre_counts = data['Class'].value_counts()
        # Create a DataFrame for plotting
    df = pd.DataFrame({'Class': genre_counts.index, 'Count': genre_counts.values})
    # Set the title and description

    st.title("Music Genre Bar Graph")
    st.write("Bar graph based on song classes (genres)")
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Class', data=train, palette='hsv')  # Using 'hsv' palette for different colors
    plt.title('Class Counts', fontsize=20)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Display the plot in Streamlit
    st.pyplot(plt)


    #heat map
    st.title('Correlation Matrix Heatmap')
    numeric_train = train.select_dtypes(include=[np.number])
    # Calculate correlation matrix
    corr_matrix = numeric_train.corr()
    # Set up the color map for the heatmap
    cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)
    # Create the heatmap using seaborn
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, ax=ax)
    # Set the title
    ax.set_title('Correlation Heatmap')
    # Display the heatmap using Streamlit
    st.pyplot(fig)


    st.title('Distribution of the features')
    train.drop(['Id','Artist Name','Track Name','Class'], axis=1, inplace=True)
    temp=train
    test.drop(['Id','Artist Name','Track Name'], axis=1, inplace=True)

    def clean_data(df):
        df['Popularity'].fillna(float(df['Popularity'].mean()), inplace=True)
        df['key'].fillna(int(df['key'].mean()), inplace=True)
        df['instrumentalness'].fillna(float(df['instrumentalness'].mean()), inplace=True)
        return df
    train = clean_data(train)
    test = clean_data(test)


    #plots

    # Plotting the distribution of each feature
    fig = plt.figure(figsize=(15, 30))
    for i in range(1, 14):  # Adjust the range based on your actual number of features
        if i <= len(temp.columns):
            fig.add_subplot(7, 2, i)
            sns.distplot(temp.iloc[:, i-1], color='green', bins=50)
    plt.tight_layout()
    # Display the plot in Streamlit
    st.pyplot(fig)



    #density plot
    st.title('Density Plots for Features')
    st.write('Density plots for each feature in the dataset.')

    # Create a grid for subplots
    num_features = len(train.columns)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.ravel()

    # Iterate over each feature and create density plots
    for i, feature in enumerate(train.columns):
        ax = axes[i]
        sns.distplot(train[feature], ax=ax)
        ax.set_title(f'Density Plot for {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')

    # Hide unused subplots
    for i in range(num_features, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)



    from wordcloud import WordCloud

    # Title for the dashboard
    st.title('Word Cloud for Artist Names')
    # Create a string of all artist names
    artist_names = ' '.join(data['Artist Name'].dropna().astype(str))
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(artist_names)
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud for Artist Names')
    # Display the word cloud using Streamlit
    st.pyplot(fig)


    st.title('Word Cloud for Track Names')
    # Create a string of all track names
    track_names = ' '.join(data['Track Name'].dropna().astype(str))
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(track_names)
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud for Track Names')
    # Display the word cloud using Streamlit
    st.pyplot(fig)


    '''PCA Plot:2D or 3D scatter plot using PCA (Principal Component Analysis) to visualize feature 
    space in a reduced dimensionality. Points can be colored by class.'''
    # Title for the dashboard
    st.title('PCA Plot')
    # Assuming the features for PCA are in 'danceability', 'energy', 'loudness'
    features = ['danceability', 'energy', 'loudness']
    # Extract features for PCA
    X = data[features]
    # Standardize the features (optional but often recommended for PCA)
    X_standardized = (X - X.mean()) / X.std()
    # Perform PCA
    pca = PCA(n_components=3)  # You can change the number of components
    X_pca = pca.fit_transform(X_standardized)
    # Create a new DataFrame with the PCA results and class labels
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Class'] = data['Class']
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['Class'], cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA Plot')
    # Display the plot using Streamlit
    st.pyplot(fig)

def clean_data(df):
    df['Popularity'].fillna(float(df['Popularity'].mean()), inplace=True)
    df['key'].fillna(int(df['key'].mean()), inplace=True)
    df['instrumentalness'].fillna(float(df['instrumentalness'].mean()), inplace=True)
    return df
# Function to display the classification page
def display_classification():
    st.title('Song Genre Classifier')
    
    # Input box to type the song name
    song_name = st.text_input('Enter the song name')

    st.write('You selected:', song_name)
    if song_name == '' :
        return
    data = sp.get_metadata(song_name)
    popularity = data["popularity"]
    danceability = data["audio_features"]["danceability"]
    energy = data["audio_features"]["energy"]
    key = data["audio_features"]["key"]
    loudness = data["audio_features"]["loudness"]
    mode = data["audio_features"]["mode"]
    speechiness = data["audio_features"]["speechiness"]
    acousticness = data["audio_features"]["acousticness"]
    instrumentalness = data["audio_features"]["instrumentalness"]
    liveness = data["audio_features"]["liveness"]
    valence = data["audio_features"]["valence"]
    tempo = data["audio_features"]["tempo"]
    duration_ms = data["audio_features"]["duration_ms"]
    time_signature = data["audio_features"]["time_signature"]

    # Create a DataFrame
    data_dict = {
        "Popularity": popularity,
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_in min/ms": duration_ms,
        "time_signature": time_signature
    }
    if data:
        st.write("Song Name:", data["name"])
        st.write("Artists:", ", ".join(data["artists"]))
        st.write("Year:", data["year"])
        st.write("Album:", data["album"])
        st.write("Popularity:", data["popularity"])
        st.write("Audio features:", data["audio_features"])
    
    file = 'finalized_model'
    loaded_model = joblib.load(file)
    # load the model using joblib and predict a sing row of data
    data_dict = pd.DataFrame(data_dict, index=[0])
    temp_df = pd.read_csv('train.csv')
    temp_df.drop(['Id','Artist Name','Track Name','Class'], axis=1, inplace=True)
    temp_df = clean_data(temp_df)
    #add data dict to the temp df
    temp_df = pd.concat([temp_df, data_dict], ignore_index=True)
    rb.fit(temp_df)
    scaled_temp_df = rb.transform(temp_df)
    scaled_data_dict = scaled_temp_df[-1]
    result = loaded_model.predict(scaled_data_dict.reshape(1, -1))
    del temp_df
    #map all the genre number to actual genre name
    result = pd.DataFrame(result)
    result = result[0].map({
    1: "Blues",
    2: "Classical",
    3: "Country",
    4: "Disco",
    5: "Hip-hop",
    6: "Jazz",
    7: "Metal",
    8: "Pop",
    9: "Reggae",
    10: "Rock"
    })
    st.write("Predicted Genre:", result[0])
    
    # Button to trigger genre prediction

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Homepage", "Visualization", "Classification"])

    if selection == "Homepage":
        display_homepage()
    elif selection == "Visualization":
        display_visualization()
    elif selection == "Classification":
        display_classification()

# Run the app
if __name__ == "__main__":
    main()
