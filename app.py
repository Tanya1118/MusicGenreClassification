import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("music_final.h5")

# Function to extract features from the audio file
def extract_features(file, num_segments=10):
    # Load the audio file
    audio, sample_rate = librosa.load(file, sr=None)

    # Get the total number of samples in the original file
    num_samples = len(audio)

    # Calculate the number of samples in each segment
    samples_per_segment = num_samples // num_segments

    # Initialize the list of extracted features
    features = []

    # Extract the features from each segment
    for segment in range(num_segments):
        # Calculate the start and finish sample for the current segment
        start = samples_per_segment * segment
        finish = start + samples_per_segment

        # Extract the mfcc
        mfcc = librosa.feature.mfcc(y=audio[start:finish], sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)

        # Add mfcc to the list of features
        features.append(mfcc.T.tolist())

    return np.array(features)



# Streamlit app
def main():
    st.title("Music Genre Classification App")

    # Upload file through Streamlit
    uploaded_file = st.file_uploader("Choose a music file", type=["wav"])

    if uploaded_file is not None:
        # Display file details
        st.audio(uploaded_file, format="audio/wav", start_time=0)
        st.write("File Details:")
        st.write(uploaded_file.type)
        st.write(uploaded_file.size)

        # Process and make prediction
        features = extract_features(uploaded_file)
        features = features[..., np.newaxis]
        prediction = model.predict(features)

        # Get the predicted genre
        predicted_genre_index = np.argmax(prediction, axis=1)[0]
        genres = ["disco", "metal", "reggae", "blues", "rock", "classical", "jazz", "hiphop", "country", "pop"]
        predicted_genre = genres[predicted_genre_index]

        # Display prediction
        st.write("Prediction:")
        st.write(f"The predicted genre is: {predicted_genre}")

if __name__ == "__main__":
    main()
