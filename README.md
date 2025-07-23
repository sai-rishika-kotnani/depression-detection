🎙️ MeloMind: Voice-Based Depression Detection System
MeloMind is a machine learning–powered web application that detects potential signs of depression from short voice recordings. By leveraging speech signal processing techniques like MFCC extraction, this tool classifies user audio as either depressed or not depressed with confidence scores, waveform and spectrogram visualizations, and even downloadable PDF reports.

🧠 Features
🎧 Audio Upload Interface – Accepts .wav files for real-time voice analysis

📊 Feature Engineering – Uses MFCC (Mel-Frequency Cepstral Coefficients) for acoustic signal extraction

 Machine Learning Model – Predicts emotional state using a trained classifier

Waveform & Spectrogram Visualization – Visual cues for understanding voice signal patterns

 PDF Report Generation – Automatically creates a downloadable depression screening report

 Teddy Companion – Adds an emotional touch with advice based on predictions

 Mood Tracker Slider – Allows users to self-rate their mood for further personalization

🛠️ Tech Stack
Frontend: Streamlit (Python-based web app framework)

Audio Processing: Librosa

ML Model: Scikit-learn (trained classifier with joblib)

Visualization: Matplotlib

Model Input: MFCC features from 3-second voice clips

How to Run 
Clone the repo
Install dependencies
streamlit run app.py
 Model Details
Input: MFCCs from 3-second voice clips

Model: Trained on labeled voice data using traditional classifiers

Performance: Optimized for binary classification (depressed / not depressed)

Scaler: StandardScaler applied to MFCC features before inference

Use Case
This application is intended for academic, research, or personal exploration of speech-based emotion detection. It is not a diagnostic tool. If you're feeling down, always reach out to a trusted friend or mental health professional 

 Developed by
Sai Rishika Kotnani
Machine Learning Enthusiast 


<img width="954" height="453" alt="Screenshot 2025-07-23 161802" src="https://github.com/user-attachments/assets/e5a11de6-08ba-4af6-a629-5c80a5eabfec" />
<img width="791" height="814" alt="Screenshot 2025-07-23 161650" src="https://github.com/user-attachments/assets/aa679e1a-280f-4220-a1fe-b9771836b42b" />
<img width="933" height="804" alt="Screenshot 2025-07-23 161627" src="https://github.com/user-attachments/assets/997575e7-8373-4c7c-8293-5efc7da83ea3" />
<img width="1916" height="803" alt="Screenshot 2025-07-23 161550" src="https://github.com/user-attachments/assets/e202eaa5-28f7-4543-b24c-9239a9a0dc10" />
