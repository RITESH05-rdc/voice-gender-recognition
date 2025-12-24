# Voice Gender Recognition

<img width="470" height="749" alt="image" src="https://github.com/user-attachments/assets/abc58fe6-7a30-457e-bc55-d80e386f5798" />

Voice-based gender recognition is an important application of speech processing and machine learning, with use cases in human–computer interaction, security systems, and speech analytics. This project presents a Voice Gender Recognition System that automatically identifies the gender of a speaker from a short voice sample. The system accepts a WAV audio file as input and performs necessary preprocessing to ensure uniform sampling and noise handling.

Key acoustic features, primarily Mel-Frequency Cepstral Coefficients (MFCCs), are extracted from the audio signal to represent the spectral characteristics of human speech. These features are visualized using an MFCC heatmap, providing an intuitive understanding of frequency variations over time. A supervised machine learning classifier (Random Forest) is trained on labeled speech data to learn gender-specific voice patterns and accurately classify unseen samples as male or female.

The trained model is deployed through a Streamlit-based web application, offering an interactive interface where users can upload audio files, preview the speech signal, visualize MFCC features, and obtain real-time gender predictions. Experimental results demonstrate that the system performs reliably for short-duration speech samples and achieves satisfactory classification accuracy. This project highlights the effectiveness of combining speech signal processing with machine learning techniques and serves as a foundation for future extensions such as emotion recognition, speaker identification, and multilingual speech analysis.
