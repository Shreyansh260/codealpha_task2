"""
Speech Emotion Recognition System
Recognizes emotions (happy, angry, sad, neutral, etc.) from speech audio
using deep learning and signal processing techniques.
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extract various acoustic features from audio files"""
    
    def __init__(self, sample_rate=22050, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
    
    def load_audio(self, file_path, offset=0.5):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                   duration=self.duration, offset=offset)
            
            # Pad or trim to fixed length
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), 'constant')
            else:
                audio = audio[:self.n_samples]
            
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_mfcc(self, audio, n_mfcc=13):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    
    def extract_chroma(self, audio):
        """Extract Chroma features"""
        chroma = librosa.feature.chroma(y=audio, sr=self.sample_rate)
        return np.mean(chroma.T, axis=0)
    
    def extract_mel_spectrogram(self, audio):
        """Extract Mel-spectrogram features"""
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        return np.mean(mel.T, axis=0)
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_prosodic_features(self, audio):
        """Extract prosodic features"""
        features = {}
        
        # Fundamental frequency (F0)
        f0 = librosa.yin(audio, fmin=50, fmax=300)
        f0_clean = f0[f0 > 0]  # Remove unvoiced frames
        
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_min'] = np.min(f0_clean)
            features['f0_max'] = np.max(f0_clean)
        else:
            features.update({'f0_mean': 0, 'f0_std': 0, 'f0_min': 0, 'f0_max': 0})
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
    
    def extract_all_features(self, file_path):
        """Extract all features from an audio file"""
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        features = {}
        
        # MFCC features
        mfcc = self.extract_mfcc(audio)
        features.update({f'mfcc_{i}': mfcc[i] for i in range(len(mfcc))})
        
        # Chroma features
        chroma = self.extract_chroma(audio)
        features.update({f'chroma_{i}': chroma[i] for i in range(len(chroma))})
        
        # Mel-spectrogram features
        mel = self.extract_mel_spectrogram(audio)
        features.update({f'mel_{i}': mel[i] for i in range(len(mel))})
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)
        
        # Prosodic features
        prosodic_features = self.extract_prosodic_features(audio)
        features.update(prosodic_features)
        
        return features

class EmotionRecognitionModel:
    """Deep learning models for emotion recognition"""
    
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def create_dnn_model(self, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        """Create a Deep Neural Network model"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.BatchNormalization()
        ])
        
        for units in hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model(self, sequence_length=128):
        """Create a 1D CNN model for sequential features"""
        # Reshape input for CNN (assuming MFCC-like sequential data)
        model = keras.Sequential([
            layers.Input(shape=(sequence_length, 1)),
            
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_lstm_model(self, sequence_length=128):
        """Create an LSTM model for sequential features"""
        model = keras.Sequential([
            layers.Input(shape=(sequence_length, 1)),
            
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_hybrid_model(self):
        """Create a hybrid CNN-LSTM model"""
        model = keras.Sequential([
            layers.Input(shape=(128, 1)),
            
            # CNN layers
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True, dropout=0.3),
            layers.LSTM(64, dropout=0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class EmotionRecognitionSystem:
    """Complete emotion recognition system"""
    
    def __init__(self, emotions=['angry', 'happy', 'neutral', 'sad']):
        self.emotions = emotions
        self.num_classes = len(emotions)
        self.feature_extractor = AudioFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(emotions)
    
    def load_dataset(self, audio_files, labels):
        """Load and preprocess dataset"""
        features_list = []
        labels_list = []
        
        print("Extracting features from audio files...")
        for i, (file_path, label) in enumerate(zip(audio_files, labels)):
            if i % 50 == 0:
                print(f"Processed {i}/{len(audio_files)} files")
            
            features = self.feature_extractor.extract_all_features(file_path)
            if features is not None:
                features_list.append(list(features.values()))
                labels_list.append(label)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_encoded
    
    def train_model(self, X, y, model_type='dnn', validation_split=0.2, epochs=100):
        """Train the emotion recognition model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create model based on type
        model_creator = EmotionRecognitionModel(X.shape[1], self.num_classes)
        
        if model_type == 'dnn':
            self.model = model_creator.create_dnn_model()
        elif model_type == 'cnn':
            # Reshape data for CNN
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            self.model = model_creator.create_cnn_model(X.shape[1])
        elif model_type == 'lstm':
            # Reshape data for LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            self.model = model_creator.create_lstm_model(X.shape[1])
        elif model_type == 'hybrid':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            self.model = model_creator.create_hybrid_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        emotion_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=emotion_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotion_names, yticklabels=emotion_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return accuracy, y_pred_classes
    
    def predict_emotion(self, audio_file):
        """Predict emotion from a single audio file"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio_file)
        if features is None:
            return None
        
        # Preprocess
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        # Reshape if needed (for CNN/LSTM models)
        if len(self.model.input_shape) == 3:
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Predict
        prediction = self.model.predict(X_scaled)
        emotion_idx = np.argmax(prediction)
        confidence = prediction[0][emotion_idx]
        
        emotion = self.label_encoder.inverse_transform([emotion_idx])[0]
        
        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'probabilities': dict(zip(self.emotions, prediction[0]))
        }
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the emotion recognition system
    emotions = ['angry', 'happy', 'neutral', 'sad', 'fearful', 'surprised']
    ser_system = EmotionRecognitionSystem(emotions)
    
    print("Speech Emotion Recognition System")
    print("=" * 50)
    print(f"Target emotions: {emotions}")
    
    # Example of how to use the system with your dataset
    """
    # Step 1: Prepare your dataset
    # For RAVDESS dataset format: Actor_Emotion_Statement_Intensity_Repetition.wav
    # For EMO-DB format: [EmotionCode][SpeakerCode][SentenceCode].wav
    
    audio_files = ['path/to/audio1.wav', 'path/to/audio2.wav', ...]
    labels = ['happy', 'sad', 'angry', ...]
    
    # Step 2: Load and preprocess dataset
    X, y = ser_system.load_dataset(audio_files, labels)
    
    # Step 3: Train model
    history = ser_system.train_model(X, y, model_type='hybrid', epochs=100)
    
    # Step 4: Plot training history
    ser_system.plot_training_history(history)
    
    # Step 5: Evaluate model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    accuracy, predictions = ser_system.evaluate_model(X_test, y_test)
    
    # Step 6: Predict on new audio
    result = ser_system.predict_emotion('path/to/new_audio.wav')
    print(f"Predicted emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
    """
    
    # Display system capabilities
    print("\nFeatures extracted:")
    print("- MFCC (Mel-Frequency Cepstral Coefficients)")
    print("- Chroma features")
    print("- Mel-spectrogram")
    print("- Spectral features (centroid, rolloff, zero-crossing rate)")
    print("- Prosodic features (F0, RMS energy)")
    
    print("\nSupported models:")
    print("- Deep Neural Network (DNN)")
    print("- Convolutional Neural Network (CNN)")
    print("- Long Short-Term Memory (LSTM)")
    print("- Hybrid CNN-LSTM")
    
    print("\nDataset compatibility:")
    print("- RAVDESS (Ryerson Audio-Visual Database)")
    print("- TESS (Toronto Emotional Speech Set)")
    print("- EMO-DB (Berlin Database of Emotional Speech)")
    print("- Custom datasets with proper labeling")