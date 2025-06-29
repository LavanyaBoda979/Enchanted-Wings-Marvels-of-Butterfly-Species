"""
Enchanted Wings: Butterfly Species Classification using Transfer Learning
Advanced deep learning model for identifying 75+ butterfly species
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

class ButterflyClassifier:
    """
    Advanced butterfly species classification system using transfer learning
    
    Features:
    - Transfer learning with pre-trained CNNs
    - Data augmentation for robust training
    - Multiple architecture support (EfficientNet, ResNet)
    - Real-time inference capabilities
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, num_classes=75, input_shape=(224, 224, 3), architecture='efficientnet'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.architecture = architecture
        self.model = None
        self.history = None
        self.class_names = []
        
    def create_model(self, fine_tune=False):
        """
        Create butterfly classification model using transfer learning
        
        Args:
            fine_tune (bool): Whether to fine-tune the base model layers
        
        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        
        # Select base model architecture
        if self.architecture == 'efficientnet':
            base_model = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.architecture == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError("Supported architectures: 'efficientnet', 'resnet'")
        
        # Freeze base model layers initially
        base_model.trainable = fine_tune
        
        # Build classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data_generators(self, train_dir, val_dir, test_dir=None, batch_size=32):
        """
        Prepare data generators with augmentation for training and validation
        
        Args:
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory
            test_dir (str): Path to test data directory (optional)
            batch_size (int): Batch size for training
        
        Returns:
            tuple: Training, validation, and test generators
        """
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        
        test_generator = None
        if test_dir:
            test_generator = val_datagen.flow_from_directory(
                test_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        return train_generator, val_generator, test_generator
    
    def train(self, train_generator, val_generator, epochs=50, patience=10):
        """
        Train the butterfly classification model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Maximum number of training epochs
            patience (int): Early stopping patience
        
        Returns:
            tf.keras.callbacks.History: Training history
        """
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_butterfly_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, train_generator, val_generator, epochs=20, learning_rate=1e-5):
        """
        Fine-tune the pre-trained base model for better performance
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of fine-tuning epochs
            learning_rate (float): Lower learning rate for fine-tuning
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Unfreeze the base model
        self.model.layers[1].trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        # Fine-tune the model
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )
        
        return fine_tune_history
    
    def evaluate(self, test_generator):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: Test data generator
        
        Returns:
            dict: Evaluation metrics
        """
        
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Evaluate model
        test_loss, test_accuracy, test_top5_accuracy = self.model.evaluate(
            test_generator, verbose=1
        )
        
        # Generate predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top5_accuracy': test_top5_accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return results
    
    def predict_single_image(self, image_path):
        """
        Predict butterfly species for a single image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Prediction results with species name and confidence
        """
        
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                'species': self.class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]
        
        result = {
            'predicted_species': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'top_5_predictions': top_5_predictions
        }
        
        return result
    
    def plot_training_history(self):
        """Plot training history curves"""
        
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-5 Accuracy
        axes[1, 0].plot(self.history.history['top_5_accuracy'], label='Training Top-5 Accuracy')
        axes[1, 0].plot(self.history.history['val_top_5_accuracy'], label='Validation Top-5 Accuracy')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy')
        axes[1, 0].legend()
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        
        # Save class names
        with open(filepath.replace('.h5', '_classes.json'), 'w') as f:
            json.dump(self.class_names, f)
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        
        # Load class names
        try:
            with open(filepath.replace('.h5', '_classes.json'), 'r') as f:
                self.class_names = json.load(f)
        except FileNotFoundError:
            print("Class names file not found. Please set class_names manually.")

# Example usage and training script
def main():
    """
    Main training script for butterfly classification
    """
    
    # Initialize classifier
    classifier = ButterflyClassifier(
        num_classes=75,
        architecture='efficientnet'
    )
    
    # Create model
    model = classifier.create_model()
    print(f"Model created with {model.count_params():,} parameters")
    
    # Prepare data (assuming data is organized in train/val/test folders)
    # train_gen, val_gen, test_gen = classifier.prepare_data_generators(
    #     train_dir='data/train',
    #     val_dir='data/validation',
    #     test_dir='data/test',
    #     batch_size=32
    # )
    
    # Train the model
    # history = classifier.train(train_gen, val_gen, epochs=50)
    
    # Fine-tune for better performance
    # classifier.fine_tune(train_gen, val_gen, epochs=20)
    
    # Evaluate on test set
    # results = classifier.evaluate(test_gen)
    # print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    
    # Save the model
    # classifier.save_model('butterfly_classifier_final.h5')
    
    # Example prediction
    # result = classifier.predict_single_image('sample_butterfly.jpg')
    # print(f"Predicted Species: {result['predicted_species']}")
    # print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()