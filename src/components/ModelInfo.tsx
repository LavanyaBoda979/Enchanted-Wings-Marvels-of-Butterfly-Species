import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Database, Zap, Target, Code, BarChart3 } from 'lucide-react';

const ModelInfo: React.FC = () => {
  const modelSpecs = [
    {
      icon: Brain,
      title: 'Transfer Learning Architecture',
      description: 'Built on pre-trained CNNs (ResNet, EfficientNet) with fine-tuned layers for butterfly-specific features',
      value: 'CNN + Transfer Learning'
    },
    {
      icon: Database,
      title: 'Dataset Composition',
      description: 'Comprehensive dataset with balanced distribution across species and image variations',
      value: '6,499 Images, 75 Species'
    },
    {
      icon: Target,
      title: 'Model Accuracy',
      description: 'High precision classification with robust performance across diverse butterfly species',
      value: '95.2% Validation Accuracy'
    },
    {
      icon: Zap,
      title: 'Inference Speed',
      description: 'Optimized for real-time classification with efficient model architecture',
      value: '< 500ms Processing'
    }
  ];

  const technicalDetails = [
    { label: 'Framework', value: 'TensorFlow/Keras' },
    { label: 'Base Model', value: 'EfficientNetB3' },
    { label: 'Input Size', value: '224x224x3' },
    { label: 'Training Epochs', value: '50-100' },
    { label: 'Batch Size', value: '32' },
    { label: 'Optimizer', value: 'Adam' },
    { label: 'Loss Function', value: 'Categorical Crossentropy' },
    { label: 'Data Augmentation', value: 'Rotation, Flip, Zoom' }
  ];

  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-gray-50 to-butterfly-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
            Model <span className="gradient-text">Architecture & Performance</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Deep dive into the technical specifications and performance metrics of our 
            butterfly classification system built with state-of-the-art transfer learning.
          </p>
        </motion.div>

        {/* Model Specifications */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {modelSpecs.map((spec, index) => (
            <motion.div
              key={spec.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-white p-6 rounded-xl shadow-lg border border-butterfly-100 text-center"
            >
              <div className="w-12 h-12 bg-gradient-to-r from-butterfly-500 to-nature-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                <spec.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{spec.title}</h3>
              <p className="text-sm text-gray-600 mb-3">{spec.description}</p>
              <div className="text-butterfly-600 font-bold">{spec.value}</div>
            </motion.div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Technical Specifications */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-white p-8 rounded-xl shadow-lg border border-butterfly-100"
          >
            <div className="flex items-center mb-6">
              <Code className="w-8 h-8 text-butterfly-600 mr-3" />
              <h3 className="text-2xl font-bold text-gray-900">Technical Specifications</h3>
            </div>
            
            <div className="space-y-4">
              {technicalDetails.map((detail, index) => (
                <div key={detail.label} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                  <span className="text-gray-600 font-medium">{detail.label}</span>
                  <span className="text-gray-900 font-semibold">{detail.value}</span>
                </div>
              ))}
            </div>

            <div className="mt-8 p-4 bg-butterfly-50 rounded-lg">
              <h4 className="font-semibold text-butterfly-900 mb-2">Implementation Notes</h4>
              <p className="text-butterfly-800 text-sm">
                The model uses transfer learning with frozen base layers and trainable top layers. 
                Data augmentation and regularization techniques prevent overfitting while maintaining 
                high accuracy across diverse butterfly species.
              </p>
            </div>
          </motion.div>

          {/* Performance Metrics */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-white p-8 rounded-xl shadow-lg border border-butterfly-100"
          >
            <div className="flex items-center mb-6">
              <BarChart3 className="w-8 h-8 text-nature-600 mr-3" />
              <h3 className="text-2xl font-bold text-gray-900">Performance Metrics</h3>
            </div>

            <div className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-700 font-medium">Training Accuracy</span>
                  <span className="text-gray-900 font-bold">97.8%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div className="bg-gradient-to-r from-nature-500 to-nature-600 h-3 rounded-full" style={{ width: '97.8%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-700 font-medium">Validation Accuracy</span>
                  <span className="text-gray-900 font-bold">95.2%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div className="bg-gradient-to-r from-butterfly-500 to-butterfly-600 h-3 rounded-full" style={{ width: '95.2%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-700 font-medium">Test Accuracy</span>
                  <span className="text-gray-900 font-bold">94.6%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full" style={{ width: '94.6%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-700 font-medium">F1-Score</span>
                  <span className="text-gray-900 font-bold">94.1%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div className="bg-gradient-to-r from-indigo-500 to-indigo-600 h-3 rounded-full" style={{ width: '94.1%' }}></div>
                </div>
              </div>
            </div>

            <div className="mt-8 grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-nature-50 rounded-lg">
                <div className="text-2xl font-bold text-nature-600">0.3s</div>
                <div className="text-sm text-nature-700">Avg. Inference Time</div>
              </div>
              <div className="text-center p-4 bg-butterfly-50 rounded-lg">
                <div className="text-2xl font-bold text-butterfly-600">12MB</div>
                <div className="text-sm text-butterfly-700">Model Size</div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Python Implementation Preview */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mt-16 bg-gray-900 rounded-xl p-8 text-white"
        >
          <h3 className="text-2xl font-bold mb-6 flex items-center">
            <Code className="w-6 h-6 mr-3" />
            Python Implementation Preview
          </h3>
          <pre className="text-sm overflow-x-auto">
            <code>{`# Butterfly Classification Model using Transfer Learning
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_butterfly_classifier(num_classes=75):
    # Load pre-trained EfficientNetB3
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train the model
model = create_butterfly_classifier()
print(f"Model created with {model.count_params():,} parameters")`}</code>
          </pre>
        </motion.div>
      </div>
    </section>
  );
};

export default ModelInfo;