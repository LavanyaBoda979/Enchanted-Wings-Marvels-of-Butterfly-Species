import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, Camera, Loader, CheckCircle, AlertCircle } from 'lucide-react';

const ClassificationDemo: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mock butterfly species data
  const mockResults = [
    { species: 'Monarch Butterfly', confidence: 0.94, scientific: 'Danaus plexippus' },
    { species: 'Blue Morpho', confidence: 0.89, scientific: 'Morpho peleides' },
    { species: 'Swallowtail', confidence: 0.87, scientific: 'Papilio machaon' },
    { species: 'Painted Lady', confidence: 0.82, scientific: 'Vanessa cardui' },
    { species: 'Red Admiral', confidence: 0.78, scientific: 'Vanessa atalanta' }
  ];

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) return;
    
    setIsClassifying(true);
    setResult(null);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock classification result
    const randomResult = mockResults[Math.floor(Math.random() * mockResults.length)];
    setResult({
      ...randomResult,
      confidence: Math.random() * 0.2 + 0.8, // Random confidence between 0.8-1.0
      processingTime: Math.random() * 0.5 + 0.3 // Random time between 0.3-0.8s
    });
    
    setIsClassifying(false);
  };

  const sampleImages = [
    'https://images.pexels.com/photos/326055/pexels-photo-326055.jpeg?auto=compress&cs=tinysrgb&w=400',
    'https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg?auto=compress&cs=tinysrgb&w=400',
    'https://images.pexels.com/photos/1563356/pexels-photo-1563356.jpeg?auto=compress&cs=tinysrgb&w=400',
    'https://images.pexels.com/photos/1805162/pexels-photo-1805162.jpeg?auto=compress&cs=tinysrgb&w=400'
  ];

  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-butterfly-50 to-nature-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
            Try the <span className="gradient-text">Classification Demo</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload a butterfly image or select from our samples to see the AI model in action.
            Experience real-time species identification with confidence scores.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="bg-white p-8 rounded-xl shadow-lg border border-butterfly-100">
              <h3 className="text-2xl font-semibold text-gray-900 mb-6">Upload Image</h3>
              
              {!selectedImage ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-butterfly-300 rounded-lg p-12 text-center cursor-pointer hover:border-butterfly-500 transition-colors"
                >
                  <Upload className="w-12 h-12 text-butterfly-400 mx-auto mb-4" />
                  <p className="text-lg text-gray-600 mb-2">Click to upload butterfly image</p>
                  <p className="text-sm text-gray-500">Supports JPG, PNG, WebP formats</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <img
                    src={selectedImage}
                    alt="Selected butterfly"
                    className="w-full h-64 object-cover rounded-lg shadow-md"
                  />
                  <div className="flex gap-3">
                    <button
                      onClick={handleClassify}
                      disabled={isClassifying}
                      className="btn-primary flex-1 flex items-center justify-center"
                    >
                      {isClassifying ? (
                        <>
                          <Loader className="w-5 h-5 mr-2 animate-spin" />
                          Classifying...
                        </>
                      ) : (
                        <>
                          <Camera className="w-5 h-5 mr-2" />
                          Classify Species
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => {
                        setSelectedImage(null);
                        setResult(null);
                      }}
                      className="btn-secondary"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              )}
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </div>

            {/* Sample Images */}
            <div className="bg-white p-6 rounded-xl shadow-lg border border-butterfly-100">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Try Sample Images</h4>
              <div className="grid grid-cols-2 gap-3">
                {sampleImages.map((image, index) => (
                  <img
                    key={index}
                    src={image}
                    alt={`Sample butterfly ${index + 1}`}
                    onClick={() => {
                      setSelectedImage(image);
                      setResult(null);
                    }}
                    className="w-full h-24 object-cover rounded-lg cursor-pointer hover:opacity-80 transition-opacity shadow-sm"
                  />
                ))}
              </div>
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-white p-8 rounded-xl shadow-lg border border-butterfly-100"
          >
            <h3 className="text-2xl font-semibold text-gray-900 mb-6">Classification Results</h3>
            
            {!result && !isClassifying && (
              <div className="text-center py-12">
                <Camera className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">Upload an image to see classification results</p>
              </div>
            )}

            {isClassifying && (
              <div className="text-center py-12">
                <Loader className="w-16 h-16 text-butterfly-500 mx-auto mb-4 animate-spin" />
                <p className="text-gray-600">Analyzing butterfly species...</p>
              </div>
            )}

            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="space-y-6"
              >
                <div className="flex items-center space-x-3 text-green-600">
                  <CheckCircle className="w-6 h-6" />
                  <span className="font-semibold">Classification Complete</span>
                </div>

                <div className="bg-gradient-to-r from-butterfly-50 to-nature-50 p-6 rounded-lg">
                  <h4 className="text-xl font-bold text-gray-900 mb-2">{result.species}</h4>
                  <p className="text-gray-600 italic mb-3">{result.scientific}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Confidence</span>
                    <span className="font-semibold text-butterfly-600">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div
                      className="bg-gradient-to-r from-butterfly-500 to-nature-500 h-2 rounded-full transition-all duration-1000"
                      style={{ width: `${result.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="text-sm text-gray-500">
                  Processing time: {result.processingTime?.toFixed(2)}s
                </div>

                <div className="bg-blue-50 p-4 rounded-lg">
                  <h5 className="font-semibold text-blue-900 mb-2">Species Information</h5>
                  <p className="text-blue-800 text-sm">
                    This classification is based on advanced transfer learning models trained on 
                    thousands of butterfly images. For research purposes, consider multiple 
                    observations and expert validation.
                  </p>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default ClassificationDemo;