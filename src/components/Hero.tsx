import React from 'react';
import { motion } from 'framer-motion';
import { Camera, Zap, Globe, BookOpen } from 'lucide-react';

const Hero: React.FC = () => {
  return (
    <section className="pt-24 pb-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center">
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6"
          >
            <span className="gradient-text">Enchanted Wings</span>
            <br />
            <span className="text-3xl sm:text-4xl lg:text-5xl text-gray-700">
              Marvels of Butterfly Species
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed"
          >
            Advanced butterfly species classification using transfer learning and deep neural networks. 
            Identify 75+ butterfly species with cutting-edge AI technology for biodiversity monitoring, 
            ecological research, and citizen science.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4 justify-center mb-12"
          >
            <button className="btn-primary">
              <Camera className="w-5 h-5 mr-2" />
              Try Classification Demo
            </button>
            <button className="btn-secondary">
              <BookOpen className="w-5 h-5 mr-2" />
              Learn More
            </button>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto"
          >
            <div className="text-center">
              <div className="text-3xl font-bold text-butterfly-600 mb-2">75+</div>
              <div className="text-gray-600">Butterfly Species</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-butterfly-600 mb-2">6,499</div>
              <div className="text-gray-600">Training Images</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-butterfly-600 mb-2">95%+</div>
              <div className="text-gray-600">Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-butterfly-600 mb-2">Real-time</div>
              <div className="text-gray-600">Classification</div>
            </div>
          </motion.div>
        </div>

        {/* Floating Butterfly Animation */}
        <motion.div
          animate={{ 
            y: [0, -20, 0],
            rotate: [0, 5, -5, 0]
          }}
          transition={{ 
            duration: 6,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute top-32 right-10 text-butterfly-400 opacity-20 hidden lg:block"
        >
          <div className="text-8xl">🦋</div>
        </motion.div>

        <motion.div
          animate={{ 
            y: [0, -15, 0],
            rotate: [0, -3, 3, 0]
          }}
          transition={{ 
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2
          }}
          className="absolute top-64 left-10 text-nature-400 opacity-20 hidden lg:block"
        >
          <div className="text-6xl">🦋</div>
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;