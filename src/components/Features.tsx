import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Camera, Database, Globe, Users } from 'lucide-react';

const Features: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: 'Transfer Learning',
      description: 'Leverages pre-trained CNNs for efficient and accurate butterfly species classification with reduced training time.',
      color: 'text-butterfly-600'
    },
    {
      icon: Zap,
      title: 'Real-time Processing',
      description: 'Instant species identification from uploaded images with optimized inference pipeline.',
      color: 'text-nature-600'
    },
    {
      icon: Camera,
      title: 'Image Recognition',
      description: 'Advanced computer vision techniques for robust feature extraction and species differentiation.',
      color: 'text-butterfly-600'
    },
    {
      icon: Database,
      title: 'Comprehensive Dataset',
      description: '6,499 high-quality images across 75 butterfly species with balanced training, validation, and test sets.',
      color: 'text-nature-600'
    },
    {
      icon: Globe,
      title: 'Conservation Impact',
      description: 'Supports biodiversity monitoring, ecological research, and habitat management efforts worldwide.',
      color: 'text-butterfly-600'
    },
    {
      icon: Users,
      title: 'Citizen Science',
      description: 'Enables public participation in butterfly research and environmental education initiatives.',
      color: 'text-nature-600'
    }
  ];

  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
            Advanced AI <span className="gradient-text">Features</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Cutting-edge machine learning technology designed for accurate butterfly species identification
            and conservation research applications.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-gradient-to-br from-white to-butterfly-50 p-6 rounded-xl shadow-lg card-hover border border-butterfly-100"
            >
              <div className={`w-12 h-12 ${feature.color} mb-4`}>
                <feature.icon className="w-full h-full" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                {feature.title}
              </h3>
              <p className="text-gray-600 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;