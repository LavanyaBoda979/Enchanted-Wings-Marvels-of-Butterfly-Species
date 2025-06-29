import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Header from './components/Header';
import Hero from './components/Hero';
import Features from './components/Features';
import Scenarios from './components/Scenarios';
import ClassificationDemo from './components/ClassificationDemo';
import ModelInfo from './components/ModelInfo';
import Footer from './components/Footer';

function App() {
  const [activeSection, setActiveSection] = useState('home');

  return (
    <div className="min-h-screen bg-gradient-to-br from-butterfly-50 via-white to-nature-50">
      <Header activeSection={activeSection} setActiveSection={setActiveSection} />
      
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <Hero />
        <Features />
        <ClassificationDemo />
        <Scenarios />
        <ModelInfo />
      </motion.main>
      
      <Footer />
    </div>
  );
}

export default App;