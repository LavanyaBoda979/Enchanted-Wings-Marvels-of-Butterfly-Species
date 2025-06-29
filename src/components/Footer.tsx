import React from 'react';
import { motion } from 'framer-motion';
import { Router as Butterfly, Github, Mail, ExternalLink } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="flex items-center space-x-2 mb-4"
            >
              <Butterfly className="h-8 w-8 text-butterfly-400" />
              <span className="text-2xl font-bold">Enchanted Wings</span>
            </motion.div>
            <p className="text-gray-300 mb-6 max-w-md">
              Advanced butterfly species classification using transfer learning and deep neural networks. 
              Supporting biodiversity conservation through AI-powered species identification.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-400 hover:text-butterfly-400 transition-colors">
                <Github className="w-6 h-6" />
              </a>
              <a href="#" className="text-gray-400 hover:text-butterfly-400 transition-colors">
                <Mail className="w-6 h-6" />
              </a>
              <a href="#" className="text-gray-400 hover:text-butterfly-400 transition-colors">
                <ExternalLink className="w-6 h-6" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">Features</a></li>
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">Demo</a></li>
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">Scenarios</a></li>
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">Model Info</a></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Resources</h3>
            <ul className="space-y-2">
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">Documentation</a></li>
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">API Reference</a></li>
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">Dataset</a></li>
              <li><a href="#" className="text-gray-300 hover:text-butterfly-400 transition-colors">Research Paper</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm">
            © 2025 Enchanted Wings. Built for biodiversity conservation and scientific research.
          </p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <a href="#" className="text-gray-400 hover:text-butterfly-400 text-sm transition-colors">Privacy Policy</a>
            <a href="#" className="text-gray-400 hover:text-butterfly-400 text-sm transition-colors">Terms of Service</a>
            <a href="#" className="text-gray-400 hover:text-butterfly-400 text-sm transition-colors">Contact</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;