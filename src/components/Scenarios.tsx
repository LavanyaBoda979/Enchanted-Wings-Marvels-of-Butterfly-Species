import React from 'react';
import { motion } from 'framer-motion';
import { Binary as Binoculars, FlaskConical, GraduationCap, Camera, MapPin, Users } from 'lucide-react';

const Scenarios: React.FC = () => {
  const scenarios = [
    {
      icon: Binoculars,
      title: 'Biodiversity Monitoring',
      description: 'Field researchers and conservationists use our system for real-time butterfly identification in diverse habitats. This enables efficient species inventory, population studies, and habitat management for data-driven conservation strategies.',
      features: [
        'Real-time field identification',
        'Species inventory tracking',
        'Population monitoring',
        'Habitat assessment tools'
      ],
      color: 'from-nature-500 to-nature-600',
      bgColor: 'bg-nature-50'
    },
    {
      icon: FlaskConical,
      title: 'Ecological Research',
      description: 'Automated classification systems support long-term butterfly behavior and distribution studies. Researchers deploy camera systems to monitor migratory patterns, habitat preferences, and environmental responses.',
      features: [
        'Automated monitoring systems',
        'Migration pattern tracking',
        'Behavioral analysis',
        'Environmental impact studies'
      ],
      color: 'from-butterfly-500 to-butterfly-600',
      bgColor: 'bg-butterfly-50'
    },
    {
      icon: GraduationCap,
      title: 'Citizen Science & Education',
      description: 'Interactive tools engage students and enthusiasts in butterfly identification and data collection. Mobile-friendly classification promotes environmental awareness and scientific participation.',
      features: [
        'Educational mobile apps',
        'Interactive learning tools',
        'Community data collection',
        'Environmental awareness programs'
      ],
      color: 'from-purple-500 to-purple-600',
      bgColor: 'bg-purple-50'
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
            Real-World <span className="gradient-text">Applications</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Our butterfly classification system serves diverse applications in conservation, 
            research, and education, making a meaningful impact on biodiversity preservation.
          </p>
        </motion.div>

        <div className="space-y-12">
          {scenarios.map((scenario, index) => (
            <motion.div
              key={scenario.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: index * 0.2 }}
              viewport={{ once: true }}
              className={`${scenario.bgColor} rounded-2xl p-8 lg:p-12`}
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                <div className={index % 2 === 1 ? 'lg:order-2' : ''}>
                  <div className={`w-16 h-16 bg-gradient-to-r ${scenario.color} rounded-xl flex items-center justify-center mb-6`}>
                    <scenario.icon className="w-8 h-8 text-white" />
                  </div>
                  
                  <h3 className="text-2xl lg:text-3xl font-bold text-gray-900 mb-4">
                    {scenario.title}
                  </h3>
                  
                  <p className="text-lg text-gray-700 mb-6 leading-relaxed">
                    {scenario.description}
                  </p>
                  
                  <ul className="space-y-3">
                    {scenario.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-center space-x-3">
                        <div className={`w-2 h-2 bg-gradient-to-r ${scenario.color} rounded-full`}></div>
                        <span className="text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className={`${index % 2 === 1 ? 'lg:order-1' : ''} relative`}>
                  <div className="bg-white rounded-xl shadow-lg p-6 transform rotate-2 hover:rotate-0 transition-transform duration-300">
                    <img
                      src={`https://images.pexels.com/photos/${
                        index === 0 ? '1805164' : index === 1 ? '326055' : '1563356'
                      }/pexels-photo-${
                        index === 0 ? '1805164' : index === 1 ? '326055' : '1563356'
                      }.jpeg?auto=compress&cs=tinysrgb&w=600`}
                      alt={`${scenario.title} illustration`}
                      className="w-full h-64 object-cover rounded-lg"
                    />
                    <div className="mt-4 flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Camera className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-gray-600">Live Classification</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <MapPin className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-gray-600">GPS Tagged</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Floating elements */}
                  <motion.div
                    animate={{ y: [0, -10, 0] }}
                    transition={{ duration: 3, repeat: Infinity }}
                    className="absolute -top-4 -right-4 bg-white rounded-full p-3 shadow-lg"
                  >
                    <Users className="w-6 h-6 text-butterfly-500" />
                  </motion.div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Scenarios;