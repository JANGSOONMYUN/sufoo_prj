'use client';

import React from 'react';
import { experienceData } from '@/lib/portfolio/data';

const ExperienceSection = () => {
  return (
    <section className="py-10">
      <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">경력</h2>
      <div className="space-y-6">
        {experienceData.map((experience, index) => (
          <div key={index} className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <div className="flex flex-col md:flex-row md:justify-between md:items-start">
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">{experience.company}</h3>
                <p className="text-lg text-gray-700 dark:text-gray-300">{experience.position}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{experience.location}</p>
              </div>
              <div className="mt-2 md:mt-0 md:text-right">
                <span className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 px-3 py-1 rounded-full text-sm">
                  {experience.period}
                </span>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{experience.duration}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default ExperienceSection;
