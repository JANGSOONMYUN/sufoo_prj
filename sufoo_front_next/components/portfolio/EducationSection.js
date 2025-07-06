'use client';

import React from 'react';
import { educationData } from '@/lib/portfolio/data';

const EducationSection = () => {
  return (
    <section className="py-10">
      <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">학력</h2>
      <div className="space-y-6">
        {educationData.map((education, index) => (
          <div key={index} className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <div className="flex flex-col md:flex-row md:justify-between md:items-start">
              <div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">{education.institution}</h3>
                <p className="text-lg text-gray-700 dark:text-gray-300">{education.major}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{education.location}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{education.degree}</p>
              </div>
              <div className="mt-2 md:mt-0 md:text-right">
                <span className="inline-block bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-100 px-3 py-1 rounded-full text-sm">
                  {education.period}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default EducationSection;
