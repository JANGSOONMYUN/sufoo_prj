'use client';

import React from 'react';
import { skillsData } from '@/lib/portfolio/data';

const SkillsSection = () => {
  return (
    <section className="py-10">
      <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">기술 스택</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">프로그래밍 언어</h3>
          <div className="flex flex-wrap gap-2">
            {skillsData.languages.map((language, index) => (
              <span 
                key={index} 
                className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 rounded-full text-sm"
              >
                {language}
              </span>
            ))}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">개발 도구</h3>
          <div className="flex flex-wrap gap-2">
            {skillsData.tools.map((tool, index) => (
              <span 
                key={index} 
                className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-100 rounded-full text-sm"
              >
                {tool}
              </span>
            ))}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">운영체제</h3>
          <div className="flex flex-wrap gap-2">
            {skillsData.os.map((os, index) => (
              <span 
                key={index} 
                className="px-3 py-1 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-100 rounded-full text-sm"
              >
                {os}
              </span>
            ))}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md md:col-span-2">
          <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">오픈 소스</h3>
          <div className="flex flex-wrap gap-2">
            {skillsData.openSources.map((openSource, index) => (
              <span 
                key={index} 
                className="px-3 py-1 bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-100 rounded-full text-sm"
              >
                {openSource}
              </span>
            ))}
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">기타 도구</h3>
          <div className="flex flex-wrap gap-2">
            {skillsData.otherTools.map((tool, index) => (
              <span 
                key={index} 
                className="px-3 py-1 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-100 rounded-full text-sm"
              >
                {tool}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default SkillsSection;
