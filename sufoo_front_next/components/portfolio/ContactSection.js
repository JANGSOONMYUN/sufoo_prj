'use client';

import React from 'react';
import { profileData } from '@/lib/portfolio/data';

const ContactSection = () => {
  return (
    <section className="py-10">
      <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">연락처</h2>
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <div className="flex items-center mb-4">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600 dark:text-blue-400 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <a href={`mailto:${profileData.email}`} className="text-blue-600 dark:text-blue-400 hover:underline">
            {profileData.email}
          </a>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;
