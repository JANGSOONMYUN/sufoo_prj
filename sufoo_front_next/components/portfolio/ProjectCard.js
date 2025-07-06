'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { projectsData } from '@/lib/portfolio/data';

const ProjectCard = ({ project }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden transition-all duration-300 hover:shadow-xl">
      <div className="relative h-48 w-full">
        <div className="absolute inset-0 bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
          {/* <span className="text-gray-500 dark:text-gray-400">이미지 준비 중</span> */}
          <img src={project.thumbnail} alt="준비 중인 이미지" className="object-cover w-full h-full" />
          
        </div>
      </div>
      <div className="p-5">
        <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-white">{project.title}</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
          {project.company}, {project.location} | {project.period}
        </p>
        <p className="text-gray-700 dark:text-gray-300 mb-4">{project.shortDescription}</p>
        <div className="flex justify-between items-center">
          <span className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 text-xs px-2 py-1 rounded">
            {project.category}
          </span>
          <Link href={`/portfolio/projects/${project.id}`} className="text-blue-600 dark:text-blue-400 hover:underline">
            자세히 보기 &rarr;
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ProjectCard;
