'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const ProjectNavigation = ({ projects }) => {
  const pathname = usePathname();
  const currentProjectId = pathname.split('/').pop();
  
  // 현재 프로젝트의 인덱스 찾기
  const currentIndex = projects.findIndex(project => project.id === currentProjectId);
  
  // 이전 및 다음 프로젝트 결정
  const prevProject = currentIndex > 0 ? projects[currentIndex - 1] : null;
  const nextProject = currentIndex < projects.length - 1 ? projects[currentIndex + 1] : null;
  
  return (
    <div className="flex justify-between items-center mt-12 pt-6 border-t border-gray-200 dark:border-gray-700">
      <div>
        {prevProject && (
          <Link 
            href={`/portfolio/projects/${prevProject.id}`}
            className="flex items-center text-blue-600 dark:text-blue-400 hover:underline"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
            </svg>
            <span>이전: {prevProject.title}</span>
          </Link>
        )}
      </div>
      
      <Link 
        href="/portfolio"
        className="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:underline"
      >
        모든 프로젝트
      </Link>
      
      <div>
        {nextProject && (
          <Link 
            href={`/portfolio/projects/${nextProject.id}`}
            className="flex items-center text-blue-600 dark:text-blue-400 hover:underline"
          >
            <span>다음: {nextProject.title}</span>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </Link>
        )}
      </div>
    </div>
  );
};

export default ProjectNavigation;
