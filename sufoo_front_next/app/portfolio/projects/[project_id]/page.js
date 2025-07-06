'use client';

import React from 'react';
import Link from 'next/link';
import { projectsData } from '@/lib/portfolio/data';
import { projectDetails } from '@/lib/portfolio/projects';
import ProjectNavigation from '@/components/portfolio/ProjectNavigation';

export default function ProjectDetailPage({ params }) {
  const { project_id } = params;
  
  // 프로젝트 데이터 찾기
  const project = projectsData.find(p => p.id === project_id);
  
  // 프로젝트가 없는 경우 처리
  if (!project) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <h1 className="text-3xl font-bold text-red-600 mb-4">프로젝트를 찾을 수 없습니다</h1>
        <p className="mb-6">요청하신 프로젝트 정보를 찾을 수 없습니다.</p>
        <Link href="/portfolio" className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 transition-colors">
          포트폴리오로 돌아가기
        </Link>
      </div>
    );
  }

  // 프로젝트 상세 정보
  const details = projectDetails[project_id] || {
    overview: '프로젝트 상세 정보가 준비 중입니다.',
    technologies: []
  };

  return (
    <div className="container mx-auto px-4 py-12">
      {/* 프로젝트 헤더 */}
      <header className="mb-12">
        <Link href="/portfolio" className="inline-flex items-center text-blue-600 mb-6 hover:underline">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
          </svg>
          포트폴리오로 돌아가기
        </Link>
        <h1 className="text-4xl font-bold mb-4 text-gray-900 dark:text-white">{project.title}</h1>
        <div className="flex flex-wrap items-center text-gray-600 dark:text-gray-400 mb-6">
          <span className="mr-4">{project.company}, {project.location}</span>
          <span>{project.period}</span>
        </div>
        <div className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 px-3 py-1 rounded-full text-sm">
          {project.category}
        </div>
      </header>

      {/* 프로젝트 개요 */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">프로젝트 개요</h2>
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <p className="text-gray-700 dark:text-gray-300 mb-4">{details.overview}</p>
          
          {details.teamSize && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">참여 인원</h3>
              <p className="text-gray-700 dark:text-gray-300">{details.teamSize}명</p>
            </div>
          )}
          
          {details.role && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">담당 역할 및 기여</h3>
              <ul className="list-disc pl-5 text-gray-700 dark:text-gray-300 space-y-1">
                {details.role.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          )}
          
          {details.technologies && (
            <div>
              <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">사용 기술</h3>
              <div className="flex flex-wrap gap-2">
                {details.technologies.map((tech, index) => (
                  <span 
                    key={index} 
                    className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 rounded-full text-sm"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 프로젝트 프로세스 */}
      {details.process && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">프로젝트 프로세스</h2>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            {details.process.map((process, index) => (
              <div key={index} className="mb-6 last:mb-0">
                <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">{process.title}</h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">{process.description}</p>
                
                {process.details && (
                  <ul className="list-disc pl-5 text-gray-700 dark:text-gray-300 space-y-1">
                    {process.details.map((detail, idx) => (
                      <li key={idx}>{detail}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 이미지 갤러리 */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">프로젝트 이미지</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {project.images.map((image, index) => (
              <div key={index} className="bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden aspect-video">
                <img
                  src={image}
                  alt={`프로젝트 이미지 ${index + 1}`}
                  className="object-cover w-full h-full"
                  onError={(e) => {
                    e.target.onerror = null; // prevent infinite loop
                    e.target.src="/images/placeholder.png" // placeholder image when load failed
                  }}
                />
              </div>
            ))}
        </div>
      </section>

      {/* 비디오 섹션 */}
      {project.video && (
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">프로젝트 비디오</h2>
        {/* <div className="bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden aspect-video flex items-center justify-center">
          <span className="text-gray-500 dark:text-gray-400">비디오 준비 중</span>
        </div> */}
        <div className="rounded-lg overflow-hidden">
            <video width="100%" height="100%" controls>
              <source src={project.video} type="video/mp4" />
              {/* Optionally add more <source> elements for different video formats */}
              Your browser does not support the video tag.
            </video>
          </div>
      </section>
      )}

      {/* 결과 및 성과 */}
      {details.results && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">결과 및 성과</h2>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <ul className="list-disc pl-5 text-gray-700 dark:text-gray-300 space-y-2">
              {details.results.map((result, index) => (
                <li key={index}>{result}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {/* 문제 해결 */}
      {details.challenges && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">문제 해결</h2>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            {details.challenges.map((challenge, index) => (
              <div key={index} className="mb-6 last:mb-0">
                <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">{challenge.problem}</h3>
                <p className="text-gray-700 dark:text-gray-300">{challenge.solution}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 프로젝트 네비게이션 */}
      <ProjectNavigation projects={projectsData} />
    </div>
  );
}
