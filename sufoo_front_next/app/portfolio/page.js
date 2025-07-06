'use client';

import React from 'react';
import { profileData, projectsData } from '@/lib/portfolio/data';
import ProjectCard from '@/components/portfolio/ProjectCard';
import SkillsSection from '@/components/portfolio/SkillsSection';
import ExperienceSection from '@/components/portfolio/ExperienceSection';
import EducationSection from '@/components/portfolio/EducationSection';
import ContactSection from '@/components/portfolio/ContactSection';

export default function PortfolioPage() {
  // 프로젝트를 featured 여부에 따라 정렬
  const sortedProjects = [...projectsData].sort((a, b) => {
    if (a.featured && !b.featured) return -1;
    if (!a.featured && b.featured) return 1;
    return 0;
  });

  return (
    <div className="container mx-auto px-4 py-12">
      {/* 헤더 섹션 */}
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-bold mb-4 text-gray-900 dark:text-white">{profileData.name}</h1>
        <h2 className="text-2xl text-blue-600 dark:text-blue-400 mb-6">{profileData.title}</h2>
        <p className="text-lg text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          {profileData.summary}
        </p>

      </header>

      <section className="mb-12 text-center">
        <div className="rounded-lg overflow-hidden ">
          <h3 className="text-2xl text-blue-600 dark:text-blue-400 mb-6">데모 비디오 (시나몬 지원 전용)</h3>
          <p className="text-lg text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
          시나몬의 3D 엔진 제어를 위한 프롬프트 엔지니어링을 Image generation(Gemini)과 Video generation(Sora) 서비스를 활용하여 구현해보았습니다.
          사이드 프로젝트인 영양정보검색사이트를 위한 영상입니다. 
          (시나몬의 데모 비디오의 캐릭터를 활용해보았습니다. 본 지원 이후 삭제 예정입니다.)
          </p>
            <video width="100%" height="100%" controls>
              <source src="/videos/portfolio/fodoit-1.mp4" type="video/mp4" />
              {/* Optionally add more <source> elements for different video formats */}
              Your browser does not support the video tag.
            </video>
          </div>
      </section>

      {/* 스킬 섹션 */}
      <SkillsSection />

      {/* 프로젝트 섹션 */}
      <section className="py-10">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">프로젝트</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {sortedProjects.map((project) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      </section>

      {/* 경력 섹션 */}
      <ExperienceSection />

      {/* 학력 섹션 */}
      <EducationSection />

      {/* 연락처 섹션 */}
      <ContactSection />
    </div>
  );
}
