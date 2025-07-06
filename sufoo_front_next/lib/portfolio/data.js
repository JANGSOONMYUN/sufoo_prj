// 포트폴리오 데이터

export const profileData = {
  name: "장순면",
  email: "jsm890803.3@gmail.com",
  title: "머신러닝 엔지니어",
  summary: "컴퓨터 비전과 머신러닝 분야에서 4년의 경력을 가진 엔지니어입니다. 딥러닝 모델 개발, 이미지 처리, OCR 시스템 구현 등 다양한 프로젝트를 수행했습니다.",
};

export const skillsData = {
  languages: ["Python", "C++"],
  tools: ["VS code", "QT", "Visual Studio"],
  os: ["Linux(Ubuntu)", "Windows"],
  openSources: [
    "Deep Learning: PyTorch, Huggingface",
    "Image Processing: OpenCV",
  ],
  otherTools: ["Docker", "Git"],
};

export const educationData = [
  {
    period: "2018.09~2020.08",
    institution: "National Cheng Kung University",
    location: "대만",
    major: "Computer Science and Information Engineering",
    degree: "석사졸업",
  },
  {
    period: "2014.09~2016.08",
    institution: "Beijing Jiaotong University",
    location: "중국",
    major: "Software Engineering",
    degree: "학사졸업",
  },
];

export const experienceData = [
  {
    period: "2023.03~현재",
    company: "더블에이파트너스",
    location: "서울",
    position: "머신러닝 엔지니어",
    duration: "2년",
  },
  {
    period: "2020.10~2022.11",
    company: "Contrel Technology",
    location: "대만",
    position: "컴퓨터비전 엔지니어",
    duration: "2년1개월",
  },
];

export const projectsData = [
  {
    id: "ai-counseling",
    title: "AI 운세상담서비스",
    company: "더블에이파트너스",
    location: "서울",
    period: "2024/01~현재",
    thumbnail: "/images/portfolio/ai-counseling.png",
    images: ["/images/portfolio/ai-counseling.png", "/images/portfolio/ai-counseling-0.png", "/images/portfolio/ai-counseling-1.png", "/images/portfolio/ai-counseling-2.png"],
    video: null,
    shortDescription: "LLM과 프롬프트 엔지니어링을 활용한 운세 및 심리상담 서비스 개발",
    category: "AI/LLM",
    featured: true,
  },
  {
    id: "moving-service",
    title: "내집이사 - 이삿짐검출 모바일어플리케이션",
    company: "더블에이파트너스",
    location: "서울",
    period: "2023/03~2024/08",
    thumbnail: "/images/portfolio/moving-service.png",
    images: ["/images/portfolio/moving-service.png", "/images/portfolio/moving-service-0.png", "/images/portfolio/moving-service-1.png", "/images/portfolio/moving-service-2.png"],
    video: null,
    shortDescription: "객체검출 및 분류 모델을 활용한 이삿짐 견적 어플리케이션 개발",
    category: "컴퓨터 비전",
    featured: true,
  },
  {
    id: "ocr-system",
    title: "OCR - 문자검출시스템",
    company: "Contrel Technology",
    location: "대만",
    period: "2022/03~2022/11",
    thumbnail: "/images/portfolio/ocr-system.png",
    images: ["/images/portfolio/ocr-system.png", "/images/portfolio/ocr-system-0.png"],
    video: null,
    shortDescription: "딥러닝 모델을 이용하여 패널의 시리얼 번호와 그 위치를 검출하는 시스템",
    category: "컴퓨터 비전",
    featured: true,
  },
  {
    id: "lcd-alignment",
    title: "LCD 정렬프로젝트",
    company: "Contrel Technology",
    location: "대만",
    period: "2021/01~2022/10",
    thumbnail: "/images/portfolio/lcd-alignment.png",
    images: ["/images/portfolio/lcd-alignment.png", "/images/portfolio/lcd-alignment-0.png"],
    video: null,
    shortDescription: "Computer Vision 기술과 CUDA 프로그래밍을 활용한 LCD 패널의 원점 정렬 시스템 개발",
    category: "컴퓨터 비전",
    featured: false,
  },
  {
    id: "robot-arm",
    title: "3D Visual-Guided Robot Arm Control",
    company: "National Cheng Kung University",
    location: "대만",
    period: "2018/09~2020/08",
    thumbnail: "/images/portfolio/robot-arm.png",
    images: ["/images/portfolio/robot-arm.png"],
    video: null,
    shortDescription: "창고 자동화 시스템을 위한 3D 비전 기반 로봇 팔 제어 연구",
    category: "로보틱스",
    featured: false,
  },
  {
    id: "nutrition-site",
    title: "영양정보검색사이트",
    company: "사이드 프로젝트",
    location: "",
    period: "2024/09~2025/01",
    thumbnail: "/images/portfolio/nutrition-site.png",
    images: ["/images/portfolio/nutrition-site.png", "/images/portfolio/nutrition-site-0.png", "/images/portfolio/nutrition-site-1.png"],
    video: "/videos/portfolio/fodoit-1.mp4",
    shortDescription: "영양 정보를 검색할 수 있는 웹사이트 개발",
    category: "웹 개발",
    featured: false,
  },
];
