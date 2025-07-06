// 포트폴리오 프로젝트 상세 데이터

export const projectDetails = {
  'ai-counseling': {
    overview: 'LLM과 프롬프트 엔지니어링을 활용한 운세 및 심리상담 서비스 개발 (타로, 사주, 점성학, 심리상담 등)',
    teamSize: 4,
    role: [
      'OpenAI의 GPT, Dall-E, Text-To-Speech 모델 사용',
      'LangChain과 Prompt 엔지니어링을 활용하여 운세 및 심리상담용 Prompts를 생성하고 관리',
      '실시간 채팅을 지원하는 웹소켓 서버 구축'
    ],
    technologies: ['Python', 'OpenAI API', 'LangChain'],
    process: [
      {
        title: '통신 & Agent Layer',
        description: '외부와의 모든 통신을 처리하고 LLM 설정 및 제어를 담당하는 레이어 구현',
        details: [
          '웹소켓: 실시간 채팅을 위한 Python Tornado WebSocket',
          'API Handling: 요청에 따라 함수 호출하고 LLM으로부터 결과를 받아 전송',
          'Request Parsing and Response Formatting'
        ]
      },
      {
        title: '데이터 & 실행 Layer',
        description: '데이터 검색 및 실행 흐름 관리를 담당하는 레이어 구현',
        details: [
          'Vector DB를 활용한 텍스트 임베딩 및 검색',
          'Chain 구조를 통한 실행 흐름 관리'
        ]
      }
    ],
    results: [
      '실시간 채팅 기반의 AI 운세 및 심리상담 서비스 구현',
      '다양한 상담 유형(타로, 사주, 점성학, 심리상담)에 대한 맞춤형 응답 제공',
      '웹 기반 서비스 출시 (https://sunnymong.com)'
    ],
    challenges: [
      {
        problem: 'LLM의 일관성 없는 응답 문제',
        solution: '상세한 프롬프트 엔지니어링과 Chain of Thought 기법을 적용하여 일관된 응답 유도'
      },
      {
        problem: '실시간 채팅 시 응답 지연 문제',
        solution: '웹소켓 서버 최적화 및 비동기 처리를 통한 응답 시간 개선'
      }
    ]
  },
  'moving-service': {
    overview: '객체검출 및 분류 모델을 활용한 이삿짐 견적 어플리케이션 개발',
    teamSize: 5,
    role: [
      '집 내부 사진에서 이사에 필요한 사물을 딥러닝 모델로 검출하여 견적 산출에 도움(DINO, ResNet)',
      '딥러닝 기술로 사진이 겹치는 부분에서 중복되는 물체를 검출하여 제거(LoFTR)',
      '딥러닝 추론 서버 구축 (Docker, REST API)'
    ],
    technologies: ['PyTorch', 'OpenCV', 'Docker'],
    process: [
      {
        title: '객체검출 모듈',
        description: 'DINO 모델과 ResNet을 조합하여 이사물품과 견적을 얻는 모델 구현',
        details: [
          'DINO: Transformer 기반의 Object Detection Model',
          'ResNet: Transfer Learning을 통한 이미지 분류',
          'Fusion Layer: DINO와 ResNet의 출력을 처리하여 최종 결과 출력'
        ]
      },
      {
        title: '중복검사 모듈',
        description: 'LoFTR 모델을 활용한 중복 물체 검출 시스템 구현',
        details: [
          'LoFTR: Feature Matching Model',
          '각 이미지의 keypoints를 얻고 두 사진에서 유사한 keypoints를 매칭',
          'LoFTR으로부터 중복되는 keypoints와 DINO의 bboxes의 위치를 비교하여 중복 여부 판별'
        ]
      },
      {
        title: '후처리 모듈',
        description: '이삿짐 견적 정보 후처리 시스템 구현',
        details: [
          '사진의 클래스, 이사물품 클래스, 바운딩박스, 견적값 그리고 중복여부 처리',
          '겹치는 물품, 유사물품, 거울속 물품, 빌트인 물품 등 후처리'
        ]
      }
    ],
    results: [
      '검출(Detection) 모델의 mAP: 0.6069 (IoU=0.50:0.95)',
      '방사진 분류(Classification) F1-score: 0.87',
      '속도: 800 ms (640x640 이미지 1개, RTX3090)',
      '일반고객용 모바일 플랫폼(B2C) 및 기업용 SAAS 솔루션(B2B) 출시',
      '앱스토어 출시 및 웹페이지 서비스 제공 (https://nezipisa.com/)'
    ],
    challenges: [
      {
        problem: '기존 사용 모델(YOLOv5)은 다양한 모양의 물체나 겹쳐진 물체 인식에 취약',
        solution: '고정된 Anchor를 사용하지 않는 Transformer 기반의 DINO 모델을 사용하여 해결'
      },
      {
        problem: '공개 데이터셋의 낮은 품질과 국내 사진 부족 문제',
        solution: '공개 데이터의 외국 위주의 집안 사진이 한국의 실내 양식과 맞지 않아 국내 데이터 수집부터 레이블링까지 수행'
      },
      {
        problem: '데이터 불균형 문제',
        solution: '일부 클래스가 다른 클래스보다 많은 데이터 불균형 문제를 해결하기 위해 클래스를 세분화하고 훈련 시에는 가중치를 적용'
      }
    ]
  },
  'ocr-system': {
    overview: '딥러닝 모델을 이용하여 패널의 시리얼 번호와 그 위치를 검출하는 시스템',
    teamSize: 5,
    role: [
      'Python과 PyTorch를 사용하여 문자 검출과 인식 모델을 시스템에 적용',
      '훈련을 위한 데이터 레이블링 및 데이터 증강',
      '여러 논문들을 연구하고 비교하여 제한된 요구사항(엣지 디바이스)에 적합한 딥러닝 모델을 선정'
    ],
    technologies: ['Python', 'PyTorch', 'OpenCV', 'QT'],
    process: [
      {
        title: '문자검출 모듈',
        description: 'FOTS Text Spotting 모델을 사용하여 LCD의 시리얼 번호를 인식하는 시스템 구현',
        details: [
          'FOTS: Text Spotting Model',
          '보통 detection과 recognition 모델을 따로 사용하는데 이 모델은 두 기능을 혼합하고 shared feature에 의해 더 효율이 좋음',
          '임베디드 시스템에서 사용 가능한 경량 모델'
        ]
      },
      {
        title: '후처리 모듈',
        description: '이미지 처리 및 소켓 통신 모듈 구현',
        details: [
          'Image processing: ROI cropping, Blurring, CLAHE, Adaptive thresholding',
          'Socket: TCP 통신 서버 구축, 시리얼 문자 인식 결과를 연결된 외부장치(client)로 전송'
        ]
      }
    ],
    results: [
      '검출(Detection) 모델의 F1-score: 0.82',
      '문자열 인식 정확도: 90%',
      'NVIDIA Jetson Xavier NX 임베디드 보드 사용',
      '이미지 처리 속도: 약 700ms (640x640 image)'
    ],
    challenges: [
      {
        problem: '산업용 폰트 인식 불가 문제',
        solution: '데이터 수집과 레이블링(바운딩박스, 문자클래스) 직접 수행하고, Transfer learning 기법을 사용하여 훈련'
      },
      {
        problem: '데이터 부족과 데이터 불균형 문제',
        solution: '보유한 이미지의 문자 부분을 잘라내어 새로운 조합으로 재배치 후 데이터 생성'
      },
      {
        problem: '임베디드 디바이스에서 실시간 컴퓨팅 필요',
        solution: '가벼운 ML 모델(FOTS) 선정하고, 입력 해상도, 레이어 수 그리고 채널 수를 줄여서 임베디드 보드 Jetson Xavier NX에 최적화'
      }
    ]
  },
  'lcd-alignment': {
    overview: 'Computer Vision 기술과 CUDA 프로그래밍을 활용한 LCD 패널의 원점 정렬 시스템 개발',
    teamSize: 20,
    role: [
      '템플릿 마커의 정확한 위치를 검출하는 템플릿 매칭 알고리즘 개발',
      '불분명한 템플릿 마커를 보정하는 이미지 프로세싱 개발',
      'LCD 패널을 정확한 위치로 이동시키기 위한 카메라와 Machine Stage 간 캘리브레이션',
      '매칭 속도 개선을 위한 CUDA 프로그래밍 활용'
    ],
    technologies: ['C++', 'OpenCV', 'CUDA', 'QT'],
    process: [
      {
        title: '템플릿 매칭 (Template matching)',
        description: 'NCC Matching과 Chamfer Matching을 활용한 템플릿 매칭 시스템 구현',
        details: [
          'NCC(Normalized Cross Correlation) 매칭 알고리즘 구현',
          'Chamfer 매칭 알고리즘 구현',
          'CUDA를 활용한 병렬 처리로 매칭 속도 개선'
        ]
      },
      {
        title: '캘리브레이션 (Calibration)',
        description: '카메라 캘리브레이션 및 Machine Stage 캘리브레이션 구현',
        details: [
          '카메라로 체스보드를 여러번 촬영하여 캘리브레이션 수행',
          'OpenCV의 캘리브레이션 함수 사용',
          'Machine Stage와 카메라 간의 좌표계 변환 매트릭스 계산'
        ]
      },
      {
        title: '정렬 (Alignment)',
        description: 'UVW 계산 및 패널 정렬 시스템 구현',
        details: [
          'UVW 값 계산 알고리즘 구현',
          'Machine Stage 제어를 통한 패널 정렬',
          '정렬 결과 검증 시스템 구현'
        ]
      }
    ],
    results: [
      'LCD 패널 정렬 정확도 향상',
      '공장 생산 자동화 구축 성공',
      '템플릿 매칭 속도 개선으로 생산성 향상'
    ],
    challenges: [
      {
        problem: '불분명한 템플릿 마커로 인한 매칭 오류',
        solution: '이미지 프로세싱 기법을 활용하여 마커 보정 및 매칭 알고리즘 개선'
      },
      {
        problem: '매칭 속도 문제',
        solution: 'CUDA 프로그래밍을 활용한 병렬 처리로 매칭 속도 개선'
      },
      {
        problem: '카메라와 Machine Stage 간 좌표계 변환 오차',
        solution: '정밀한 캘리브레이션 방법 개발 및 오차 보정 알고리즘 구현'
      }
    ]
  },
  'robot-arm': {
    overview: '창고 자동화 시스템을 위한 3D 비전 기반 로봇 팔 제어 연구',
    teamSize: 3,
    role: [
      '3D 비전 시스템 개발',
      '로봇 팔 제어 알고리즘 구현',
      '물체 인식 및 그래스핑 시스템 개발'
    ],
    technologies: ['Python', 'ROS', 'OpenCV', 'PCL'],
    process: [
      {
        title: '3D 비전 시스템',
        description: '스테레오 카메라 및 깊이 센서를 활용한 3D 비전 시스템 구현',
        details: [
          '스테레오 카메라 캘리브레이션',
          '깊이 맵 생성 및 포인트 클라우드 처리',
          '물체 인식 및 위치 추정'
        ]
      },
      {
        title: '로봇 팔 제어',
        description: 'ROS 기반 로봇 팔 제어 시스템 구현',
        details: [
          '역기구학 알고리즘 구현',
          '경로 계획 및 충돌 회피',
          '그래스핑 전략 개발'
        ]
      }
    ],
    results: [
      '창고 환경에서 다양한 물체 인식 및 그래스핑 성공률 향상',
      '자동화 시스템 프로토타입 개발 완료',
      '연구 논문 발표'
    ],
    challenges: [
      {
        problem: '다양한 형태의 물체 인식 및 그래스핑 어려움',
        solution: '딥러닝 기반 물체 인식 및 포즈 추정 알고리즘 개발'
      },
      {
        problem: '실시간 처리 속도 문제',
        solution: '알고리즘 최적화 및 병렬 처리 기법 적용'
      }
    ]
  },
  'nutrition-site': {
    overview: '영양 정보를 검색할 수 있는 웹사이트 개발',
    teamSize: 2,
    role: [
      '서버 구축',
      'LLM 처리 알고리즘 개발'
    ],
    technologies: ['LLM', 'JavaScript', 'Next.js', 'MariaDB'],
    process: [
      {
        title: '웹사이트 개발',
        description: 'Next.js 백엔드 및 프론트엔드 개발',
        details: [
          '반응형 웹 디자인 구현',
          'REST API 개발'
        ]
      }
    ],
    results: [
      '사용자 친화적인 영양 정보 검색 웹사이트 개발 완료 (https://fodoit.com)',
      '다양한 식품의 영양 정보 제공',
      '개인 프로젝트로 기술 역량 향상'
    ],
    challenges: [
      {
        problem: '개인화된 검색 결과 필요',
        solution: '개인마다 다른 조건을 처리하기 위해 Gemini API(LLM)을 활용하고 프롬프트 엔지니어링으로 최적화'
      }
    ]
  }
};
