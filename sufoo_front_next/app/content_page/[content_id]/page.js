'use client';

import { useSearchParams } from 'next/navigation';
import { useState, useEffect, useRef } from 'react';
import LoadingScreen from '@/app/ui_loading/page';
import ReactMarkdown from 'react-markdown'
import '@/styles/contentPage.css';
import Link from 'next/link'; // Link 컴포넌트 import
import {
  applyLlmStreamPatch,
  buildInitialLlmViewData,
  deepClone,
  normalizeLlmResultForView,
  postSSE,
} from '@/lib/llmStreaming';

// import { FaBars } from 'react-icons/fa'; // Font Awesome 아이콘 import
import { FaBars, FaShareAlt } from 'react-icons/fa'; // Font Awesome 아이콘 import

export default function Page({ params }) {
  const { content_id } = params;
  const searchParams = useSearchParams();

  const userId = searchParams.get('userId');
  const sessionId = searchParams.get('sessionId');
  const relative_url = content_id //searchParams.get('relative_url');
  const llmJsonData = useRef(JSON.parse(searchParams.get('llmJsonData') || '{}'));

  const [responseData, setResponseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [noData, setNoData] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  const isFetching = useRef(false);
  const streamAbortRef = useRef(null);
  const streamSeqRef = useRef(0);
  const streamDataRef = useRef(null);
  const rafScheduledRef = useRef(false);
  const hasInsertedRef = useRef(false);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    const startTotal = performance.now();

    if (!userId && !relative_url) {
      setNoData(true);
      setLoading(false);
      return;
    }
    //let isFetching = false; // fetch 중복 방지 플래그

    const handleScroll = () => {
      setShowScrollButton(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);

    const fetchData = async () => {
      const startFetch = performance.now();
      if (isFetching.current) return; // 이미 fetch 중이면 실행하지 않음
      isFetching.current = true;
      hasInsertedRef.current = false;

      setLoading(true);
      try {
        const startApiCall = performance.now();

        let existingData = null;

        // 1. 기존 데이터 확인
        if (relative_url) {
          const checkResponse = await fetch(`/api/select_contents?url=${relative_url}`, {
            method: 'GET',
          });
           if (checkResponse.ok) {
              const result = await checkResponse.json();
              if(result && result.pageInfo && result.pageInfo.content_data){
                  console.log("기존 데이터 있음");
                 const contentData = normalizeLlmResultForView(JSON.parse(result.pageInfo.content_data));
                 result.include_images = contentData?.include_images;
                 setResponseData(result);
                 setNoData(false);
                 existingData = true;
              }
           }
          console.log(`기존 데이터 확인: ${performance.now() - startApiCall}ms`);
        }

        // 2. 기존 데이터가 없으면 API 호출
        if (!existingData && userId) {
          console.log("기존 데이터 없음");
          const controller = new AbortController();
          streamAbortRef.current?.abort();
          streamAbortRef.current = controller;
          const mySeq = ++streamSeqRef.current;

          const scheduleUiUpdate = () => {
            if (rafScheduledRef.current) return;
            rafScheduledRef.current = true;
            requestAnimationFrame(() => {
              rafScheduledRef.current = false;
              if (!streamDataRef.current) return;
              setResponseData(deepClone(streamDataRef.current));
            });
          };

          // 스트리밍 도중에도 현재 UI 레이아웃(폰트/위치) 그대로 렌더링되도록 초기 템플릿을 세팅
          streamDataRef.current = buildInitialLlmViewData(llmJsonData.current);
          setResponseData(deepClone(streamDataRef.current));
          setNoData(false);
          setIsStreaming(true);
          setLoading(false); // 전체 로딩 화면 대신 "실시간" 렌더링 시작

          const handleStreamEnd = async (finalDataStr) => {
            const parsed = normalizeLlmResultForView(JSON.parse(finalDataStr));

            streamDataRef.current = parsed;
            setResponseData(deepClone(parsed));
            setIsStreaming(false);

            if (hasInsertedRef.current) return;
            hasInsertedRef.current = true;

            // 최종 결과만 DB에 저장
            await fetch('/api/insert_contents', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                userId: String(userId),
                searchWords: llmJsonData.current.question,
                advertiseInfo: "",
                data: JSON.stringify(parsed),
                url: String(relative_url),
              }),
            });
          };

          await postSSE(
            'https://fodoit.com:20000/llm_ver3',
            { content: JSON.stringify(llmJsonData.current) },
            ({ event, data }) => {
              if (streamSeqRef.current !== mySeq) return;

              if (event === 'chunk') {
                try {
                  const patch = JSON.parse(data);
                  applyLlmStreamPatch(streamDataRef.current, patch);
                  scheduleUiUpdate();
                } catch (e) {
                  console.warn('chunk JSON 파싱 실패:', e, data);
                }
              } else if (event === 'end') {
                // async 저장 로직은 별도 처리(미-await) + 에러는 내부에서 캐치
                handleStreamEnd(data).catch((e) => {
                  console.error('end 처리 실패:', e);
                  setResponseData({ error: e.message });
                  setIsStreaming(false);
                });
              } else if (event === 'error') {
                setResponseData({ error: data || 'LLM 스트리밍 오류' });
                setIsStreaming(false);
              }
            },
            { signal: controller.signal }
          );
        }

        const endApiCall = performance.now();
        console.log(`Total api call duration: ${endApiCall - startApiCall}ms`);


      } catch (error) {
        // AbortError는 화면 에러로 취급하지 않음(페이지 이동/언마운트 시 정상)
        if (error?.name === 'AbortError') return;
        console.error('Error:', error);
        setResponseData({ error: error.message });
        setIsStreaming(false);
      } finally {
        isFetching.current = false; // fetch 완료 후 플래그 초기화
        // 스트리밍 중에는 이미 loading=false로 전환했으므로, 여기서 강제로 덮어도 무방
        setLoading(false);
        const endFetch = performance.now();
        console.log(`Total fetch duration: ${endFetch - startFetch}ms`);
      }
    };

    fetchData();

    return () => {
      window.removeEventListener('scroll',handleScroll);
      streamAbortRef.current?.abort();
      const endTotal = performance.now();
      console.log(`Total effect duration: ${endTotal - startTotal}ms`);
    };
  }, [userId, relative_url]); // userId와 relative_url을 의존성으로 추가

  // 공유 버튼 컴포넌트
  const ShareButton = ({ url, showIcon, showText, iconColor, textColor }) => {
    const handleClick = (e) => {
      e.preventDefault();
      const shareUrl = `https://fodoit.com/content_page/${url}`;
      const shareData = {
        title: '당신만을 위한 영양 검색',
        text: '나에게 필요한 영양 정보, 더 이상 헤매지 마세요! 검색하세요.',
        url: shareUrl,
      };
  
      if (navigator.share) {
        navigator.share(shareData)
          .then(() => console.log('공유가 성공적으로 완료되었습니다.'))
          .catch((error) => console.error('공유에 실패했습니다:', error));
      } else {
        alert('이 기능은 현재 브라우저에서 지원되지 않습니다.');
      }
    };
  
    return (
      <button
        onClick={handleClick}
        className="px-2 py-1 rounded max-w-xs flex items-center justify-center"
      >
        {showIcon && <FaShareAlt style={{ color: iconColor || 'white', marginRight: '8px', fontSize: '1.2em' }} />}
        {showText && <span style={{ color: textColor || 'white' }}>공유하기</span>}
      </button>
    );
  };

  // 데이터를 렌더링하는 함수
  const renderContent = () => {
    const includeImages = responseData?.include_images?.request;

    // includeImages가 배열이 아닌 경우 예외 처리
    if (!Array.isArray(includeImages)) {
      console.log('Invalid includeImages:', includeImages); // 디버깅용 로그
      return <p>유효한 데이터를 찾을 수 없습니다.</p>;
    }

    // 데이터를 순회하며 렌더링    
    return includeImages.map((item, index) => (
      // <div key={index} style={{ marginBottom: '20px', padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}>
      <div key={index} className="space-y-3 bg-white p-4 rounded-lg shadow" style={{ marginBottom: '20px' }} // 여기에 스타일 추가
    >
        {/* 제목 */}
        <h2 className="text-xl font-semibold">{item.title}</h2>

        {/* 설명 */}
        <div className="text-sm space-y-2 content-style">
          <ReactMarkdown>{item.description}</ReactMarkdown>
        </div>

        {/* 결과 */}
        <div className="text-sm space-y-2 content-style">
          <ReactMarkdown>{item.result}</ReactMarkdown>
        </div>

        {/* 대표 이미지 */}
        {item.image_url && (
      <div style={{ overflow: 'hidden', width: '100%', height: 'auto', aspectRatio: '7 / 4' }}> {/* 이미지 컨테이너 */}
        <img
          src={item.image_url}
          alt={item.representative_image_name || '이미지'}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',  // 이미지가 컨테이너에 꽉 차도록 조정
            objectPosition: 'center center',  // 이미지를 가운데 기준으로 자르기
            borderRadius: '8px',
            marginTop: '10px'
          }}
        />
      </div>
      )}

        {/* 상세 항목 (subject) */}
        {item.subject?.map((subjectItem, subIndex) => (
          <div key={subIndex} style={{ marginTop: '10px' }}>
            <h3 className ="space-y-3 text-lg font-semibold">{subjectItem.sub_title}</h3>
            <div className="text-sm space-y-2 content-style">
              <ReactMarkdown>{subjectItem.sub_description}</ReactMarkdown>
            </div>
            <div className="text-sm space-y-2 content-style">
              <ReactMarkdown>{subjectItem.sub_result}</ReactMarkdown>
            </div>
          </div>
        ))}
      </div>
    ));
  };

  if (loading) {
    return <LoadingScreen question={llmJsonData.current.question} />;
  }

  if (noData) {
    return <div style={{ textAlign: 'center' }}>자료 없음</div>;
  }

  return (
    <div>
      {/* 상단 바 */}
      <div className="bg-[#83AE4C] h-16 flex items-center justify-between px-4">
         {/* 홈 아이콘 */}
        <Link href="/">
          <img src="/logo/fodoit_title_white.png" alt="홈으로 이동" style={{ height: '30px', width: 'auto' }} />
        </Link>
        {/* 메뉴 버튼 */}
        {/* <button className="text-2xl text-white hover:text-gray-200">
          <FaBars style={{ color: 'white' }} />
        </button> */}
        {/* 상단 공유 버튼 */}
        <ShareButton className="flex justify-end"
          url={relative_url}
          showIcon={true}
          showText={false}
          iconColor="white"
          textColor="white"
        />
      </div>
      {/* 페이지 내용 */}
      <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
        {isStreaming && (
          <div className="mb-3 text-sm text-gray-500">
            실시간으로 생성 중입니다...
          </div>
        )}
        {responseData?.error && (
          <div className="mb-3 text-sm text-red-600">
            오류: {responseData.error}
          </div>
        )}
        {renderContent()}

        {/* 공유 섹션 */}
        {responseData && responseData.include_images && (
          <>
            <div id="kakaotalk-sharing-btn"></div>
            <div className="flex justify-center my-4">
              <ShareButton
                url={relative_url}
                showIcon={true}
                showText={true}
                iconColor="black"
                textColor="black"
              />
            </div>
          </>
        )}
        {showScrollButton && (
          <button
            onClick={scrollToTop}
            className="fixed bottom-4 right-4 bg-blue-500 text-white p-4 rounded-full shadow hover:bg-blue-600 transition duration-300 text-2xl font-bold"
          >
          </button>
        )}
      </div>
    </div>
  );
}