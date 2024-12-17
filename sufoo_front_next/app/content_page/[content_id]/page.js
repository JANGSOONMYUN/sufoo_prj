'use client';

import { useSearchParams } from 'next/navigation';
import { useState, useEffect, useRef } from 'react';
import LoadingScreen from '@/app/ui_loading/page'; // ui_loading 컴포넌트 import
import ReactMarkdown from 'react-markdown'

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

  useEffect(() => {
    if (!userId && !relative_url) {
      setNoData(true);
      setLoading(false);
      return;
    }
    let isFetching = false; // fetch 중복 방지 플래그

    const fetchData = async () => {
      if (isFetching) return; // 이미 fetch 중이면 실행하지 않음
      isFetching = true;

      setLoading(true);
      try {
        if (!userId && relative_url) {
          const response = await fetch(`/api/select_contents?url=${relative_url}`, {
            method: 'GET',
          });
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const result = await response.json();
          // Parse the content_data and include it in the responseData
          const contentData = JSON.parse(result.pageInfo.content_data); // content_data를 JSON으로 변환
          console.log('Fetched contentData:', contentData);
          result.include_images = contentData.include_images; // include_images를 result에 추가
          setResponseData(result);
          console.log('Fetched result:', result);
          if (result) {
            setNoData(false); // 자료가 있을 경우 noData를 false로 설정
          }
        }
        else {
          console.log('Sending data:', llmJsonData.current);
          const response = await fetch('http://jsm0803.iptime.org:20000/llm', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ content: JSON.stringify(llmJsonData.current) }),
          });
  
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
  
          const result = await response.json();
          setResponseData(result);
  
          // 인서트된 사용자 데이터로 페이지 정보 삽입 요청
          await fetch('/api/insert_contents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              userId: String(userId),
              searchWords: llmJsonData.current.question,
              advertiseInfo: "", // 필요한 광고 정보가 있다면 여기에 추가
              data: JSON.stringify(result), // API 응답 결과를 데이터로 사용
              url: String(relative_url) // relative_url을 사용
            }),
          });
        }
        
      } catch (error) {
        console.error('Error:', error);
        setResponseData({ error: error.message });
      } finally {
        isFetching = false; // fetch 완료 후 플래그 초기화
        setLoading(false);
      }
    };

    fetchData();
  }, [userId, relative_url]); // userId와 relative_url을 의존성으로 추가

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
      <div key={index} className="space-y-2 bg-white p-4 rounded-lg shadow" style={{ marginBottom: '20px' }} // 여기에 스타일 추가
    >
        {/* 제목 */}
        <h2 className="text-3xl font-semibold">{item.title}</h2>

        {/* 설명 */}
        <ReactMarkdown>{item.description}</ReactMarkdown>

        {/* 결과 */}
        <ReactMarkdown>{item.result}</ReactMarkdown>

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
            <h3 className ="text-xl font-semibold">{subjectItem.sub_title}</h3>
            <ReactMarkdown>{subjectItem.sub_description}</ReactMarkdown>
            <ReactMarkdown>{subjectItem.sub_result}</ReactMarkdown>
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
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      {/* <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <h2>Question:</h2>
        <p>{llmJsonData.current.question}</p>
      </div> */}
      {/* 이미지와 세부 데이터 렌더링 */}
      {renderContent()}
    </div>
  );
}
