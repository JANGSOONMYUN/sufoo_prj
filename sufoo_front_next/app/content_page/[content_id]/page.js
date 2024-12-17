'use client';

import { useSearchParams } from 'next/navigation';
import { useState, useEffect, useRef } from 'react';
import LoadingScreen from '@/app/ui_loading/page'; // ui_loading 컴포넌트 import

export default function Page({ params }) {
  const { content_id } = params;
  const searchParams = useSearchParams();

  const userId = searchParams.get('userId');
  const sessionId = searchParams.get('sessionId');
  const relative_url = searchParams.get('relative_url');
  const llmJsonData = useRef(JSON.parse(searchParams.get('llmJsonData') || '{}'));

  const [responseData, setResponseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [noData, setNoData] = useState(false);

  useEffect(() => {
    if (!userId) {
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
      } catch (error) {
        console.error('Error:', error);
        setResponseData({ error: error.message });
      } finally {
        isFetching = false; // fetch 완료 후 플래그 초기화
        setLoading(false);
      }
    };

    fetchData();
  }, [userId]); // userId만 의존성으로 설정

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
        <p>{item.description}</p>

        {/* 결과 */}
        <p>{item.result}</p>

        {/* 대표 이미지 */}
        {item.image_url && (
          <img
            src={item.image_url}
            alt={item.representative_image_name || '이미지'}
            style={{ maxWidth: '100%', borderRadius: '8px', marginTop: '10px' }}
          />
        )}

        {/* 상세 항목 (subject) */}
        {item.subject?.map((subjectItem, subIndex) => (
          <div key={subIndex} style={{ marginTop: '10px' }}>
            <h3>{subjectItem.sub_title}</h3>
            <p>{subjectItem.sub_description}</p>
            <p>{subjectItem.sub_result}</p>
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
