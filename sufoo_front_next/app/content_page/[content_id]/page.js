// app/content_page/[content_id]/page.js

'use client';

import { useSearchParams } from 'next/navigation';
import { useState, useEffect } from 'react';

export default function Page({ params }) {
  const { content_id } = params;
  const searchParams = useSearchParams();
  
  const userId = searchParams.get('userId');
  const sessionId = searchParams.get('sessionId');
  const relative_url = searchParams.get('relative_url');
  const llmJsonData = JSON.parse(searchParams.get('llmJsonData') || '{}');

  const [responseData, setResponseData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [noData, setNoData] = useState(false); // 자료 없음 상태 추가

  useEffect(() => {
    // console.log('userId:', userId);
    // console.log('sessionId:', sessionId);
    // console.log('relative_url:', relative_url);
    // console.log('llmJsonData:', llmJsonData);

    if (!userId) {
      setNoData(true); // 자료 없음 상태 설정
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      try {
        console.log('Sending data:', llmJsonData); // 전송 데이터 로그
        const response = await fetch('http://jsm0803.iptime.org:20000/llm', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ content: JSON.stringify(llmJsonData) }), // llmJsonData를 문자열로 변환하여 전송
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
        setLoading(false);
      }
    };

    fetchData();
  }, []); // 페이지 로드 시 한 번 실행

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      {noData ? (
        <div style={{ textAlign: 'center' }}>자료 없음</div>
      ) : (
        <>
          <div style={{ textAlign: 'center', marginBottom: '20px' }}>
            <h2>Question:</h2>
            <p>{llmJsonData.question}</p>
          </div>
          {loading ? (
            <div style={{ textAlign: 'center' }}>Loading...</div>
          ) : (
            responseData && (
              <div style={{ border: '1px solid #ccc', padding: '10px', borderRadius: '5px', overflowX: 'auto' }}>
                <h3>Response Data:</h3>
                <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>{JSON.stringify(responseData, null, 2)}</pre>
              </div>
            )
          )}
        </>
      )}
    </div>
  );
}
