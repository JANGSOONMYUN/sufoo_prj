// // app/page.jsx
// 'use client';

// import { useEffect } from 'react';
// import { useRouter } from 'next/navigation';

// const Home = () => {
//   const router = useRouter();

//   useEffect(() => {
//     // 페이지 로딩 시 ui_insert_user_data로 리디렉션
//     router.push('/ui_insert_user_data');
//   }, [router]);

//   return null;  // 리디렉션 처리 후 렌더링은 하지 않음
// };

// export default Home;


// app/page.jsx
import React from 'react';
import UiInsertUserData from './ui_insert_user_data/page'; // ui_insert_user_data의 page.jsx 임포트

const Home = () => {
  return (
    <div className="App">
      <UiInsertUserData /> {/* 메인 페이지로 ui_insert_user_data의 내용을 바로 표시 */}
    </div>
  );
};

export default Home;
