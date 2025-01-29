
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
