import React from 'react';
import '@/styles/loadingScreen.css';



const LoadingScreen = () => {
  return (
    <div className="loading-container">
      <h1>생성중...</h1>
      <div className="loading-card">
        <div className="loading-line"></div>
        <div className="loading-line short"></div>
        <div className="loading-line medium"></div>
        <div className="loading-line long"></div>
        <div className="loading-block"></div>
      </div>
    </div>
  );
};

export default LoadingScreen;
