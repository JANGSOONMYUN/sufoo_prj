"use client";

import React, { useEffect, useState } from 'react';
import '@/styles/loadingScreen.css';

export default function LoadingScreen() {
  const [dots, setDots] = useState('');
  const [question, setQuestion] = useState('');

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    setQuestion(searchParams.get('question'));

    const interval = setInterval(() => {
      setDots(prev => prev.length < 3 ? prev + '.' : '');
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loading-container">
      <h1>생성중{dots}</h1>
      {/* {question && (
        <div className="question-container">
          <h2>Question:</h2>
          <p>{question}</p>
        </div>
      )} */}
      <div className="loading-card">
        <div className="loading-line"></div>
        <div className="loading-line short"></div>
        <div className="loading-line medium"></div>
        <div className="loading-line long"></div>
        <div className="loading-block"></div>
      </div>
    </div>
  );
}