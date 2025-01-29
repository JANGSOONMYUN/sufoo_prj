// "use client";

// import React, { useEffect } from 'react';

// export default function KakaoShareButton() {
//   useEffect(() => {
//     const script = document.createElement('script');
//     script.src = "https://t1.kakaocdn.net/kakao_js_sdk/2.7.4/kakao.min.js";
//     script.integrity = "sha384-DKYJZ8NLiK8MN4/C5P2dtSmLQ4KwPaoqAfyA/DfmEc1VDxu4yyC7wy6K1Hs90nka";
//     script.crossOrigin = "anonymous";
//     script.onload = () => {
//       if (window.Kakao) {
//         window.Kakao.init('c2837fa943b022f477e2fcd2a696ddd4'); // Replace with your actual JavaScript key
//         window.Kakao.Share.createScrapButton({
//           container: '#kakaotalk-sharing-btn',
//           requestUrl: 'https://fodoit.com',
//         });
//       }
//     };
//     document.head.appendChild(script);
//   }, []);

//   const handleClick = (e) => {
//     e.preventDefault(); // 기본 동작 방지
//     // Kakao 공유 기능 호출
//     if (window.Kakao) {
//       window.Kakao.Share.sendDefault({
//         objectType: 'feed',
//         content: {
//           title: 'Title',
//           description: 'Description',
//           imageUrl: 'https://developers.kakao.com/assets/img/about/logos/kakaotalksharing/kakaotalk_sharing_btn_medium.png',
//           link: {
//             mobileWebUrl: 'https://fodoit.com',
//             webUrl: 'https://fodoit.com',
//           },
//         },
//       });
//     }
//   };

//   return (
//     <button id="kakaotalk-sharing-btn" onClick={handleClick}>
//     <img src="https://developers.kakao.com/assets/img/about/logos/kakaotalksharing/kakaotalk_sharing_btn_medium.png" alt="카카오톡 공유 보내기 버튼" />
//   </button>

//   );
// }

"use client";

import React from 'react';

export default function ShareButton() {
  const handleClick = (e) => {
    e.preventDefault(); // 기본 동작 방지

    if (navigator.share) {
      navigator.share({
        title: 'Title',
        text: 'Description',
        url: 'https://fodoit.com',
      })
      .then(() => console.log('공유가 성공적으로 완료되었습니다.'))
      .catch((error) => console.error('공유에 실패했습니다:', error));
    } else {
      alert('이 기능은 현재 브라우저에서 지원되지 않습니다.');
    }
  };

  return (
    <button onClick={handleClick}>
      공유하기
    </button>
  );
}