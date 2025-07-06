export const metadata = {
  title: '김민수 ♥ 이지영 결혼식 - 모바일 청첩장',
  description: '소중한 분들을 모시고 저희 두 사람의 새로운 시작을 함께 축복해 주시기 바랍니다.',
  keywords: '결혼식, 청첩장, 모바일 청첩장, 웨딩, 김민수, 이지영',
  openGraph: {
    title: '김민수 ♥ 이지영 결혼식',
    description: '소중한 분들을 모시고 저희 두 사람의 새로운 시작을 함께 축복해 주시기 바랍니다.',
    type: 'website',
    locale: 'ko_KR',
  },
  robots: {
    index: true,
    follow: true,
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
    userScalable: false,
  },
}

export default function WeddingLayout({ children }) {
  return (
    <div className="min-h-screen bg-gradient-to-b from-wedding-secondary to-wedding-light">
      {children}
    </div>
  )
} 