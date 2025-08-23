/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['sharp']
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: '/api/:path*',
      },
    ]
  },
  // 파일 업로드 크기 제한 설정
  api: {
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
  // 이미지 도메인 설정
  images: {
    domains: ['via.placeholder.com'],
  },
}

module.exports = nextConfig 