import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// 파일 경로 설정
const CONFIG_FILE = path.join(process.cwd(), 'data', 'wedding-config.json')

// 기본 웨딩 데이터
const defaultWeddingData = {
  couple: {
    groom: {
      name: "김민수",
      phone: "010-1234-5678",
      father: "김아버지",
      mother: "김어머니",
      account: "신한은행 110-123-456789"
    },
    bride: {
      name: "이지영",
      phone: "010-9876-5432",
      father: "이아버지",
      mother: "이어머니",
      account: "국민은행 123-456-789012"
    }
  },
  wedding: {
    date: "2024년 12월 14일 토요일",
    time: "오후 2시",
    venue: "그랜드 웨딩홀",
    address: "서울시 강남구 테헤란로 123",
    floor: "3층 그랜드홀",
    phone: "02-1234-5678"
  },
  message: "소중한 분들을 모시고\n저희 두 사람의 새로운 시작을\n함께 축복해 주시기 바랍니다.\n\n여러분의 따뜻한 마음과 축복이\n저희에게는 가장 큰 선물입니다."
}

// 간단한 뮤텍스 구현 (동시성 보장)
const mutexes = new Map()

async function withMutex(key, fn) {
  if (mutexes.has(key)) {
    await mutexes.get(key)
  }
  
  const promise = fn()
  mutexes.set(key, promise)
  
  try {
    return await promise
  } finally {
    mutexes.delete(key)
  }
}

// 데이터 디렉토리 생성
function ensureDataDir() {
  const dataDir = path.join(process.cwd(), 'data')
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true })
  }
}

// 웨딩 설정 데이터 읽기
function readWeddingConfig() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      const data = fs.readFileSync(CONFIG_FILE, 'utf8')
      return JSON.parse(data)
    }
    return defaultWeddingData
  } catch (error) {
    console.error('웨딩 설정 읽기 오류:', error)
    return defaultWeddingData
  }
}

// 웨딩 설정 데이터 쓰기
function writeWeddingConfig(data) {
  try {
    ensureDataDir()
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(data, null, 2), 'utf8')
  } catch (error) {
    console.error('웨딩 설정 쓰기 오류:', error)
    throw error
  }
}

// GET: 웨딩 설정 조회
export async function GET() {
  try {
    const config = await withMutex('wedding-config', () => {
      return readWeddingConfig()
    })
    
    return NextResponse.json({
      success: true,
      data: config
    })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: '웨딩 설정을 불러오는데 실패했습니다.' },
      { status: 500 }
    )
  }
}

// POST: 웨딩 설정 저장
export async function POST(request) {
  try {
    const newConfig = await request.json()
    
    // 입력 검증 (기본적인 필수 필드 확인)
    if (!newConfig.couple || !newConfig.wedding || !newConfig.message) {
      return NextResponse.json(
        { success: false, error: '필수 정보가 누락되었습니다.' },
        { status: 400 }
      )
    }
    
    const savedConfig = await withMutex('wedding-config', () => {
      const configToSave = {
        ...newConfig,
        updatedAt: new Date().toISOString()
      }
      writeWeddingConfig(configToSave)
      return configToSave
    })
    
    return NextResponse.json({
      success: true,
      message: '웨딩 설정이 저장되었습니다.',
      data: savedConfig
    })
  } catch (error) {
    console.error('웨딩 설정 저장 오류:', error)
    return NextResponse.json(
      { success: false, error: '설정 저장에 실패했습니다.' },
      { status: 500 }
    )
  }
} 