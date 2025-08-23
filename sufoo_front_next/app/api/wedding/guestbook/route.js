import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// 파일 경로 설정
const GUESTBOOK_FILE = path.join(process.cwd(), 'data', 'guestbook.json')

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

// 방명록 데이터 읽기
function readGuestbook() {
  try {
    if (fs.existsSync(GUESTBOOK_FILE)) {
      const data = fs.readFileSync(GUESTBOOK_FILE, 'utf8')
      return JSON.parse(data)
    }
    return []
  } catch (error) {
    console.error('방명록 읽기 오류:', error)
    return []
  }
}

// 방명록 데이터 쓰기
function writeGuestbook(data) {
  try {
    ensureDataDir()
    fs.writeFileSync(GUESTBOOK_FILE, JSON.stringify(data, null, 2), 'utf8')
  } catch (error) {
    console.error('방명록 쓰기 오류:', error)
    throw error
  }
}

// GET: 방명록 조회
export async function GET() {
  try {
    const guestbook = await withMutex('guestbook', () => {
      return readGuestbook()
    })
    
    return NextResponse.json({
      success: true,
      data: guestbook.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: '방명록을 불러오는데 실패했습니다.' },
      { status: 500 }
    )
  }
}

// POST: 방명록 추가
export async function POST(request) {
  try {
    const { name, message } = await request.json()
    
    // 입력 검증
    if (!name || !message) {
      return NextResponse.json(
        { success: false, error: '이름과 메시지를 모두 입력해주세요.' },
        { status: 400 }
      )
    }
    
    if (name.length > 50 || message.length > 500) {
      return NextResponse.json(
        { success: false, error: '이름은 50자, 메시지는 500자 이내로 입력해주세요.' },
        { status: 400 }
      )
    }
    
    const newEntry = {
      id: Date.now(),
      name: name.trim(),
      message: message.trim(),
      createdAt: new Date().toISOString(),
      date: new Date().toLocaleDateString('ko-KR')
    }
    
    const updatedGuestbook = await withMutex('guestbook', () => {
      const currentGuestbook = readGuestbook()
      const updated = [...currentGuestbook, newEntry]
      writeGuestbook(updated)
      return updated
    })
    
    return NextResponse.json({
      success: true,
      message: '축하 메시지가 등록되었습니다.',
      data: newEntry
    })
  } catch (error) {
    console.error('방명록 추가 오류:', error)
    return NextResponse.json(
      { success: false, error: '메시지 등록에 실패했습니다.' },
      { status: 500 }
    )
  }
} 