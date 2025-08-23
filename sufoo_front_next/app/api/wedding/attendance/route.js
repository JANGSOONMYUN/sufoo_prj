import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// 파일 경로 설정
const ATTENDANCE_FILE = path.join(process.cwd(), 'data', 'attendance.json')

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

// 참석 데이터 읽기
function readAttendance() {
  try {
    if (fs.existsSync(ATTENDANCE_FILE)) {
      const data = fs.readFileSync(ATTENDANCE_FILE, 'utf8')
      return JSON.parse(data)
    }
    return []
  } catch (error) {
    console.error('참석 여부 읽기 오류:', error)
    return []
  }
}

// 참석 데이터 쓰기
function writeAttendance(data) {
  try {
    ensureDataDir()
    fs.writeFileSync(ATTENDANCE_FILE, JSON.stringify(data, null, 2), 'utf8')
  } catch (error) {
    console.error('참석 여부 쓰기 오류:', error)
    throw error
  }
}

// GET: 참석 여부 조회
export async function GET() {
  try {
    const attendance = await withMutex('attendance', () => {
      return readAttendance()
    })
    
    // 통계 계산
    const stats = {
      totalAttending: attendance.filter(a => a.attending === 'yes').length,
      totalNotAttending: attendance.filter(a => a.attending === 'no').length,
      totalGuestCount: attendance
        .filter(a => a.attending === 'yes')
        .reduce((sum, a) => sum + (a.guestCount || 1), 0),
      total: attendance.length
    }
    
    return NextResponse.json({
      success: true,
      data: attendance.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)),
      stats
    })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: '참석 여부를 불러오는데 실패했습니다.' },
      { status: 500 }
    )
  }
}

// POST: 참석 여부 추가/수정
export async function POST(request) {
  try {
    const { name, phone, attending, guestCount } = await request.json()
    
    // 입력 검증
    if (!name || !phone || !attending) {
      return NextResponse.json(
        { success: false, error: '이름, 연락처, 참석 여부를 모두 입력해주세요.' },
        { status: 400 }
      )
    }
    
    if (name.length > 50 || phone.length > 20) {
      return NextResponse.json(
        { success: false, error: '이름은 50자, 연락처는 20자 이내로 입력해주세요.' },
        { status: 400 }
      )
    }
    
    if (!['yes', 'no'].includes(attending)) {
      return NextResponse.json(
        { success: false, error: '참석 여부는 yes 또는 no만 가능합니다.' },
        { status: 400 }
      )
    }
    
    const finalGuestCount = attending === 'yes' ? (guestCount || 1) : 0
    
    if (finalGuestCount > 10) {
      return NextResponse.json(
        { success: false, error: '참석 인원은 최대 10명까지 입력 가능합니다.' },
        { status: 400 }
      )
    }
    
    const newEntry = {
      id: Date.now(),
      name: name.trim(),
      phone: phone.trim(),
      attending,
      guestCount: finalGuestCount,
      createdAt: new Date().toISOString(),
      date: new Date().toLocaleDateString('ko-KR')
    }
    
    const updatedAttendance = await withMutex('attendance', () => {
      const currentAttendance = readAttendance()
      
      // 같은 연락처나 이름으로 이미 등록된 경우 업데이트
      const existingIndex = currentAttendance.findIndex(
        a => a.phone === phone.trim() || a.name === name.trim()
      )
      
      let updated
      if (existingIndex !== -1) {
        // 기존 항목 업데이트
        updated = [...currentAttendance]
        updated[existingIndex] = { ...updated[existingIndex], ...newEntry }
      } else {
        // 새 항목 추가
        updated = [...currentAttendance, newEntry]
      }
      
      writeAttendance(updated)
      return updated
    })
    
    return NextResponse.json({
      success: true,
      message: '참석 여부가 등록되었습니다.',
      data: newEntry
    })
  } catch (error) {
    console.error('참석 여부 추가 오류:', error)
    return NextResponse.json(
      { success: false, error: '참석 여부 등록에 실패했습니다.' },
      { status: 500 }
    )
  }
} 