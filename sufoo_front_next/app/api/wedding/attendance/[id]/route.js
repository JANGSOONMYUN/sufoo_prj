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
    const dataDir = path.join(process.cwd(), 'data')
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true })
    }
    fs.writeFileSync(ATTENDANCE_FILE, JSON.stringify(data, null, 2), 'utf8')
  } catch (error) {
    console.error('참석 여부 쓰기 오류:', error)
    throw error
  }
}

// DELETE: 참석 여부 항목 삭제
export async function DELETE(request, { params }) {
  try {
    const { id } = params
    
    if (!id) {
      return NextResponse.json(
        { success: false, error: 'ID가 필요합니다.' },
        { status: 400 }
      )
    }
    
    const updatedAttendance = await withMutex('attendance', () => {
      const currentAttendance = readAttendance()
      const updated = currentAttendance.filter(entry => entry.id !== parseInt(id))
      writeAttendance(updated)
      return updated
    })
    
    return NextResponse.json({
      success: true,
      message: '참석 여부 항목이 삭제되었습니다.'
    })
  } catch (error) {
    console.error('참석 여부 삭제 오류:', error)
    return NextResponse.json(
      { success: false, error: '삭제에 실패했습니다.' },
      { status: 500 }
    )
  }
} 