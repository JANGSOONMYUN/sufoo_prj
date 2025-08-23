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
    const dataDir = path.join(process.cwd(), 'data')
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true })
    }
    fs.writeFileSync(GUESTBOOK_FILE, JSON.stringify(data, null, 2), 'utf8')
  } catch (error) {
    console.error('방명록 쓰기 오류:', error)
    throw error
  }
}

// DELETE: 방명록 항목 삭제
export async function DELETE(request, { params }) {
  try {
    const { id } = params
    
    if (!id) {
      return NextResponse.json(
        { success: false, error: 'ID가 필요합니다.' },
        { status: 400 }
      )
    }
    
    const updatedGuestbook = await withMutex('guestbook', () => {
      const currentGuestbook = readGuestbook()
      const updated = currentGuestbook.filter(entry => entry.id !== parseInt(id))
      writeGuestbook(updated)
      return updated
    })
    
    return NextResponse.json({
      success: true,
      message: '방명록 항목이 삭제되었습니다.'
    })
  } catch (error) {
    console.error('방명록 삭제 오류:', error)
    return NextResponse.json(
      { success: false, error: '삭제에 실패했습니다.' },
      { status: 500 }
    )
  }
} 