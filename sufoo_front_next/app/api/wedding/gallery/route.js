import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// 파일 경로 설정
const GALLERY_FILE = path.join(process.cwd(), 'data', 'gallery.json')

// 기본 갤러리 데이터
const defaultGalleryData = {
  settings: {
    gridCols: 3,
    gridRows: 2,
    maxVisible: 6,
    randomOrder: false
  },
  images: [
    { 
      id: 1, 
      src: "https://via.placeholder.com/300x200/d4af37/ffffff?text=Wedding+1", 
      alt: "웨딩 사진 1",
      gridSize: { width: 1, height: 1 },
      position: { x: 0, y: 0 }
    },
    { 
      id: 2, 
      src: "https://via.placeholder.com/300x200/f8f6f0/2c2c2c?text=Wedding+2", 
      alt: "웨딩 사진 2",
      gridSize: { width: 1, height: 1 },
      position: { x: 1, y: 0 }
    },
    { 
      id: 3, 
      src: "https://via.placeholder.com/300x200/8b4513/ffffff?text=Wedding+3", 
      alt: "웨딩 사진 3",
      gridSize: { width: 1, height: 1 },
      position: { x: 2, y: 0 }
    },
    { 
      id: 4, 
      src: "https://via.placeholder.com/300x200/d4af37/ffffff?text=Wedding+4", 
      alt: "웨딩 사진 4",
      gridSize: { width: 1, height: 1 },
      position: { x: 0, y: 1 }
    },
    { 
      id: 5, 
      src: "https://via.placeholder.com/300x200/f8f6f0/2c2c2c?text=Wedding+5", 
      alt: "웨딩 사진 5",
      gridSize: { width: 1, height: 1 },
      position: { x: 1, y: 1 }
    },
    { 
      id: 6, 
      src: "https://via.placeholder.com/300x200/8b4513/ffffff?text=Wedding+6", 
      alt: "웨딩 사진 6",
      gridSize: { width: 1, height: 1 },
      position: { x: 2, y: 1 }
    }
  ]
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

// 갤러리 데이터 읽기
function readGallery() {
  try {
    if (fs.existsSync(GALLERY_FILE)) {
      const data = fs.readFileSync(GALLERY_FILE, 'utf8')
      return JSON.parse(data)
    }
    return defaultGalleryData
  } catch (error) {
    console.error('갤러리 읽기 오류:', error)
    return defaultGalleryData
  }
}

// 갤러리 데이터 쓰기
function writeGallery(data) {
  try {
    ensureDataDir()
    fs.writeFileSync(GALLERY_FILE, JSON.stringify(data, null, 2), 'utf8')
  } catch (error) {
    console.error('갤러리 쓰기 오류:', error)
    throw error
  }
}

// GET: 갤러리 조회
export async function GET() {
  try {
    const gallery = await withMutex('gallery', () => {
      return readGallery()
    })
    
    return NextResponse.json({
      success: true,
      data: gallery
    })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: '갤러리를 불러오는데 실패했습니다.' },
      { status: 500 }
    )
  }
}

// POST: 갤러리 저장
export async function POST(request) {
  try {
    const newGallery = await request.json()
    
    // 입력 검증
    if (!newGallery.settings || !newGallery.images) {
      return NextResponse.json(
        { success: false, error: '갤러리 데이터가 올바르지 않습니다.' },
        { status: 400 }
      )
    }
    
    const savedGallery = await withMutex('gallery', () => {
      const galleryToSave = {
        ...newGallery,
        updatedAt: new Date().toISOString()
      }
      writeGallery(galleryToSave)
      return galleryToSave
    })
    
    return NextResponse.json({
      success: true,
      message: '갤러리가 저장되었습니다.',
      data: savedGallery
    })
  } catch (error) {
    console.error('갤러리 저장 오류:', error)
    return NextResponse.json(
      { success: false, error: '갤러리 저장에 실패했습니다.' },
      { status: 500 }
    )
  }
} 