import { NextResponse } from 'next/server'
import { writeFile } from 'fs/promises'
import path from 'path'
import fs from 'fs'

export async function POST(request) {
  try {
    console.log('=== 파일 업로드 API 시작 ===')
    
    // 기본 테스트
    const contentType = request.headers.get('content-type')
    console.log('Content-Type:', contentType)
    
    if (!contentType || !contentType.includes('multipart/form-data')) {
      console.log('잘못된 Content-Type')
      return NextResponse.json({
        success: false,
        error: 'multipart/form-data가 아닙니다.'
      }, { status: 400 })
    }
    
    const formData = await request.formData()
    console.log('FormData 파싱 완료')
    
    const files = formData.getAll('files')
    console.log('받은 파일 개수:', files.length)
    
    if (!files || files.length === 0) {
      console.log('파일이 없습니다.')
      return NextResponse.json({
        success: false,
        error: '업로드할 파일이 없습니다.'
      }, { status: 400 })
    }

    // 첫 번째 파일만 테스트로 처리
    const file = files[0]
    console.log('첫 번째 파일:', {
      name: file.name,
      size: file.size,
      type: file.type
    })

    // 파일 크기 체크 (5MB 제한)
    const maxSize = 5 * 1024 * 1024
    if (file.size > maxSize) {
      return NextResponse.json({
        success: false,
        error: '파일 크기가 너무 큽니다. (최대 5MB)'
      }, { status: 400 })
    }

    // 파일 타입 체크
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json({
        success: false,
        error: `지원되지 않는 파일 형식입니다: ${file.type}`
      }, { status: 400 })
    }

    // 업로드 디렉토리 생성
    const uploadDir = path.join(process.cwd(), 'public', 'wedding')
    console.log('업로드 디렉토리:', uploadDir)
    
    if (!fs.existsSync(uploadDir)) {
      console.log('디렉토리 생성 중...')
      fs.mkdirSync(uploadDir, { recursive: true })
    }

    // 파일명 생성
    const timestamp = Date.now()
    const fileName = `test_${timestamp}_${file.name}`
    const filePath = path.join(uploadDir, fileName)
    
    console.log('저장할 파일 경로:', filePath)

    // 파일 저장
    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)
    
    console.log('파일 저장 중... 크기:', buffer.length)
    
    await writeFile(filePath, buffer)
    
    console.log('파일 저장 완료')
    
    // 파일이 실제로 저장되었는지 확인
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath)
      console.log('저장된 파일 통계:', {
        size: stats.size,
        created: stats.birthtime
      })
      
      return NextResponse.json({
        success: true,
        message: '파일이 성공적으로 업로드되었습니다.',
        files: [{
          originalName: file.name,
          fileName: fileName,
          url: `/wedding/${fileName}`,
          size: file.size,
          type: file.type
        }]
      })
    } else {
      throw new Error('파일 저장 후 확인 실패')
    }

  } catch (error) {
    console.error('=== 업로드 오류 ===')
    console.error('오류 타입:', error.constructor.name)
    console.error('오류 메시지:', error.message)
    console.error('스택 트레이스:', error.stack)
    
    return NextResponse.json({
      success: false,
      error: `업로드 실패: ${error.message}`
    }, { status: 500 })
  }
} 