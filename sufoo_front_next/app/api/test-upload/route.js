import { NextResponse } from 'next/server'

export async function POST(request) {
  try {
    console.log('테스트 업로드 API 시작')
    
    const formData = await request.formData()
    const files = formData.getAll('files')
    
    console.log('받은 파일 개수:', files.length)
    
    if (files.length === 0) {
      return NextResponse.json({
        success: false,
        error: '파일이 없습니다.'
      })
    }
    
    const fileInfo = files.map(file => ({
      name: file.name,
      size: file.size,
      type: file.type
    }))
    
    console.log('파일 정보:', fileInfo)
    
    return NextResponse.json({
      success: true,
      message: '파일 정보를 성공적으로 받았습니다.',
      files: fileInfo
    })
    
  } catch (error) {
    console.error('테스트 업로드 오류:', error)
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 })
  }
} 