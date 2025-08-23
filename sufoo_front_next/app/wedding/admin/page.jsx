"use client"

import React, { useState, useEffect } from 'react'
import { Eye, EyeOff, Trash2, Edit, Save, X, Users, MessageCircle, Settings, Lock, Image, Plus, Grid, Shuffle, Move, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { GalleryGrid } from '@/components/ui/gallery-grid'

const ADMIN_PASSWORD = '0803' // 초기 비밀번호

function AdminPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [activeTab, setActiveTab] = useState('guestbook')
  const [isLoading, setIsLoading] = useState(false)

  // 데이터 상태
  const [guestbook, setGuestbook] = useState([])
  const [attendance, setAttendance] = useState([])
  const [attendanceStats, setAttendanceStats] = useState({})
  const [weddingConfig, setWeddingConfig] = useState(null)
  const [editingConfig, setEditingConfig] = useState(null)
  const [isEditing, setIsEditing] = useState(false) // 편집 모드 상태를 별도로 관리
  
  // 갤러리 상태
  const [galleryData, setGalleryData] = useState(null)
  const [editingGallery, setEditingGallery] = useState(null)
  const [isEditingGallery, setIsEditingGallery] = useState(false)

  useEffect(() => {
    if (isAuthenticated) {
      loadAllData()
    }
  }, [isAuthenticated])

  // 비밀번호 확인
  const handleLogin = () => {
    if (password === ADMIN_PASSWORD) {
      setIsAuthenticated(true)
      setPassword('')
      loadAllData()
    } else {
      alert('비밀번호가 틀렸습니다.')
    }
  }

  // 모든 데이터 로드
  const loadAllData = async () => {
    await Promise.all([
      loadGuestbook(),
      loadAttendance(),
      loadWeddingConfig(),
      loadGalleryData()
    ])
  }

  // 갤러리 데이터 로드
  const loadGalleryData = async () => {
    try {
      const response = await fetch('/api/wedding/gallery')
      const result = await response.json()
      if (result.success) {
        setGalleryData(result.data)
      }
    } catch (error) {
      console.error('갤러리 로드 오류:', error)
    }
  }

  // 방명록 로드
  const loadGuestbook = async () => {
    try {
      const response = await fetch('/api/wedding/guestbook')
      const result = await response.json()
      if (result.success) {
        setGuestbook(result.data)
      }
    } catch (error) {
      console.error('방명록 로드 오류:', error)
    }
  }

  // 참석 여부 로드
  const loadAttendance = async () => {
    try {
      const response = await fetch('/api/wedding/attendance')
      const result = await response.json()
      if (result.success) {
        setAttendance(result.data)
        setAttendanceStats(result.stats)
      }
    } catch (error) {
      console.error('참석 여부 로드 오류:', error)
    }
  }

  // 웨딩 설정 로드
  const loadWeddingConfig = async () => {
    try {
      const response = await fetch('/api/wedding/config')
      const result = await response.json()
      if (result.success) {
        setWeddingConfig(result.data)
      }
    } catch (error) {
      console.error('웨딩 설정 로드 오류:', error)
    }
  }

  // 방명록 삭제
  const deleteGuestbookEntry = async (id) => {
    if (!confirm('이 메시지를 삭제하시겠습니까?')) return

    setIsLoading(true)
    try {
      const response = await fetch(`/api/wedding/guestbook/${id}`, {
        method: 'DELETE'
      })
      const result = await response.json()
      if (result.success) {
        loadGuestbook()
        alert(result.message)
      } else {
        alert(result.error)
      }
    } catch (error) {
      console.error('방명록 삭제 오류:', error)
      alert('삭제에 실패했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  // 참석 여부 삭제
  const deleteAttendanceEntry = async (id) => {
    if (!confirm('이 참석 정보를 삭제하시겠습니까?')) return

    setIsLoading(true)
    try {
      const response = await fetch(`/api/wedding/attendance/${id}`, {
        method: 'DELETE'
      })
      const result = await response.json()
      if (result.success) {
        loadAttendance()
        alert(result.message)
      } else {
        alert(result.error)
      }
    } catch (error) {
      console.error('참석 여부 삭제 오류:', error)
      alert('삭제에 실패했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  // 웨딩 설정 편집 시작
  const startEditingConfig = () => {
    if (!weddingConfig) {
      alert('웨딩 정보를 먼저 로드해주세요.')
      return
    }
    
    try {
      // 깊은 복사를 위해 JSON 방식 사용
      const deepCopy = JSON.parse(JSON.stringify(weddingConfig))
      setEditingConfig(deepCopy)
      setIsEditing(true)
    } catch (error) {
      console.error('편집 모드 활성화 중 에러:', error)
      alert('편집 모드 활성화에 실패했습니다.')
    }
  }

  // 웨딩 설정 저장
  const saveWeddingConfig = async () => {
    if (!editingConfig) {
      alert('편집할 데이터가 없습니다.')
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('/api/wedding/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(editingConfig)
      })
      const result = await response.json()
      if (result.success) {
        setWeddingConfig(result.data)
        setEditingConfig(null)
        setIsEditing(false)
        alert(result.message)
      } else {
        alert(result.error)
      }
    } catch (error) {
      console.error('웨딩 설정 저장 오류:', error)
      alert('저장에 실패했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  // 편집 취소
  const cancelEditing = () => {
    setEditingConfig(null)
    setIsEditing(false)
  }

  // 로그아웃
  const handleLogout = () => {
    setIsAuthenticated(false)
    setPassword('')
    setActiveTab('guestbook')
    setGuestbook([])
    setAttendance([])
    setWeddingConfig(null)
    setEditingConfig(null)
    setIsEditing(false)
  }

  // 갤러리 편집 시작
  const startEditingGallery = () => {
    if (!galleryData) {
      alert('갤러리 데이터를 먼저 로드해주세요.')
      return
    }
    
    try {
      // 깊은 복사
      const deepCopy = JSON.parse(JSON.stringify(galleryData))
      setEditingGallery(deepCopy)
      setIsEditingGallery(true)
    } catch (error) {
      console.error('갤러리 편집 모드 활성화 중 에러:', error)
      alert('편집 모드 활성화에 실패했습니다.')
    }
  }

  // 갤러리 저장
  const saveGalleryData = async () => {
    if (!editingGallery) {
      alert('편집할 갤러리 데이터가 없습니다.')
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('/api/wedding/gallery', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(editingGallery)
      })
      const result = await response.json()
      if (result.success) {
        setGalleryData(result.data)
        setEditingGallery(null)
        setIsEditingGallery(false)
        alert(result.message)
      } else {
        alert(result.error)
      }
    } catch (error) {
      console.error('갤러리 저장 오류:', error)
      alert('갤러리 저장에 실패했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  // 갤러리 편집 취소
  const cancelEditingGallery = () => {
    setEditingGallery(null)
    setIsEditingGallery(false)
  }

  // 빈 공간 찾기 함수
  const findEmptySpace = (width = 1, height = 1, existingImages = []) => {
    const { gridCols, gridRows } = editingGallery?.settings || { gridCols: 3, gridRows: 2 }
    
    // 특정 위치와 크기가 다른 이미지와 겹치는지 확인
    const isOverlapping = (x, y, w, h) => {
      return existingImages.some(img => {
        const imgRight = img.position.x + img.gridSize.width
        const imgBottom = img.position.y + img.gridSize.height
        const newRight = x + w
        const newBottom = y + h
        
        return !(x >= imgRight || newRight <= img.position.x || 
                 y >= imgBottom || newBottom <= img.position.y)
      })
    }

    for (let y = 0; y <= gridRows - height; y++) {
      for (let x = 0; x <= gridCols - width; x++) {
        if (!isOverlapping(x, y, width, height)) {
          return { x, y }
        }
      }
    }
    
    // 빈 공간이 없으면 격자 확장을 고려하여 위치 반환
    return { x: 0, y: gridRows }
  }

  // 이미지 압축 함수
  const compressImage = (file, maxSizeKB = 800) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      const img = document.createElement('img')
      
      img.onload = () => {
        // 이미지 크기 계산 (최대 1200px)
        let { width, height } = img
        const maxDimension = 1200
        
        if (width > height && width > maxDimension) {
          height = (height * maxDimension) / width
          width = maxDimension
        } else if (height > maxDimension) {
          width = (width * maxDimension) / height
          height = maxDimension
        }
        
        canvas.width = width
        canvas.height = height
        
        // 이미지 그리기
        ctx.drawImage(img, 0, 0, width, height)
        
        // 품질 조정하면서 압축
        let quality = 0.8
        const compress = () => {
          canvas.toBlob((blob) => {
            const sizeKB = blob.size / 1024
            console.log(`압축 결과: ${sizeKB.toFixed(1)}KB (품질: ${quality})`)
            
            if (sizeKB <= maxSizeKB || quality <= 0.1) {
              // 압축된 파일 객체 생성
              const compressedFile = new File([blob], file.name, {
                type: blob.type,
                lastModified: Date.now()
              })
              resolve(compressedFile)
            } else {
              quality -= 0.1
              compress()
            }
          }, 'image/jpeg', quality)
        }
        
        compress()
      }
      
      img.onerror = () => {
        console.error('이미지 로드 실패')
        resolve(file) // 압축 실패 시 원본 파일 반환
      }
      
      img.src = URL.createObjectURL(file)
    })
  }

  // 이미지 추가 (압축 기능 포함)
  const addImages = async (files) => {
    if (!editingGallery || !files || files.length === 0) {
      console.log('조건 체크 실패:', {
        editingGallery: !!editingGallery,
        files: !!files,
        filesLength: files?.length || 0
      })
      return
    }

    console.log('=== 이미지 추가 시작 ===')
    console.log('선택된 파일 개수:', files.length)
    
    setIsLoading(true)
    
    try {
      // 첫 번째 파일만 테스트
      const file = files[0]
      console.log('원본 파일 정보:', {
        name: file.name,
        size: file.size,
        type: file.type,
        sizeKB: Math.round(file.size / 1024)
      })

      // 이미지 파일인지 확인
      if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드 가능합니다.')
        return
      }

      // 이미지 압축
      console.log('이미지 압축 시작...')
      const compressedFile = await compressImage(file)
      
      console.log('압축된 파일 정보:', {
        name: compressedFile.name,
        size: compressedFile.size,
        type: compressedFile.type,
        sizeKB: Math.round(compressedFile.size / 1024)
      })

      // FormData 생성
      const formData = new FormData()
      formData.append('files', compressedFile)
      
      console.log('FormData 생성 완료')
      
      // 서버로 전송
      console.log('서버로 업로드 요청 전송...')
      
      const response = await fetch('/api/wedding/upload', {
        method: 'POST',
        body: formData
      })

      console.log('서버 응답 상태:', response.status)
      console.log('서버 응답 헤더:', Object.fromEntries(response.headers.entries()))
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('서버 오류 응답:', errorText)
        
        if (response.status === 413) {
          alert(`파일이 여전히 너무 큽니다.\n압축된 크기: ${Math.round(compressedFile.size / 1024)}KB\n\nnginx 설정을 확인하거나 더 작은 이미지를 사용해주세요.`)
        } else {
          alert(`서버 오류: ${response.status} ${errorText}`)
        }
        return
      }

      const result = await response.json()
      console.log('서버 응답 결과:', result)
      
      if (result.success) {
        console.log('업로드 성공!')
        alert(`성공: ${result.message}\n원본 크기: ${Math.round(file.size / 1024)}KB → 압축 후: ${Math.round(compressedFile.size / 1024)}KB`)
        
        // 갤러리에 이미지 추가
        if (result.files && result.files.length > 0) {
          const newImage = {
            id: Date.now(),
            src: result.files[0].url,
            alt: result.files[0].originalName,
            gridSize: { width: 1, height: 1 },
            position: { x: 0, y: 0 },
            fileName: result.files[0].fileName
          }
          
          setEditingGallery(prev => ({
            ...prev,
            images: [...prev.images, newImage]
          }))
          
          console.log('갤러리에 이미지 추가 완료')
        }
        
      } else {
        console.error('업로드 실패:', result.error)
        alert(`실패: ${result.error}`)
      }

    } catch (error) {
      console.error('=== 업로드 전체 오류 ===')
      console.error('오류 타입:', error.constructor.name)
      console.error('오류 메시지:', error.message)
      console.error('스택 트레이스:', error.stack)
      alert(`오류: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // 이미지 삭제
  const deleteImage = (imageId) => {
    if (!editingGallery) return

    setEditingGallery(prev => ({
      ...prev,
      images: prev.images.filter(img => img.id !== imageId)
    }))
  }

  // 갤러리 설정 업데이트
  const updateGallerySettings = (field, value) => {
    if (!editingGallery) return

    setEditingGallery(prev => ({
      ...prev,
      settings: {
        ...prev.settings,
        [field]: value
      }
    }))
  }

  // 이미지 위치 업데이트
  const updateImagePosition = (imageId, newPosition) => {
    if (!editingGallery) return

    setEditingGallery(prev => ({
      ...prev,
      images: prev.images.map(img => 
        img.id === imageId ? { ...img, position: newPosition } : img
      )
    }))
  }

  // 이미지 크기 업데이트
  const updateImageSize = (imageId, newSize) => {
    if (!editingGallery) return

    setEditingGallery(prev => ({
      ...prev,
      images: prev.images.map(img => 
        img.id === imageId ? { ...img, gridSize: newSize } : img
      )
    }))
  }

  // 랜덤 배치
  const randomizeImagePositions = () => {
    if (!editingGallery) return

    const { gridCols, gridRows } = editingGallery.settings
    const arrangedImages = []

    // 이미지들을 랜덤하게 섞기
    const shuffledImages = [...editingGallery.images].sort(() => Math.random() - 0.5)

    // 각 이미지를 빈 공간에 배치
    shuffledImages.forEach(image => {
      const emptySpace = findEmptySpace(image.gridSize.width, image.gridSize.height, arrangedImages)
      
      const newImage = {
        ...image,
        position: emptySpace
      }
      
      arrangedImages.push(newImage)
    })

    // 모든 이미지 위치 업데이트
    setEditingGallery(prev => ({
      ...prev,
      images: arrangedImages
    }))
  }

  // URL로 이미지 추가
  const addImageByUrl = () => {
    const imageUrl = prompt('이미지 URL을 입력하세요:')
    if (!imageUrl || !editingGallery) return

    const newImage = {
      id: Date.now(),
      src: imageUrl,
      alt: `이미지 ${editingGallery.images.length + 1}`,
      gridSize: { width: 1, height: 1 },
      position: { x: 0, y: 0 }
    }

    setEditingGallery(prev => ({
      ...prev,
      images: [...prev.images, newImage]
    }))

    alert('이미지가 추가되었습니다.')
  }

  // 테스트 파일 업로드 함수
  const testFileUpload = async (files) => {
    if (!files || files.length === 0) {
      alert('파일을 선택해주세요.')
      return
    }

    console.log('테스트 파일 업로드 시작:', files.length, '개 파일')
    
    try {
      const formData = new FormData()
      Array.from(files).forEach(file => {
        console.log('테스트 파일 추가:', file.name, file.type, file.size)
        formData.append('files', file)
      })

      const response = await fetch('/api/test-upload', {
        method: 'POST',
        body: formData
      })

      console.log('테스트 응답 상태:', response.status)
      
      const result = await response.json()
      console.log('테스트 결과:', result)
      
      if (result.success) {
        alert(`테스트 성공: ${result.message}\n\n파일 정보:\n${result.files.map(f => `${f.name} (${f.size} bytes, ${f.type})`).join('\n')}`)
      } else {
        alert(`테스트 실패: ${result.error}`)
      }
      
    } catch (error) {
      console.error('테스트 오류:', error)
      alert(`테스트 오류: ${error.message}`)
    }
  }

  // 압축 없이 바로 업로드 (작은 파일용)
  const addImagesDirectly = async (files) => {
    if (!editingGallery || !files || files.length === 0) return

    console.log('=== 직접 업로드 시작 ===')
    setIsLoading(true)
    
    try {
      const file = files[0]
      const sizeKB = Math.round(file.size / 1024)
      
      console.log('파일 정보:', {
        name: file.name,
        size: file.size,
        type: file.type,
        sizeKB: sizeKB
      })

      // 크기 체크 (1MB = 1024KB)
      if (sizeKB > 1024) {
        alert(`파일이 너무 큽니다: ${sizeKB}KB\n1MB(1024KB) 이하의 파일을 선택하거나 압축 업로드를 사용해주세요.`)
        return
      }

      const formData = new FormData()
      formData.append('files', file)
      
      const response = await fetch('/api/wedding/upload', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('서버 오류:', errorText)
        alert(`업로드 실패: ${response.status} ${errorText}`)
        return
      }

      const result = await response.json()
      
      if (result.success && result.files && result.files.length > 0) {
        const newImage = {
          id: Date.now(),
          src: result.files[0].url,
          alt: result.files[0].originalName,
          gridSize: { width: 1, height: 1 },
          position: { x: 0, y: 0 },
          fileName: result.files[0].fileName
        }
        
        setEditingGallery(prev => ({
          ...prev,
          images: [...prev.images, newImage]
        }))
        
        alert(`업로드 성공: ${result.message}`)
      } else {
        alert(`업로드 실패: ${result.error}`)
      }

    } catch (error) {
      console.error('직접 업로드 오류:', error)
      alert(`오류: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // 인증되지 않은 경우 로그인 화면
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <div className="mx-auto w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mb-4">
              <Lock className="w-6 h-6 text-blue-600" />
            </div>
            <CardTitle className="text-2xl font-bold text-gray-800">관리자 로그인</CardTitle>
            <p className="text-gray-600">웨딩 초대장 관리 페이지</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative">
              <Input
                type={showPassword ? 'text' : 'password'}
                placeholder="비밀번호를 입력하세요"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
            <Button
              onClick={handleLogin}
              className="w-full bg-blue-600 hover:bg-blue-700"
              disabled={!password}
            >
              로그인
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 헤더 */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-800">웨딩 관리자 페이지</h1>
            <Button
              onClick={handleLogout}
              variant="outline"
              className="text-red-600 border-red-600 hover:bg-red-50"
            >
              로그아웃
            </Button>
          </div>
          
          {/* 탭 네비게이션 */}
          <div className="flex space-x-1 mt-4">
            <button
              onClick={() => setActiveTab('guestbook')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'guestbook'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <MessageCircle className="inline w-4 h-4 mr-2" />
              방명록 관리
            </button>
            <button
              onClick={() => setActiveTab('attendance')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'attendance'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Users className="inline w-4 h-4 mr-2" />
              참석 관리
            </button>
            <button
              onClick={() => setActiveTab('gallery')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'gallery'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Image className="inline w-4 h-4 mr-2" />
              갤러리 관리
            </button>
            <button
              onClick={() => setActiveTab('config')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'config'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Settings className="inline w-4 h-4 mr-2" />
              웨딩 정보
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* 갤러리 관리 */}
        {activeTab === 'gallery' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">갤러리 관리</h2>
              {!isEditingGallery ? (
                <Button
                  onClick={startEditingGallery}
                  className="bg-blue-600 hover:bg-blue-700 text-white border-2 border-blue-600 hover:border-blue-700 font-semibold px-4 py-2 rounded-lg shadow-md"
                >
                  <Edit className="w-4 h-4 mr-2 text-white" />
                  편집
                </Button>
              ) : (
                <div className="space-x-2">
                  <Button
                    onClick={saveGalleryData}
                    className="bg-green-600 hover:bg-green-700 text-white border-2 border-green-600 hover:border-green-700 font-semibold px-4 py-2 rounded-lg shadow-md"
                    disabled={isLoading}
                  >
                    <Save className="w-4 h-4 mr-2 text-white" />
                    저장
                  </Button>
                  <Button
                    onClick={cancelEditingGallery}
                    className="bg-red-600 hover:bg-red-700 text-white border-2 border-red-600 hover:border-red-700 font-semibold px-4 py-2 rounded-lg shadow-md"
                  >
                    <X className="w-4 h-4 mr-2 text-white" />
                    취소
                  </Button>
                </div>
              )}
            </div>

            {/* 편집 모드 상태 표시 */}
            {isEditingGallery && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-blue-800 text-sm">
                  🖼️ 갤러리 편집 모드: 설정을 변경하고 이미지를 편집한 후 저장 버튼을 눌러주세요.
                </p>
              </div>
            )}

            {galleryData && (
              <div className="space-y-6">
                {/* 갤러리 설정 */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Settings className="w-5 h-5 mr-2" />
                      갤러리 설정
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">격자 가로 개수</label>
                        <Input
                          type="number"
                          min="1"
                          max="10"
                          value={isEditingGallery ? editingGallery.settings.gridCols : galleryData.settings.gridCols}
                          onChange={(e) => updateGallerySettings('gridCols', parseInt(e.target.value))}
                          disabled={!isEditingGallery}
                          className={isEditingGallery ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">격자 세로 개수</label>
                        <Input
                          type="number"
                          min="1"
                          max="10"
                          value={isEditingGallery ? editingGallery.settings.gridRows : galleryData.settings.gridRows}
                          onChange={(e) => updateGallerySettings('gridRows', parseInt(e.target.value))}
                          disabled={!isEditingGallery}
                          className={isEditingGallery ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">최대 노출 개수</label>
                        <Input
                          type="number"
                          min="1"
                          max="50"
                          value={isEditingGallery ? editingGallery.settings.maxVisible : galleryData.settings.maxVisible}
                          onChange={(e) => updateGallerySettings('maxVisible', parseInt(e.target.value))}
                          disabled={!isEditingGallery}
                          className={isEditingGallery ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">랜덤 배치</label>
                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={isEditingGallery ? editingGallery.settings.randomOrder : galleryData.settings.randomOrder}
                            onChange={(e) => updateGallerySettings('randomOrder', e.target.checked)}
                            disabled={!isEditingGallery}
                            className="rounded"
                          />
                          {isEditingGallery && (
                            <Button
                              onClick={randomizeImagePositions}
                              size="sm"
                              variant="outline"
                              className="text-xs"
                            >
                              <Shuffle className="w-3 h-3 mr-1" />
                              섞기
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* 이미지 그리드 */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center">
                        <Grid className="w-5 h-5 mr-2" />
                        이미지 그리드
                      </div>
                      <div className="text-sm text-gray-600">
                        총 {isEditingGallery ? editingGallery.images.length : galleryData.images.length}개
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <GalleryGrid
                      images={isEditingGallery ? editingGallery.images : galleryData.images}
                      settings={isEditingGallery ? editingGallery.settings : galleryData.settings}
                      isEditing={isEditingGallery}
                      onImageDelete={deleteImage}
                      onImageMove={updateImagePosition}
                      onImageResize={updateImageSize}
                    />
                    
                    {/* 이미지 추가 버튼들 */}
                    {isEditingGallery && (
                      <div className="mt-4 space-y-2">
                        {/* URL로 이미지 추가 버튼 */}
                        <div className="flex justify-center">
                          <button
                            onClick={addImageByUrl}
                            className="flex items-center justify-center px-4 py-2 border-2 border-dashed border-green-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors"
                          >
                            <Plus className="w-5 h-5 mr-2 text-green-600" />
                            <span className="text-green-600 font-medium">URL로 이미지 추가</span>
                          </button>
                        </div>

                        {/* 직접 업로드 버튼 (1MB 이하) */}
                        <div className="flex justify-center">
                          <label className="cursor-pointer">
                            <input
                              type="file"
                              accept="image/*"
                              onChange={(e) => addImagesDirectly(e.target.files)}
                              className="hidden"
                            />
                            <div className="flex items-center justify-center px-4 py-2 border-2 border-dashed border-purple-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors">
                              <Plus className="w-5 h-5 mr-2 text-purple-600" />
                              <span className="text-purple-600 font-medium">직접 업로드 (1MB 이하)</span>
                            </div>
                          </label>
                        </div>

                        {/* 테스트 업로드 버튼 */}
                        <div className="flex justify-center">
                          <label className="cursor-pointer">
                            <input
                              type="file"
                              multiple
                              accept="image/*"
                              onChange={(e) => testFileUpload(e.target.files)}
                              className="hidden"
                            />
                            <div className="flex items-center justify-center px-4 py-2 border-2 border-dashed border-blue-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors">
                              <Plus className="w-5 h-5 mr-2 text-blue-600" />
                              <span className="text-blue-600 font-medium">테스트 파일 업로드</span>
                            </div>
                          </label>
                        </div>
                        
                        {/* 압축 업로드 버튼 */}
                        <div className="flex justify-center">
                          <label className="cursor-pointer">
                            <input
                              type="file"
                              multiple
                              accept="image/*"
                              onChange={(e) => addImages(e.target.files)}
                              className="hidden"
                            />
                            <div className="flex items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors">
                              <div className="text-center">
                                <Plus className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                                <p className="text-sm text-gray-600">압축 업로드 (큰 파일용)</p>
                                <p className="text-xs text-gray-500">자동으로 800KB 이하로 압축</p>
                              </div>
                            </div>
                          </label>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        )}

        {/* 방명록 관리 */}
        {activeTab === 'guestbook' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">방명록 관리</h2>
              <div className="text-sm text-gray-600">
                총 {guestbook.length}개의 메시지
              </div>
            </div>
            
            <div className="grid gap-4">
              {guestbook.map((entry) => (
                <Card key={entry.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-semibold text-blue-600">{entry.name}</span>
                          <span className="text-xs text-gray-500">{entry.date}</span>
                        </div>
                        <p className="text-gray-700 whitespace-pre-wrap">{entry.message}</p>
                      </div>
                      <Button
                        onClick={() => deleteGuestbookEntry(entry.id)}
                        variant="ghost"
                        size="sm"
                        className="ml-4 text-red-600 hover:bg-red-50"
                        disabled={isLoading}
                      >
                        <Trash2 size={16} />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
              {guestbook.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  아직 등록된 방명록이 없습니다.
                </div>
              )}
            </div>
          </div>
        )}

        {/* 참석 관리 */}
        {activeTab === 'attendance' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">참석 관리</h2>
              <div className="text-sm text-gray-600 space-x-4">
                <span>참석: {attendanceStats.totalAttending}명</span>
                <span>불참: {attendanceStats.totalNotAttending}명</span>
                <span>총 인원: {attendanceStats.totalGuestCount}명</span>
              </div>
            </div>
            
            <div className="grid gap-4">
              {attendance.map((entry) => (
                <Card key={entry.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-semibold text-blue-600">{entry.name}</span>
                          <span className="text-xs text-gray-500">{entry.date}</span>
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          <p>연락처: {entry.phone}</p>
                          <p className="flex items-center">
                            참석 여부: 
                            <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${
                              entry.attending === 'yes' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {entry.attending === 'yes' ? '참석' : '불참'}
                            </span>
                          </p>
                          {entry.attending === 'yes' && (
                            <p>참석 인원: {entry.guestCount}명</p>
                          )}
                        </div>
                      </div>
                      <Button
                        onClick={() => deleteAttendanceEntry(entry.id)}
                        variant="ghost"
                        size="sm"
                        className="ml-4 text-red-600 hover:bg-red-50"
                        disabled={isLoading}
                      >
                        <Trash2 size={16} />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
              {attendance.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  아직 등록된 참석 정보가 없습니다.
                </div>
              )}
            </div>
          </div>
        )}

        {/* 웨딩 정보 관리 */}
        {activeTab === 'config' && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">웨딩 정보 관리</h2>
              {!isEditing ? (
                <Button
                  onClick={startEditingConfig}
                  className="bg-blue-600 hover:bg-blue-700 text-white border-2 border-blue-600 hover:border-blue-700 font-semibold px-4 py-2 rounded-lg shadow-md"
                >
                  <Edit className="w-4 h-4 mr-2 text-white" />
                  편집
                </Button>
              ) : (
                <div className="space-x-2">
                  <Button
                    onClick={saveWeddingConfig}
                    className="bg-green-600 hover:bg-green-700 text-white border-2 border-green-600 hover:border-green-700 font-semibold px-4 py-2 rounded-lg shadow-md"
                    disabled={isLoading}
                  >
                    <Save className="w-4 h-4 mr-2 text-white" />
                    저장
                  </Button>
                  <Button
                    onClick={cancelEditing}
                    className="bg-red-600 hover:bg-red-700 text-white border-2 border-red-600 hover:border-red-700 font-semibold px-4 py-2 rounded-lg shadow-md"
                  >
                    <X className="w-4 h-4 mr-2 text-white" />
                    취소
                  </Button>
                </div>
              )}
            </div>

            {/* 편집 모드 상태 표시 */}
            {isEditing && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-blue-800 text-sm">
                  🖊️ 편집 모드: 정보를 수정한 후 저장 버튼을 눌러주세요.
                </p>
              </div>
            )}

            {weddingConfig && (
              <div className="grid gap-6">
                {/* 신랑 정보 */}
                <Card>
                  <CardHeader>
                    <CardTitle>신랑 정보</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">이름</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.groom?.name || '') : (weddingConfig.couple?.groom?.name || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  groom: { ...editingConfig.couple.groom, name: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">연락처</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.groom?.phone || '') : (weddingConfig.couple?.groom?.phone || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  groom: { ...editingConfig.couple.groom, phone: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">아버지</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.groom?.father || '') : (weddingConfig.couple?.groom?.father || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  groom: { ...editingConfig.couple.groom, father: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">어머니</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.groom?.mother || '') : (weddingConfig.couple?.groom?.mother || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  groom: { ...editingConfig.couple.groom, mother: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium mb-1">계좌번호</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.groom?.account || '') : (weddingConfig.couple?.groom?.account || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  groom: { ...editingConfig.couple.groom, account: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* 신부 정보 */}
                <Card>
                  <CardHeader>
                    <CardTitle>신부 정보</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">이름</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.bride?.name || '') : (weddingConfig.couple?.bride?.name || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  bride: { ...editingConfig.couple.bride, name: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">연락처</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.bride?.phone || '') : (weddingConfig.couple?.bride?.phone || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  bride: { ...editingConfig.couple.bride, phone: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">아버지</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.bride?.father || '') : (weddingConfig.couple?.bride?.father || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  bride: { ...editingConfig.couple.bride, father: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">어머니</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.bride?.mother || '') : (weddingConfig.couple?.bride?.mother || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  bride: { ...editingConfig.couple.bride, mother: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium mb-1">계좌번호</label>
                        <Input
                          value={isEditing ? (editingConfig?.couple?.bride?.account || '') : (weddingConfig.couple?.bride?.account || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                couple: {
                                  ...editingConfig.couple,
                                  bride: { ...editingConfig.couple.bride, account: e.target.value }
                                }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* 결혼식 정보 */}
                <Card>
                  <CardHeader>
                    <CardTitle>결혼식 정보</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">날짜</label>
                        <Input
                          value={isEditing ? (editingConfig?.wedding?.date || '') : (weddingConfig.wedding?.date || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                wedding: { ...editingConfig.wedding, date: e.target.value }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">시간</label>
                        <Input
                          value={isEditing ? (editingConfig?.wedding?.time || '') : (weddingConfig.wedding?.time || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                wedding: { ...editingConfig.wedding, time: e.target.value }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">장소명</label>
                        <Input
                          value={isEditing ? (editingConfig?.wedding?.venue || '') : (weddingConfig.wedding?.venue || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                wedding: { ...editingConfig.wedding, venue: e.target.value }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">층/홀</label>
                        <Input
                          value={isEditing ? (editingConfig?.wedding?.floor || '') : (weddingConfig.wedding?.floor || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                wedding: { ...editingConfig.wedding, floor: e.target.value }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium mb-1">주소</label>
                        <Input
                          value={isEditing ? (editingConfig?.wedding?.address || '') : (weddingConfig.wedding?.address || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                wedding: { ...editingConfig.wedding, address: e.target.value }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium mb-1">문의 전화</label>
                        <Input
                          value={isEditing ? (editingConfig?.wedding?.phone || '') : (weddingConfig.wedding?.phone || '')}
                          onChange={(e) => {
                            if (isEditing && editingConfig) {
                              setEditingConfig({
                                ...editingConfig,
                                wedding: { ...editingConfig.wedding, phone: e.target.value }
                              })
                            }
                          }}
                          disabled={!isEditing}
                          className={isEditing ? 'border-blue-300 bg-blue-50' : ''}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* 인사말 */}
                <Card>
                  <CardHeader>
                    <CardTitle>인사말</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Textarea
                      value={isEditing ? (editingConfig?.message || '') : (weddingConfig.message || '')}
                      onChange={(e) => {
                        if (isEditing && editingConfig) {
                          setEditingConfig({
                            ...editingConfig,
                            message: e.target.value
                          })
                        }
                      }}
                      disabled={!isEditing}
                      rows={6}
                      className={`w-full ${isEditing ? 'border-blue-300 bg-blue-50' : ''}`}
                    />
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default AdminPage 