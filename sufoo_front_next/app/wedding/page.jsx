"use client"

import React, { useState, useEffect } from 'react'
import { Heart, MapPin, Phone, Copy, Calendar, Gift, MessageCircle, Music, Users, Camera, X, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogTrigger } from '@/components/ui/dialog'
import './wedding.css'

function WeddingPage() {
  const [guestbook, setGuestbook] = useState([])
  const [newMessage, setNewMessage] = useState({ name: '', message: '' })
  const [dDay, setDDay] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [attendance, setAttendance] = useState([])
  const [attendanceStats, setAttendanceStats] = useState({ totalAttending: 0, totalGuestCount: 0 })
  const [attendanceForm, setAttendanceForm] = useState({ name: '', phone: '', attending: '', guestCount: 1 })
  const [isLoading, setIsLoading] = useState(false)
  const [weddingData, setWeddingData] = useState(null)
  const [galleryData, setGalleryData] = useState(null)
  const [showAllGallery, setShowAllGallery] = useState(false)
  const [selectedImage, setSelectedImage] = useState(null)

  // 초기 데이터 로드
  useEffect(() => {
    loadWeddingData()
    loadGuestbook()
    loadAttendance()
    loadGalleryData()
  }, [])

  // 웨딩 데이터 로드
  const loadWeddingData = async () => {
    try {
      const response = await fetch('/api/wedding/config')
      const result = await response.json()
      if (result.success) {
        setWeddingData(result.data)
      }
    } catch (error) {
      console.error('웨딩 데이터 로드 오류:', error)
    }
  }

  // D-Day 계산
  useEffect(() => {
    if (weddingData) {
      // 날짜 형식에서 실제 날짜 추출 (예: "2024년 12월 14일 토요일" -> "2024-12-14")
      const dateMatch = weddingData.wedding.date.match(/(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일/)
      if (dateMatch) {
        const [, year, month, day] = dateMatch
        const weddingDate = new Date(parseInt(year), parseInt(month) - 1, parseInt(day))
        const today = new Date()
        const diffTime = weddingDate - today
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
        setDDay(diffDays)
      }
    }
  }, [weddingData])

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

  // 계좌번호 복사 함수
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('계좌번호가 복사되었습니다.')
    })
  }

  // 전화걸기 함수
  const makeCall = (phoneNumber) => {
    window.location.href = `tel:${phoneNumber}`
  }

  // 지도 열기 함수
  const openMap = (type) => {
    if (!weddingData) return
    const address = encodeURIComponent(weddingData.wedding.address)
    const urls = {
      kakao: `https://map.kakao.com/link/search/${address}`,
      naver: `https://map.naver.com/v5/search/${address}`,
      google: `https://maps.google.com/maps?q=${address}`
    }
    window.open(urls[type], '_blank')
  }

  // 방명록 추가 함수
  const addGuestbookEntry = async () => {
    if (!newMessage.name || !newMessage.message) {
      alert('이름과 메시지를 모두 입력해주세요.')
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('/api/wedding/guestbook', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newMessage.name,
          message: newMessage.message
        })
      })

      const result = await response.json()
      if (result.success) {
        setNewMessage({ name: '', message: '' })
        loadGuestbook() // 새로고침
        alert(result.message)
      } else {
        alert(result.error || '메시지 등록에 실패했습니다.')
      }
    } catch (error) {
      console.error('방명록 추가 오류:', error)
      alert('메시지 등록에 실패했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  // 캘린더에 추가 함수
  const addToCalendar = () => {
    if (!weddingData) return
    
    // 날짜 형식에서 실제 날짜 추출
    const dateMatch = weddingData.wedding.date.match(/(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일/)
    if (dateMatch) {
      const [, year, month, day] = dateMatch
      const formattedDate = `${year}${month.padStart(2, '0')}${day.padStart(2, '0')}`
      const startDate = `${formattedDate}T140000`
      const endDate = `${formattedDate}T170000`
      const title = encodeURIComponent(`${weddingData.couple.groom.name} ♥ ${weddingData.couple.bride.name} 결혼식`)
      const location = encodeURIComponent(weddingData.wedding.address)
      
      const googleCalendarUrl = `https://calendar.google.com/calendar/render?action=TEMPLATE&text=${title}&dates=${startDate}/${endDate}&location=${location}`
      window.open(googleCalendarUrl, '_blank')
    }
  }

  // 참석 여부 확인 함수
  const submitAttendance = async () => {
    if (!attendanceForm.name || !attendanceForm.phone || !attendanceForm.attending) {
      alert('모든 필수 항목을 입력해주세요.')
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('/api/wedding/attendance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: attendanceForm.name,
          phone: attendanceForm.phone,
          attending: attendanceForm.attending,
          guestCount: attendanceForm.guestCount
        })
      })

      const result = await response.json()
      if (result.success) {
        setAttendanceForm({ name: '', phone: '', attending: '', guestCount: 1 })
        loadAttendance() // 새로고침
        alert(result.message)
      } else {
        alert(result.error || '참석 여부 등록에 실패했습니다.')
      }
    } catch (error) {
      console.error('참석 여부 추가 오류:', error)
      alert('참석 여부 등록에 실패했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  // 배경음악 토글 함수
  const toggleMusic = () => {
    setIsPlaying(!isPlaying)
    if (!isPlaying) {
      alert('배경음악이 재생됩니다.')
    } else {
      alert('배경음악이 정지됩니다.')
    }
  }

  // 웨딩 데이터가 로드되지 않았을 때 로딩 표시
  if (!weddingData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-wedding-primary">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-wedding-accent mx-auto mb-4"></div>
          <p className="text-wedding-accent">웨딩 정보를 불러오는 중...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="wedding-container">
      {/* 메인 헤더 */}
      <section className="wedding-section wedding-fade-in">
        <div className="text-center">
          <h1 className="wedding-title">Wedding Invitation</h1>
          <p className="wedding-subtitle">우리의 특별한 날에 초대합니다</p>
          <div className="wedding-divider"></div>
          
          <div className="wedding-names">
            {weddingData.couple.groom.name} ♥ {weddingData.couple.bride.name}
          </div>
          
          <div className="wedding-date">
            {weddingData.wedding.date}
          </div>
          <div className="wedding-location">
            {weddingData.wedding.time} | {weddingData.wedding.venue}
          </div>
          
          {dDay > 0 && (
            <div className="wedding-dday">
              D-{dDay}
            </div>
          )}
          {dDay === 0 && (
            <div className="wedding-dday">
              오늘이 바로 그날! 💒
            </div>
          )}
          
          {/* 배경음악 컨트롤 */}
          <div className="text-center mt-4">
            <Button 
              onClick={toggleMusic}
              variant="outline"
              size="sm"
              className="wedding-button"
            >
              <Music className="mr-2" size={16} />
              {isPlaying ? '음악 정지' : '배경음악 재생'}
            </Button>
          </div>
        </div>
      </section>

      {/* 인사말 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <Heart className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">인사말</h2>
            </div>
            <p className="wedding-text whitespace-pre-line">
              {weddingData.message}
            </p>
          </CardContent>
        </Card>
      </section>

      {/* 예식 정보 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <Calendar className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">예식 정보</h2>
            </div>
            
            <div className="space-y-3">
              <div className="text-center">
                <p className="font-semibold">{weddingData.wedding.date}</p>
                <p>{weddingData.wedding.time}</p>
              </div>
              
              <div className="text-center">
                <p className="font-semibold">{weddingData.wedding.venue}</p>
                <p>{weddingData.wedding.address}</p>
                <p>{weddingData.wedding.floor}</p>
                <p className="text-sm text-gray-600">문의: {weddingData.wedding.phone}</p>
              </div>
              
              <div className="flex justify-center mt-4">
                <Button 
                  onClick={addToCalendar}
                  className="wedding-button"
                >
                  <Calendar className="mr-2" size={16} />
                  캘린더에 추가
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 오시는 길 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <MapPin className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">오시는 길</h2>
            </div>
            
            <div className="text-center mb-4">
              <p className="font-semibold">{weddingData.wedding.venue}</p>
              <p className="text-sm">{weddingData.wedding.address}</p>
              <Button 
                onClick={() => copyToClipboard(weddingData.wedding.address)}
                variant="outline" 
                size="sm" 
                className="mt-2"
              >
                <Copy className="mr-2" size={14} />
                주소 복사
              </Button>
            </div>
            
            <div className="grid grid-cols-3 gap-2">
              <Button 
                onClick={() => openMap('kakao')}
                variant="outline"
                className="text-xs"
              >
                카카오맵
              </Button>
              <Button 
                onClick={() => openMap('naver')}
                variant="outline"
                className="text-xs"
              >
                네이버지도
              </Button>
              <Button 
                onClick={() => openMap('google')}
                variant="outline"
                className="text-xs"
              >
                구글맵
              </Button>
            </div>
            
            <div className="mt-4 p-3 bg-wedding-secondary rounded-lg">
              <h4 className="font-semibold mb-2">교통 정보</h4>
              <p className="text-sm">🚇 지하철 2호선 강남역 3번 출구 도보 5분</p>
              <p className="text-sm">🚌 버스 146, 740, 341번 강남역 하차</p>
              <p className="text-sm">🚗 주차 가능 (2시간 무료)</p>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 연락처 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <Phone className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">연락처</h2>
            </div>
            
            <div className="space-y-4">
              <div className="wedding-contact">
                <div>
                  <p className="font-semibold">신랑 {weddingData.couple.groom.name}</p>
                  <p className="text-sm text-gray-600">{weddingData.couple.groom.father} · {weddingData.couple.groom.mother}의 아들</p>
                </div>
                <Button 
                  onClick={() => makeCall(weddingData.couple.groom.phone)}
                  size="sm"
                  className="wedding-button"
                >
                  <Phone size={16} />
                </Button>
              </div>
              
              <div className="wedding-contact">
                <div>
                  <p className="font-semibold">신부 {weddingData.couple.bride.name}</p>
                  <p className="text-sm text-gray-600">{weddingData.couple.bride.father} · {weddingData.couple.bride.mother}의 딸</p>
                </div>
                <Button 
                  onClick={() => makeCall(weddingData.couple.bride.phone)}
                  size="sm"
                  className="wedding-button"
                >
                  <Phone size={16} />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 축의금 계좌 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <Gift className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">축의금 계좌</h2>
            </div>
            
            <div className="space-y-3">
              <div className="wedding-account">
                <p className="font-semibold mb-1">신랑측</p>
                <p className="text-sm mb-2">{weddingData.couple.groom.account}</p>
                <Button 
                  onClick={() => copyToClipboard(weddingData.couple.groom.account)}
                  size="sm"
                  variant="outline"
                >
                  <Copy className="mr-2" size={14} />
                  복사
                </Button>
              </div>
              
              <div className="wedding-account">
                <p className="font-semibold mb-1">신부측</p>
                <p className="text-sm mb-2">{weddingData.couple.bride.account}</p>
                <Button 
                  onClick={() => copyToClipboard(weddingData.couple.bride.account)}
                  size="sm"
                  variant="outline"
                >
                  <Copy className="mr-2" size={14} />
                  복사
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 갤러리 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <Camera className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">갤러리</h2>
            </div>
            
            {galleryData && galleryData.images && (
              <div className="space-y-4">
                {/* 갤러리 그리드 */}
                <div 
                  className="grid gap-2 rounded-lg overflow-hidden"
                  style={{
                    gridTemplateColumns: `repeat(${galleryData.settings.gridCols}, 1fr)`,
                    gridTemplateRows: `repeat(${galleryData.settings.gridRows}, 1fr)`,
                    aspectRatio: `${galleryData.settings.gridCols}/${galleryData.settings.gridRows}`
                  }}
                >
                  {galleryData.images
                    .slice(0, showAllGallery ? galleryData.images.length : galleryData.settings.maxVisible)
                    .map((image) => (
                      <div
                        key={image.id}
                        className="relative overflow-hidden rounded-lg cursor-pointer hover:scale-105 transition-transform duration-300 bg-gray-100"
                        style={{
                          gridColumn: `${image.position.x + 1} / span ${image.gridSize.width}`,
                          gridRow: `${image.position.y + 1} / span ${image.gridSize.height}`
                        }}
                        onClick={() => setSelectedImage(image)}
                      >
                        <img
                          src={image.src}
                          alt={image.alt}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            // 이미지 로딩 실패 시 placeholder로 대체
                            e.target.src = `https://via.placeholder.com/300x200/d4af37/ffffff?text=Wedding+${image.id}`
                          }}
                        />
                      </div>
                    ))}
                </div>

                {/* 더보기 버튼 */}
                {!showAllGallery && galleryData.images.length > galleryData.settings.maxVisible && (
                  <div className="flex justify-center mt-4">
                    <button
                      onClick={() => setShowAllGallery(true)}
                      className="relative w-12 h-12 rounded-full flex items-center justify-center bg-gradient-to-br from-wedding-primary to-wedding-accent text-white shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110"
                    >
                      <Plus size={24} />
                      <div className="absolute inset-0 rounded-full bg-gradient-to-br from-white/20 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300" />
                    </button>
                  </div>
                )}

                {/* 모든 이미지 표시 시 접기 버튼 */}
                {showAllGallery && (
                  <div className="flex justify-center mt-4">
                    <Button
                      onClick={() => setShowAllGallery(false)}
                      variant="outline"
                      size="sm"
                    >
                      접기
                    </Button>
                  </div>
                )}
              </div>
            )}

            {/* 갤러리 데이터가 없거나 이미지가 없는 경우 */}
            {(!galleryData || !galleryData.images || galleryData.images.length === 0) && (
              <div className="text-center py-8 text-gray-500">
                <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>곧 아름다운 웨딩 사진이 올라올 예정입니다 💕</p>
              </div>
            )}

            {/* 이미지 상세보기 모달 */}
            {selectedImage && (
              <Dialog open={!!selectedImage} onOpenChange={() => setSelectedImage(null)}>
                <DialogContent className="max-w-4xl w-full p-0">
                  <div className="relative">
                    <img
                      src={selectedImage.src}
                      alt={selectedImage.alt}
                      className="w-full h-auto max-h-[80vh] object-contain"
                    />
                    <button
                      onClick={() => setSelectedImage(null)}
                      className="absolute top-4 right-4 bg-black/50 text-white p-2 rounded-full hover:bg-black/70 transition-colors"
                    >
                      <X size={20} />
                    </button>
                  </div>
                </DialogContent>
              </Dialog>
            )}
          </CardContent>
        </Card>
      </section>

      {/* 방명록 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <MessageCircle className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">축하 메시지</h2>
            </div>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Input
                  placeholder="성함을 입력해주세요"
                  value={newMessage.name}
                  onChange={(e) => setNewMessage({...newMessage, name: e.target.value})}
                  maxLength={50}
                />
                <Textarea
                  placeholder="축하 메시지를 남겨주세요"
                  value={newMessage.message}
                  onChange={(e) => setNewMessage({...newMessage, message: e.target.value})}
                  rows={3}
                  maxLength={500}
                />
                <Button 
                  onClick={addGuestbookEntry}
                  className="wedding-button w-full"
                  disabled={isLoading}
                >
                  {isLoading ? '등록 중...' : '메시지 등록'}
                </Button>
              </div>
              
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {guestbook.map((entry) => (
                  <div key={entry.id} className="wedding-message">
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-semibold text-sm">{entry.name}</span>
                      <span className="text-xs text-gray-500">{entry.date}</span>
                    </div>
                    <p className="text-sm">{entry.message}</p>
                  </div>
                ))}
                {guestbook.length === 0 && (
                  <p className="text-center text-gray-500 text-sm py-4">
                    첫 번째 축하 메시지를 남겨주세요 💝
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 참석 여부 확인 */}
      <section className="wedding-section">
        <Card className="wedding-card">
          <CardContent className="p-6">
            <div className="text-center mb-4">
              <Users className="wedding-icon mx-auto mb-2" size={24} />
              <h2 className="text-xl font-semibold text-wedding-accent">참석 여부 확인</h2>
              <p className="text-sm text-gray-600 mt-2">
                정확한 식사 준비를 위해 참석 여부를 알려주세요
              </p>
            </div>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Input
                  placeholder="성함을 입력해주세요"
                  value={attendanceForm.name}
                  onChange={(e) => setAttendanceForm({...attendanceForm, name: e.target.value})}
                  maxLength={50}
                />
                <Input
                  placeholder="연락처를 입력해주세요"
                  value={attendanceForm.phone}
                  onChange={(e) => setAttendanceForm({...attendanceForm, phone: e.target.value})}
                  maxLength={20}
                />
                
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    onClick={() => setAttendanceForm({...attendanceForm, attending: 'yes'})}
                    variant={attendanceForm.attending === 'yes' ? 'default' : 'outline'}
                    className={attendanceForm.attending === 'yes' ? 'wedding-button' : ''}
                  >
                    참석
                  </Button>
                  <Button
                    onClick={() => setAttendanceForm({...attendanceForm, attending: 'no'})}
                    variant={attendanceForm.attending === 'no' ? 'default' : 'outline'}
                    className={attendanceForm.attending === 'no' ? 'wedding-button' : ''}
                  >
                    불참
                  </Button>
                </div>
                
                {attendanceForm.attending === 'yes' && (
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-medium">참석 인원:</label>
                    <Input
                      type="number"
                      min="1"
                      max="10"
                      value={attendanceForm.guestCount}
                      onChange={(e) => setAttendanceForm({...attendanceForm, guestCount: parseInt(e.target.value)})}
                      className="w-20"
                    />
                    <span className="text-sm">명</span>
                  </div>
                )}
                
                <Button 
                  onClick={submitAttendance}
                  className="wedding-button w-full"
                  disabled={isLoading}
                >
                  {isLoading ? '등록 중...' : '참석 여부 등록'}
                </Button>
              </div>
              
              <div className="mt-4 p-3 bg-wedding-secondary rounded-lg">
                <h4 className="font-semibold mb-2">참석 현황</h4>
                <p className="text-sm">
                  총 {attendanceStats.totalAttending}명 참석 예정
                </p>
                <p className="text-sm">
                  총 인원: {attendanceStats.totalGuestCount}명
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 푸터 */}
      <footer className="wedding-section text-center pb-8">
        <div className="wedding-divider"></div>
        <p className="text-sm text-gray-500 mb-2">
          {weddingData.couple.groom.name} ♥ {weddingData.couple.bride.name}
        </p>
        <p className="text-xs text-gray-400">
          소중한 분들과 함께하는 특별한 날
        </p>
      </footer>
    </div>
  )
}

export default WeddingPage 