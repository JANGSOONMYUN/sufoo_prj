import React, { useState, useEffect } from 'react'
import { Heart, MapPin, Phone, Copy, Calendar, Gift, MessageCircle, Music, Users, Camera, X } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent } from '@/components/ui/card.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Dialog, DialogContent, DialogTrigger } from '@/components/ui/dialog.jsx'
import './App.css'

// 갤러리 이미지 import
import wedding1 from './assets/wedding1.jpeg'
import wedding2 from './assets/wedding2.jpg'
import wedding3 from './assets/wedding3.jpg'
import wedding4 from './assets/wedding4.jpg'
import wedding5 from './assets/wedding5.jpg'
import wedding6 from './assets/wedding6.jpg'

// 웨딩 데이터 (실제로는 CMS나 데이터베이스에서 가져올 수 있음)
const weddingData = {
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
  message: "소중한 분들을 모시고\n저희 두 사람의 새로운 시작을\n함께 축복해 주시기 바랍니다.\n\n여러분의 따뜻한 마음과 축복이\n저희에게는 가장 큰 선물입니다.",
  gallery: [
    { id: 1, src: wedding1, alt: "웨딩 사진 1" },
    { id: 2, src: wedding2, alt: "웨딩 사진 2" },
    { id: 3, src: wedding3, alt: "웨딩 사진 3" },
    { id: 4, src: wedding4, alt: "웨딩 사진 4" },
    { id: 5, src: wedding5, alt: "웨딩 사진 5" },
    { id: 6, src: wedding6, alt: "웨딩 사진 6" }
  ]
}

function App() {
  const [currentSection, setCurrentSection] = useState('main')
  const [guestbook, setGuestbook] = useState([])
  const [newMessage, setNewMessage] = useState({ name: '', message: '' })
  const [dDay, setDDay] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [attendance, setAttendance] = useState([])
  const [attendanceForm, setAttendanceForm] = useState({ name: '', phone: '', attending: '', guestCount: 1 })

  // D-Day 계산
  useEffect(() => {
    const weddingDate = new Date('2024-12-14')
    const today = new Date()
    const diffTime = weddingDate - today
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
    setDDay(diffDays)
  }, [])

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
    const address = encodeURIComponent(weddingData.wedding.address)
    const urls = {
      kakao: `https://map.kakao.com/link/search/${address}`,
      naver: `https://map.naver.com/v5/search/${address}`,
      google: `https://maps.google.com/maps?q=${address}`
    }
    window.open(urls[type], '_blank')
  }

  // 방명록 추가 함수
  const addGuestbookEntry = () => {
    if (newMessage.name && newMessage.message) {
      setGuestbook([...guestbook, { 
        ...newMessage, 
        id: Date.now(),
        date: new Date().toLocaleDateString()
      }])
      setNewMessage({ name: '', message: '' })
      alert('축하 메시지가 등록되었습니다.')
    }
  }

  // 캘린더에 추가 함수
  const addToCalendar = () => {
    const startDate = '20241214T140000'
    const endDate = '20241214T170000'
    const title = encodeURIComponent('김민수 ♥ 이지영 결혼식')
    const location = encodeURIComponent(weddingData.wedding.address)
    
    const googleCalendarUrl = `https://calendar.google.com/calendar/render?action=TEMPLATE&text=${title}&dates=${startDate}/${endDate}&location=${location}`
    window.open(googleCalendarUrl, '_blank')
  }

  // 참석 여부 확인 함수
  const submitAttendance = () => {
    if (attendanceForm.name && attendanceForm.phone && attendanceForm.attending) {
      setAttendance([...attendance, { 
        ...attendanceForm, 
        id: Date.now(),
        date: new Date().toLocaleDateString()
      }])
      setAttendanceForm({ name: '', phone: '', attending: '', guestCount: 1 })
      alert('참석 여부가 등록되었습니다.')
    } else {
      alert('모든 필수 항목을 입력해주세요.')
    }
  }

  // 배경음악 토글 함수
  const toggleMusic = () => {
    setIsPlaying(!isPlaying)
    // 실제 음악 재생/정지 로직은 여기에 구현
    if (!isPlaying) {
      alert('배경음악이 재생됩니다.')
    } else {
      alert('배경음악이 정지됩니다.')
    }
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
            
            <div className="wedding-gallery">
              {weddingData.gallery.map((image) => (
                <Dialog key={image.id}>
                  <DialogTrigger asChild>
                    <img
                      src={image.src}
                      alt={image.alt}
                      className="cursor-pointer hover:scale-105 transition-transform duration-300"
                    />
                  </DialogTrigger>
                  <DialogContent className="max-w-4xl w-full p-0">
                    <div className="relative">
                      <img
                        src={image.src}
                        alt={image.alt}
                        className="w-full h-auto max-h-[80vh] object-contain"
                      />
                    </div>
                  </DialogContent>
                </Dialog>
              ))}
            </div>
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
                />
                <Textarea
                  placeholder="축하 메시지를 남겨주세요"
                  value={newMessage.message}
                  onChange={(e) => setNewMessage({...newMessage, message: e.target.value})}
                  rows={3}
                />
                <Button 
                  onClick={addGuestbookEntry}
                  className="wedding-button w-full"
                >
                  메시지 등록
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
                />
                <Input
                  placeholder="연락처를 입력해주세요"
                  value={attendanceForm.phone}
                  onChange={(e) => setAttendanceForm({...attendanceForm, phone: e.target.value})}
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
                >
                  참석 여부 등록
                </Button>
              </div>
              
              <div className="mt-4 p-3 bg-wedding-secondary rounded-lg">
                <h4 className="font-semibold mb-2">참석 현황</h4>
                <p className="text-sm">
                  총 {attendance.filter(a => a.attending === 'yes').length}명 참석 예정
                </p>
                <p className="text-sm">
                  총 인원: {attendance.filter(a => a.attending === 'yes').reduce((sum, a) => sum + a.guestCount, 0)}명
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

export default App

