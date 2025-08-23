"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Trash2, Move, RotateCcw } from 'lucide-react'

export function GalleryGrid({ images, settings, isEditing, onImageDelete, onImageMove, onImageResize }) {
  const [draggedImage, setDraggedImage] = useState(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [resizingImage, setResizingImage] = useState(null)
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0 })
  const [gridLayout, setGridLayout] = useState({ cellWidth: 100, cellHeight: 100 })
  const gridRef = useRef(null)

  const { gridCols, gridRows } = settings

  // 격자 레이아웃 계산
  const calculateGridLayout = () => {
    const gridContainer = gridRef.current
    if (!gridContainer) return { cellWidth: 100, cellHeight: 100 }

    const containerWidth = gridContainer.offsetWidth
    const containerHeight = Math.min(600, containerWidth / gridCols * gridRows)
    
    const cellWidth = containerWidth / gridCols
    const cellHeight = containerHeight / gridRows

    return { cellWidth, cellHeight }
  }

  // 격자 레이아웃 업데이트
  useEffect(() => {
    const updateLayout = () => {
      const layout = calculateGridLayout()
      setGridLayout(layout)
    }
    
    updateLayout()
    window.addEventListener('resize', updateLayout)
    return () => window.removeEventListener('resize', updateLayout)
  }, [gridCols, gridRows])

  // 특정 위치와 크기가 다른 이미지와 겹치는지 확인
  const isOverlapping = (x, y, width, height, excludeId = null, imagesList = images) => {
    return imagesList.some(img => {
      if (img.id === excludeId) return false
      
      const imgRight = img.position.x + img.gridSize.width
      const imgBottom = img.position.y + img.gridSize.height
      const newRight = x + width
      const newBottom = y + height
      
      return !(x >= imgRight || newRight <= img.position.x || 
               y >= imgBottom || newBottom <= img.position.y)
    })
  }

  // 빈 공간 찾기 (왼쪽부터 채우기)
  const findEmptySpace = (width = 1, height = 1, excludeId = null, imagesList = images) => {
    // 왼쪽 위부터 오른쪽으로 스캔하면서 빈 공간 찾기
    for (let y = 0; y <= gridRows - height; y++) {
      for (let x = 0; x <= gridCols - width; x++) {
        if (!isOverlapping(x, y, width, height, excludeId, imagesList)) {
          return { x, y }
        }
      }
    }
    
    // 빈 공간이 없으면 격자 확장을 고려하여 위치 반환
    return { x: 0, y: gridRows }
  }

  // 모든 이미지를 최적화하여 재배치 (빈 공간 없애기)
  const optimizeLayout = (imagesList = images, excludeId = null) => {
    const optimizedImages = []
    
    // 이미지를 크기 순으로 정렬 (큰 것부터)
    const sortedImages = [...imagesList]
      .filter(img => img.id !== excludeId)
      .sort((a, b) => {
        const aArea = a.gridSize.width * a.gridSize.height
        const bArea = b.gridSize.width * b.gridSize.height
        return bArea - aArea
      })
    
    // 각 이미지를 왼쪽부터 빈 공간에 배치
    sortedImages.forEach(image => {
      const emptySpace = findEmptySpace(
        image.gridSize.width, 
        image.gridSize.height, 
        null, 
        optimizedImages
      )
      
      optimizedImages.push({
        ...image,
        position: emptySpace
      })
    })
    
    return optimizedImages
  }

  // 이미지들을 자동으로 재배치
  const rearrangeImages = (resizedImageId, newSize) => {
    const resizedImage = images.find(img => img.id === resizedImageId)
    if (!resizedImage) return
    
    // 리사이즈된 이미지 제외하고 최적화
    const otherImages = optimizeLayout(images, resizedImageId)
    
    // 리사이즈된 이미지를 위한 빈 공간 찾기
    const emptySpace = findEmptySpace(
      newSize.width, 
      newSize.height, 
      null, 
      otherImages
    )
    
    // 모든 이미지 위치 업데이트
    otherImages.forEach(img => {
      if (img.position.x !== images.find(orig => orig.id === img.id)?.position.x ||
          img.position.y !== images.find(orig => orig.id === img.id)?.position.y) {
        onImageMove(img.id, img.position)
      }
    })
    
    // 리사이즈된 이미지 업데이트
    onImageResize(resizedImageId, newSize)
    onImageMove(resizedImageId, emptySpace)
  }

  // 드래그 시작
  const handleDragStart = (e, imageId) => {
    if (!isEditing) return

    const rect = e.currentTarget.getBoundingClientRect()
    const offsetX = e.clientX - rect.left
    const offsetY = e.clientY - rect.top

    setDraggedImage(imageId)
    setDragOffset({ x: offsetX, y: offsetY })
    
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', imageId)
  }

  // 드래그 오버
  const handleDragOver = (e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }

  // 드롭
  const handleDrop = (e) => {
    e.preventDefault()
    if (!draggedImage || !isEditing) return

    const { cellWidth, cellHeight } = gridLayout
    const rect = gridRef.current.getBoundingClientRect()
    
    const dropX = e.clientX - rect.left - dragOffset.x
    const dropY = e.clientY - rect.top - dragOffset.y

    // 격자 좌표로 변환
    const gridX = Math.max(0, Math.min(gridCols - 1, Math.round(dropX / cellWidth)))
    const gridY = Math.max(0, Math.min(gridRows - 1, Math.round(dropY / cellHeight)))

    const draggedImg = images.find(img => img.id === draggedImage)
    if (!draggedImg) return

    // 새 위치에서 격자 범위 체크
    const maxX = gridCols - draggedImg.gridSize.width
    const maxY = gridRows - draggedImg.gridSize.height
    const finalX = Math.max(0, Math.min(maxX, gridX))
    const finalY = Math.max(0, Math.min(maxY, gridY))

    // 이동 후 전체 레이아웃 최적화
    onImageMove(draggedImage, { x: finalX, y: finalY })
    
    // 잠시 후에 레이아웃 최적화 실행
    setTimeout(() => {
      const optimizedImages = optimizeLayout()
      optimizedImages.forEach(img => {
        const originalImg = images.find(orig => orig.id === img.id)
        if (originalImg && 
            (img.position.x !== originalImg.position.x || img.position.y !== originalImg.position.y)) {
          onImageMove(img.id, img.position)
        }
      })
    }, 100)
    
    setDraggedImage(null)
  }

  // 리사이즈 시작
  const handleResizeStart = (e, imageId) => {
    if (!isEditing) return
    
    e.preventDefault()
    e.stopPropagation()
    
    const image = images.find(img => img.id === imageId)
    if (!image) return

    setResizingImage(imageId)
    setResizeStart({ x: e.clientX, y: e.clientY })

    const startSize = { ...image.gridSize }
    let lastUpdateTime = 0

    const handleMouseMove = (moveEvent) => {
      const now = Date.now()
      if (now - lastUpdateTime < 50) return // 스로틀링으로 부드러운 움직임
      lastUpdateTime = now

      const { cellWidth, cellHeight } = gridLayout
      const deltaX = moveEvent.clientX - resizeStart.x
      const deltaY = moveEvent.clientY - resizeStart.y

      // 더 부드러운 크기 조절 (0.3 격자 단위로)
      const cellsX = Math.round(deltaX / (cellWidth * 0.3))
      const cellsY = Math.round(deltaY / (cellHeight * 0.3))
      
      const newWidth = Math.max(1, startSize.width + cellsX)
      const newHeight = Math.max(1, startSize.height + cellsY)

      // 격자 범위 체크
      const maxWidth = gridCols - image.position.x
      const maxHeight = gridRows - image.position.y

      const finalWidth = Math.min(newWidth, maxWidth)
      const finalHeight = Math.min(newHeight, maxHeight)

      // 크기가 변경된 경우 자동 재배치
      if (finalWidth !== image.gridSize.width || finalHeight !== image.gridSize.height) {
        rearrangeImages(imageId, { width: finalWidth, height: finalHeight })
      }
    }

    const handleMouseUp = () => {
      setResizingImage(null)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      
      // 최종 레이아웃 최적화
      setTimeout(() => {
        const optimizedImages = optimizeLayout()
        optimizedImages.forEach(img => {
          const originalImg = images.find(orig => orig.id === img.id)
          if (originalImg && 
              (img.position.x !== originalImg.position.x || img.position.y !== originalImg.position.y)) {
            onImageMove(img.id, img.position)
          }
        })
      }, 100)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  // 이미지 자동 정렬 (현재 순서 유지, 빈칸만 채우기)
  const autoArrangeImages = () => {
    const optimizedImages = optimizeLayout()
    optimizedImages.forEach(img => {
      onImageMove(img.id, img.position)
    })
  }

  // 이미지 렌더링
  const renderImage = (image) => {
    const { cellWidth, cellHeight } = gridLayout
    
    const style = {
      position: 'absolute',
      left: image.position.x * cellWidth,
      top: image.position.y * cellHeight,
      width: image.gridSize.width * cellWidth,
      height: image.gridSize.height * cellHeight,
      transition: draggedImage === image.id || resizingImage === image.id ? 'none' : 'all 0.3s ease',
      zIndex: draggedImage === image.id ? 10 : 1
    }

    return (
      <div
        key={image.id}
        style={style}
        className={`group rounded-lg overflow-hidden shadow-md ${
          isEditing ? 'cursor-move hover:shadow-lg' : ''
        } ${draggedImage === image.id ? 'opacity-70' : ''} ${
          resizingImage === image.id ? 'ring-2 ring-blue-400' : ''
        }`}
        draggable={isEditing}
        onDragStart={(e) => handleDragStart(e, image.id)}
      >
        <img
          src={image.src}
          alt={image.alt}
          className="w-full h-full object-cover pointer-events-none"
        />
        
        {/* 편집 모드 컨트롤 */}
        {isEditing && (
          <>
            {/* 삭제 버튼 */}
            <button
              onClick={() => onImageDelete(image.id)}
              className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600 z-20"
            >
              <Trash2 size={14} />
            </button>

            {/* 이동 표시 */}
            <div className="absolute top-2 left-2 bg-blue-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
              <Move size={14} />
            </div>

            {/* 리사이즈 핸들 */}
            <div
              className="absolute bottom-0 right-0 w-8 h-8 bg-blue-500 hover:bg-blue-600 opacity-0 group-hover:opacity-100 transition-all duration-200 z-20 flex items-end justify-end rounded-tl-lg"
              onMouseDown={(e) => handleResizeStart(e, image.id)}
              style={{
                cursor: 'se-resize'
              }}
            >
              {/* 리사이즈 표시 아이콘 */}
              <div className="w-4 h-4 border-r-2 border-b-2 border-white opacity-80 mb-1 mr-1" />
            </div>

            {/* 격자 크기 표시 */}
            <div className="absolute bottom-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
              {image.gridSize.width}×{image.gridSize.height}
            </div>
          </>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* 자동 정렬 버튼 */}
      {isEditing && (
        <div className="flex justify-end">
          <button
            onClick={autoArrangeImages}
            className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 transition-colors"
          >
            <RotateCcw size={14} className="inline mr-1" />
            자동 정렬
          </button>
        </div>
      )}

      {/* 격자 프리뷰 */}
      <div
        ref={gridRef}
        className={`relative w-full bg-gray-100 rounded-lg overflow-hidden ${
          isEditing ? 'border-2 border-dashed border-gray-300' : ''
        }`}
        style={{
          height: `${Math.min(600, (gridRef.current?.offsetWidth || 400) / gridCols * gridRows)}px`
        }}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {/* 격자 가이드 (편집 모드에만 표시) */}
        {isEditing && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ zIndex: 0 }}
          >
            {/* 세로 선 */}
            {Array.from({ length: gridCols + 1 }, (_, i) => (
              <line
                key={`v-${i}`}
                x1={`${(i / gridCols) * 100}%`}
                y1="0%"
                x2={`${(i / gridCols) * 100}%`}
                y2="100%"
                stroke="#e5e7eb"
                strokeWidth="1"
              />
            ))}
            {/* 가로 선 */}
            {Array.from({ length: gridRows + 1 }, (_, i) => (
              <line
                key={`h-${i}`}
                x1="0%"
                y1={`${(i / gridRows) * 100}%`}
                x2="100%"
                y2={`${(i / gridRows) * 100}%`}
                stroke="#e5e7eb"
                strokeWidth="1"
              />
            ))}
          </svg>
        )}

        {/* 이미지들 */}
        {images.map(renderImage)}
      </div>

      {/* 이미지 목록 (편집 모드에만 표시) */}
      {isEditing && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">이미지 목록</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {images.map((image) => (
              <div key={image.id} className="relative group">
                <img
                  src={image.src}
                  alt={image.alt}
                  className="w-full h-16 object-cover rounded border"
                />
                <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity rounded flex items-center justify-center">
                  <button
                    onClick={() => onImageDelete(image.id)}
                    className="bg-red-500 text-white p-1 rounded-full hover:bg-red-600"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-white text-xs px-1 py-0.5 rounded-b">
                  {image.position.x},{image.position.y} ({image.gridSize.width}×{image.gridSize.height})
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
} 