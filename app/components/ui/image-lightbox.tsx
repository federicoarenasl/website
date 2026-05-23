'use client'

import React, { useEffect, useState, useCallback, useRef } from 'react'

type ImageLightboxProps = {
  src: string
  alt?: string
  width?: number | string
  height?: number | string
  isSvg?: boolean
}

const MIN_SCALE = 1
const MAX_SCALE = 8
const DOUBLE_CLICK_SCALE = 2.5

type Transform = { scale: number; x: number; y: number }

const IDENTITY: Transform = { scale: 1, x: 0, y: 0 }

export function ImageLightbox({ src, alt, width, height, isSvg }: ImageLightboxProps) {
  const [open, setOpen] = useState(false)
  const [transform, setTransform] = useState<Transform>(IDENTITY)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const overlayRef = useRef<HTMLDivElement | null>(null)

  // Pointer-drag state
  const dragRef = useRef<{ active: boolean; startX: number; startY: number; originX: number; originY: number }>({
    active: false, startX: 0, startY: 0, originX: 0, originY: 0,
  })

  // Pinch state: distance and midpoint at gesture start
  const pinchRef = useRef<{ active: boolean; startDist: number; startScale: number; centerX: number; centerY: number; originX: number; originY: number } | null>(null)

  const close = useCallback(() => {
    setOpen(false)
    setTransform(IDENTITY)
  }, [])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
      if (e.key === '0') setTransform(IDENTITY)
    }
    document.addEventListener('keydown', onKey)
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = previousOverflow
    }
  }, [open, close])

  // Wheel zoom centered on cursor. Attached via native listener so we can preventDefault (React's onWheel is passive).
  useEffect(() => {
    if (!open) return
    const overlay = overlayRef.current
    if (!overlay) return

    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = overlay.getBoundingClientRect()
      const cx = e.clientX - rect.left - rect.width / 2
      const cy = e.clientY - rect.top - rect.height / 2
      const factor = Math.exp(-e.deltaY * 0.0015)
      setTransform((t) => zoomAt(t, factor, cx, cy))
    }
    overlay.addEventListener('wheel', onWheel, { passive: false })
    return () => overlay.removeEventListener('wheel', onWheel as EventListener)
  }, [open])

  function zoomAt(t: Transform, factor: number, cx: number, cy: number): Transform {
    const nextScale = clamp(t.scale * factor, MIN_SCALE, MAX_SCALE)
    const f = nextScale / t.scale
    // Keep the point under (cx, cy) stationary: new translate = c - (c - old) * f
    const x = cx - (cx - t.x) * f
    const y = cy - (cy - t.y) * f
    return constrain({ scale: nextScale, x, y })
  }

  function constrain(t: Transform): Transform {
    // When zoomed out to 1, snap pan back to 0 so the image re-centers.
    if (t.scale <= MIN_SCALE + 0.001) return { scale: MIN_SCALE, x: 0, y: 0 }
    return t
  }

  function clamp(n: number, min: number, max: number) {
    return Math.min(max, Math.max(min, n))
  }

  const onPointerDown = (e: React.PointerEvent<HTMLImageElement>) => {
    if (pinchRef.current?.active) return
    if (transform.scale <= 1.001) return
    ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
    dragRef.current = {
      active: true,
      startX: e.clientX,
      startY: e.clientY,
      originX: transform.x,
      originY: transform.y,
    }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLImageElement>) => {
    if (!dragRef.current.active) return
    const dx = e.clientX - dragRef.current.startX
    const dy = e.clientY - dragRef.current.startY
    setTransform((t) => ({ ...t, x: dragRef.current.originX + dx, y: dragRef.current.originY + dy }))
  }

  const endDrag = (e: React.PointerEvent<HTMLImageElement>) => {
    dragRef.current.active = false
    ;(e.target as HTMLElement).releasePointerCapture?.(e.pointerId)
  }

  const onDoubleClick = (e: React.MouseEvent<HTMLImageElement>) => {
    e.stopPropagation()
    const overlay = overlayRef.current
    if (!overlay) return
    const rect = overlay.getBoundingClientRect()
    const cx = e.clientX - rect.left - rect.width / 2
    const cy = e.clientY - rect.top - rect.height / 2
    setTransform((t) => {
      if (t.scale > 1.05) return IDENTITY
      const factor = DOUBLE_CLICK_SCALE / t.scale
      return zoomAt(t, factor, cx, cy)
    })
  }

  // Touch pinch handlers (use native listeners so we can preventDefault on touchmove)
  useEffect(() => {
    if (!open) return
    const overlay = overlayRef.current
    if (!overlay) return

    const distance = (a: Touch, b: Touch) => {
      const dx = a.clientX - b.clientX
      const dy = a.clientY - b.clientY
      return Math.hypot(dx, dy)
    }

    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        const rect = overlay.getBoundingClientRect()
        const a = e.touches[0]
        const b = e.touches[1]
        const midX = (a.clientX + b.clientX) / 2 - rect.left - rect.width / 2
        const midY = (a.clientY + b.clientY) / 2 - rect.top - rect.height / 2
        pinchRef.current = {
          active: true,
          startDist: distance(a, b),
          startScale: transform.scale,
          centerX: midX,
          centerY: midY,
          originX: transform.x,
          originY: transform.y,
        }
        dragRef.current.active = false
      }
    }

    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && pinchRef.current?.active) {
        e.preventDefault()
        const a = e.touches[0]
        const b = e.touches[1]
        const dist = distance(a, b)
        const factor = dist / pinchRef.current.startDist
        const targetScale = clamp(pinchRef.current.startScale * factor, MIN_SCALE, MAX_SCALE)
        const f = targetScale / pinchRef.current.startScale
        const { centerX: cx, centerY: cy, originX: ox, originY: oy } = pinchRef.current
        setTransform(constrain({
          scale: targetScale,
          x: cx - (cx - ox) * f,
          y: cy - (cy - oy) * f,
        }))
      }
    }

    const onTouchEnd = (e: TouchEvent) => {
      if (e.touches.length < 2 && pinchRef.current) {
        pinchRef.current.active = false
        pinchRef.current = null
      }
    }

    overlay.addEventListener('touchstart', onTouchStart, { passive: false })
    overlay.addEventListener('touchmove', onTouchMove, { passive: false })
    overlay.addEventListener('touchend', onTouchEnd)
    overlay.addEventListener('touchcancel', onTouchEnd)
    return () => {
      overlay.removeEventListener('touchstart', onTouchStart as EventListener)
      overlay.removeEventListener('touchmove', onTouchMove as EventListener)
      overlay.removeEventListener('touchend', onTouchEnd as EventListener)
      overlay.removeEventListener('touchcancel', onTouchEnd as EventListener)
    }
  }, [open, transform.scale, transform.x, transform.y])

  const onOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Only close when the backdrop itself is clicked (not bubbled from image/caption/buttons).
    if (e.target === e.currentTarget) close()
  }

  const isZoomed = transform.scale > 1.001
  const imgCursor = isZoomed ? (dragRef.current.active ? 'grabbing' : 'grab') : 'zoom-in'

  return (
    <>
      <img
        alt={alt || ''}
        src={src}
        onClick={() => setOpen(true)}
        className="rounded-lg block cursor-zoom-in"
        style={{
          maxWidth: '100%',
          width: '100%',
          height: 'auto',
          objectFit: 'contain',
          ...(isSvg && {
            maxHeight: 'none',
            WebkitTransform: 'translateZ(0)',
          }),
          display: 'block',
        }}
        width={width as number | undefined}
        {...(isSvg ? {} : { height: height as number | undefined })}
        loading="lazy"
      />

      {open && (
        <div
          ref={overlayRef}
          role="dialog"
          aria-modal="true"
          aria-label={alt || 'Expanded image'}
          onClick={onOverlayClick}
          className="fixed inset-0 z-50 flex flex-col items-center justify-center p-4 sm:p-8 select-none"
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            touchAction: 'none',
            overscrollBehavior: 'contain',
          }}
        >
          <button
            type="button"
            onClick={close}
            aria-label="Close"
            className="absolute top-4 right-4 text-white text-3xl leading-none font-light hover:opacity-70"
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', zIndex: 2 }}
          >
            &times;
          </button>

          <div
            style={{
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              maxWidth: '95vw',
              maxHeight: alt ? '78vh' : '90vh',
              overflow: 'hidden',
            }}
          >
            <img
              ref={imgRef}
              alt={alt || ''}
              src={src}
              draggable={false}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              onDoubleClick={onDoubleClick}
              onClick={(e) => e.stopPropagation()}
              className="rounded-lg"
              style={{
                maxWidth: '95vw',
                maxHeight: alt ? '78vh' : '90vh',
                width: 'auto',
                height: 'auto',
                objectFit: 'contain',
                transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
                transformOrigin: 'center center',
                transition: dragRef.current.active || pinchRef.current?.active ? 'none' : 'transform 80ms ease-out',
                cursor: imgCursor,
                willChange: 'transform',
                userSelect: 'none',
                touchAction: 'none',
              }}
            />
          </div>

          {alt && (
            <p
              onClick={(e) => e.stopPropagation()}
              className="text-sm italic mt-4 text-center max-w-3xl px-4"
              style={{ cursor: 'default', color: '#e5e7eb' }}
            >
              {alt}
            </p>
          )}

          <div
            onClick={(e) => e.stopPropagation()}
            className="absolute bottom-3 left-1/2 text-xs"
            style={{
              transform: 'translateX(-50%)',
              color: 'rgba(229, 231, 235, 0.55)',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
            }}
          >
            scroll or pinch to zoom · drag to pan · double-click to toggle · 0 to reset · esc to close
          </div>
        </div>
      )}
    </>
  )
}
