import { ImageResponse } from 'next/og'

// Geometry of the site mark (see app/icon.svg): a 16-pointed asterisk built
// from 8 bars rotated in even increments. Drawn with plain divs rather than a
// text glyph so it never depends on a font that has the ✺ character.
const BAR_COUNT = 8
const BAR_WIDTH = 14
const BAR_LENGTH = 132

export function GET(request: Request) {
  let url = new URL(request.url)
  let title = url.searchParams.get('title') || 'Federico Arenas'

  return new ImageResponse(
    (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          height: '100%',
          alignItems: 'flex-start',
          justifyContent: 'center',
          backgroundColor: '#ffffff',
          padding: '80px',
        }}
      >
        {/* Site mark — the same glyph used as the favicon */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            width: '180px',
            height: '180px',
            borderRadius: '32px',
            backgroundColor: '#000000',
            marginBottom: '56px',
          }}
        >
          {Array.from({ length: BAR_COUNT }, (_, i) => (
            <div
              key={i}
              style={{
                position: 'absolute',
                width: `${BAR_WIDTH}px`,
                height: `${BAR_LENGTH}px`,
                borderRadius: `${BAR_WIDTH / 2}px`,
                backgroundColor: '#ffffff',
                transform: `rotate(${(i * 180) / BAR_COUNT}deg)`,
              }}
            />
          ))}
        </div>
        <div
          style={{
            display: 'flex',
            fontSize: '68px',
            fontWeight: 700,
            letterSpacing: '-0.02em',
            color: '#000000',
            lineHeight: 1.1,
          }}
        >
          {title}
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
    }
  )
}
