import Link from 'next/link'
import dynamic from 'next/dynamic'

const navItems = {
  '/': {
    name: 'home',
  },
  '/projects': {
    name: 'projects',
  },
  '/thoughts': {
    name: 'thoughts',
  },
}

const P5Wrapper = dynamic(() => import('./p5wrapper'), { ssr: false });


export function Navbar() {
  return (
    <aside className="-ml-3 mb-4 tracking-tight">
      <div className="lg:sticky lg:top-20">
        <nav
          className="flex flex-row items-center justify-between relative px-0 pb-0 fade md:overflow-auto scroll-pr-6 md:relative w-full"
          id="nav"
        >
          {/* Navigation Links (Left Side) */}
          <div className="flex flex-row space-x-6 items-center">
            {Object.entries(navItems).map(([path, { name }]) => {
              return (
                <Link
                  key={path}
                  href={path}
                  className="transition-all hover:text-neutral-800 dark:hover:text-neutral-200 flex align-middle relative py-1 px-2 m-1"
                >
                  {name}
                </Link>
              );
            })}
          </div>

          {/* Animated Sphere (Right Side, Fixed) */}
          <div className="items-center justify-center w-[80px] h-[80px] overflow-hidden">
            <P5Wrapper />
          </div>
        </nav>
      </div>
    </aside>
  );
}

