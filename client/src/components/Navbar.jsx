import React from 'react';
import { Link, NavLink } from 'react-router-dom';
import HealthBadge from './HealthBadge';

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <div className="h-9 w-9 rounded-full bg-mango-500 flex items-center justify-center shadow-md">
        <span className="text-white text-xl">🥭</span>
      </div>
      <span className="font-semibold text-lg md:text-xl">Mango Pesticide Detector</span>
    </div>
  );
}

export default function Navbar() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/40 bg-white/60 backdrop-blur-md shadow-sm">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4">
        <Link to="/" className="no-underline text-inherit">
          <Logo />
        </Link>
        <nav className="flex items-center gap-2">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `px-3 py-2 rounded-full text-sm font-medium ${isActive ? 'text-white bg-mango-500' : 'text-gray-700 hover:bg-amber-100'}`
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/compare"
            className={({ isActive }) =>
              `px-3 py-2 rounded-full text-sm font-medium ${isActive ? 'text-white bg-mango-500' : 'text-gray-700 hover:bg-amber-100'}`
            }
          >
            Compare
          </NavLink>
          <NavLink
            to="/about"
            className={({ isActive }) =>
              `px-3 py-2 rounded-full text-sm font-medium ${isActive ? 'text-white bg-mango-500' : 'text-gray-700 hover:bg-amber-100'}`
            }
          >
            About
          </NavLink>
          <HealthBadge />
        </nav>
      </div>
    </header>
  );
}
