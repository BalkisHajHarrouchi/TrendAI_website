// src/components/Header.jsx
import React, { useEffect, useState } from 'react';
import { Link } from 'react-scroll';

const sections = [
  { id: 'home', label: 'Home' },
  { id: 'crypto-trend', label: 'Crypto Trend' },
  { id: 'video-gen', label: 'Video Gen' },
  { id: 'img-gen', label: 'Img Gen' }
];

const Header = () => {
  const [activeSection, setActiveSection] = useState('home');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      {
        root: null,
        rootMargin: '0px',
        threshold: 0.6
      }
    );

    sections.forEach((section) => {
      const el = document.getElementById(section.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <header className="fixed top-0 w-full z-50 backdrop-blur-md bg-black/30 border-b border-white/10 shadow-md">
      <nav className="max-w-7xl mx-auto px-6 py-3 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">TrendAI</h1>
        <ul className="flex gap-6 text-white">
          {sections.map((section) => (
            <li key={section.id}>
              <Link
                to={section.id}
                smooth={true}
                duration={600}
                offset={-80}
                className={`cursor-pointer transition-colors duration-300 text-sm md:text-base font-medium ${
                  activeSection === section.id
                    ? 'text-cyan-400'
                    : 'text-white hover:text-cyan-200'
                }`}
              >
                {section.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
};

export default Header;
