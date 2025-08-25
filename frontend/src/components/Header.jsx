import React, { useEffect, useState } from "react";
import { Link as ScrollLink } from "react-scroll";
import { Link as RouterLink } from "react-router-dom";

const SECTIONS = [
  { id: "top",        label: "Overview" },
  { id: "how",        label: "How it works" },
  { id: "features",   label: "Core features" },
  { id: "insights",   label: "Insights" },
  { id: "use-cases",  label: "Use cases" },
  { id: "pricing",    label: "Pricing" },
  { id: "security",   label: "Security" },
  { id: "faq",        label: "FAQ" },
  { id: "contact",    label: "Contact" },
];


const Header = () => {
  const [activeSection, setActiveSection] = useState("top");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visibles = entries.filter((e) => e.isIntersecting);
        if (visibles.length === 0) return;
        let best = visibles[0];
        for (const e of visibles) {
          if ((e.intersectionRatio || 0) > (best.intersectionRatio || 0)) best = e;
        }
        const id = best.target.id;
        if (id && id !== activeSection) setActiveSection(id);
      },
      { root: null, rootMargin: "-45% 0px -45% 0px", threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] }
    );

    SECTIONS.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [activeSection]);

  return (
    <header className="fixed top-0 inset-x-0 z-50 backdrop-blur-md bg-black/30 border-b border-white/10 shadow-md">
      <nav className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        {/* Brand -> scroll to top */}
        <ScrollLink
          to="top"
          smooth
          duration={600}
          offset={-96}
          className="cursor-pointer text-2xl font-bold text-white"
          onClick={() => setOpen(false)}
        >
          TrendAI
        </ScrollLink>

        {/* Desktop links */}
        <ul className="hidden md:flex gap-6 text-white items-center">
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <ScrollLink
                to={s.id}
                smooth
                duration={600}
                offset={-96}
                className={`cursor-pointer transition-colors duration-300 text-sm md:text-base font-medium ${
                  activeSection === s.id ? "text-cyan-400" : "text-white hover:text-cyan-200"
                }`}
                onClick={() => setOpen(false)}
              >
                {s.label}
              </ScrollLink>
            </li>
          ))}
          {/* CTA -> router to /studio */}
          <li>
            <RouterLink
              to="/studio"
              className="cursor-pointer inline-flex items-center rounded-xl bg-cyan-500/90 hover:bg-cyan-400 text-black font-semibold px-4 py-2 transition"
            >
              Get Early Access
            </RouterLink>
          </li>
        </ul>

        {/* Mobile toggle */}
        <button
          className="md:hidden inline-flex items-center justify-center rounded-lg border border-white/15 p-2 text-white/80"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            {open ? (
              <path d="M18 6L6 18M6 6l12 12" strokeWidth="2" strokeLinecap="round" />
            ) : (
              <path d="M3 6h18M3 12h18M3 18h18" strokeWidth="2" strokeLinecap="round" />
            )}
          </svg>
        </button>
      </nav>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden border-t border-white/10">
          <div className="max-w-7xl mx-auto px-6 py-3 flex flex-col gap-2">
            {SECTIONS.map((s) => (
              <ScrollLink
                key={s.id}
                to={s.id}
                smooth
                duration={600}
                offset={-96}
                onClick={() => setOpen(false)}
                className={`rounded-lg px-2 py-2 text-white/85 hover:bg-white/10 transition ${
                  activeSection === s.id ? "text-cyan-400" : ""
                }`}
              >
                {s.label}
              </ScrollLink>
            ))}
            <RouterLink
              to="/studio"
              onClick={() => setOpen(false)}
              className="mt-1 inline-flex items-center justify-center rounded-lg bg-cyan-500/90 hover:bg-cyan-400 text-black font-semibold px-3 py-2 transition"
            >
              Get Early Access
            </RouterLink>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;
