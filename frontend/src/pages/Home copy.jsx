import React from "react";
import { Link } from "react-router-dom";
import CryptoMap from "../components/CryptoMap";
import Banner3DSlider from "../components/Banner3DSlider";

const Home = () => {
  // You can mix images & videos here (Banner3DSlider treats .mp4/.webm/.ogg as video)
  const carouselImages = [
    "/videos/luna.mp4",
    "/videos/generated_5.mp4",
    "/videos/robot.mp4",
    "/videos/fight.mp4",
    "/videos/generated_2.mp4",
    "/videos/animation.mp4",
    "/videos/generated_1.mp4",
    "/videos/hello.mp4",
    "/videos/dione.mp4",
    "/videos/generated_3.mp4",
  ];

  // Placeholder numbers for social proof — wire these to your backend when ready
  const stats = [
    { value: "50+", label: "AI tokens tracked" },
    { value: "Daily", label: "Fresh trend scans" },
    { value: "Seconds", label: "To generate a post" },
  ];

  return (
    <div className="text-white min-h-screen bg-gradient-to-b from-sky-300/20 via-blue-900/30 to-[#0a0a0a]">
      {/* Hero / Banner */}
      <Banner3DSlider
        images={carouselImages}
        title=""
        subtitle=""
        authorTitle=""
        authorRole=""
        authorText=""
        modelImage={"/images/hero.png"}
      />

      {/* HERO (headline + sub + CTA) */}
      <section className="relative z-10 max-w-5xl mx-auto pt-10 pb-16 px-6 text-center">
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight">
          Be first to spot <span className="text-cyan-400">AI-crypto</span> trends
        </h1>
        <p className="mt-4 text-lg md:text-xl text-white/80">
          Check live token momentum, sentiment, and generate posts that ride the wave.
        </p>
        <div className="mt-8 flex items-center justify-center gap-4">
          <Link
            to="/cryptotrend"
            className="px-6 py-3 rounded-xl bg-cyan-500/90 hover:bg-cyan-400 text-black font-semibold shadow-md transition"
          >
            Open Dashboard
          </Link>
          <Link
            to="/videogen"
            className="px-6 py-3 rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 font-semibold transition"
          >
            Generate Content
          </Link>
        </div>
      </section>

      {/* HOW IT WORKS (3 steps) */}
      <section className="relative z-10 max-w-6xl mx-auto px-6 py-12">
        <h2 className="text-2xl md:text-3xl font-bold mb-6">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
            <div className="text-cyan-300 text-3xl mb-3">①</div>
            <h3 className="text-lg font-semibold mb-2">Check the dashboard</h3>
            <p className="text-white/80">
              See the newest AI token trends. Specify the tokens and insights you like, and
              customize your filters to focus on what matters.
            </p>
          </div>
          <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
            <div className="text-cyan-300 text-3xl mb-3">②</div>
            <h3 className="text-lg font-semibold mb-2">Generate content instantly</h3>
            <p className="text-white/80">
              Use our social media agent with a simple prompt. It blends live trends with your
              intent to create tweets, images, and videos that can go viral.
            </p>
          </div>
          <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
            <div className="text-cyan-300 text-3xl mb-3">③</div>
            <h3 className="text-lg font-semibold mb-2">Publish & iterate</h3>
            <p className="text-white/80">
              Post, monitor engagement, and refine prompts. Stay ahead as sentiment shifts.
            </p>
          </div>
        </div>
      </section>

      {/* USE CASES (enhanced) */}
<section className="relative z-10 max-w-6xl mx-auto px-6 py-12">
  <h2 className="text-2xl md:text-3xl font-bold mb-6">Use cases</h2>

  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
    {/* Trader */}
    <article className="group relative rounded-3xl p-[1px] bg-gradient-to-br from-cyan-400/40 via-white/10 to-transparent hover:from-cyan-400/70 hover:via-white/20 transition">
      <div className="rounded-3xl h-full bg-black/30 backdrop-blur-md border border-white/10 p-6 group-hover:translate-y-[-2px] transition">
        <div className="flex items-center justify-between">
          <span className="inline-flex items-center gap-2 text-sm px-3 py-1 rounded-full bg-cyan-400/10 text-cyan-300 ring-1 ring-cyan-400/30">
            <span className="w-2 h-2 rounded-full bg-cyan-300" /> Trader
          </span>
          {/* trending-up icon */}
          <svg width="28" height="28" viewBox="0 0 24 24" className="text-cyan-300">
            <path d="M3 17l6-6 4 4 7-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <h3 className="mt-4 text-xl font-semibold">Catch momentum early</h3>
        <p className="mt-2 text-white/80">Spot moves fast, filter by tokens/themes, and react to sentiment in minutes.</p>
        <ul className="mt-4 text-sm text-white/70 space-y-2">
          <li>• Momentum radar & sparklines</li>
          <li>• Theme & token filters</li>
          <li>• One-click post from agent</li>
        </ul>
        <div className="mt-6">
          <Link to="/cryptotrend" className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-500/90 text-black font-semibold hover:bg-cyan-400 transition">
            Open Dashboard
          </Link>
        </div>
      </div>
    </article>

    {/* Research (emphasized center card) */}
    <article className="group relative rounded-3xl p-[1px] bg-gradient-to-br from-fuchsia-400/40 via-cyan-400/30 to-transparent hover:from-fuchsia-400/70 hover:via-cyan-400/50 transition md:scale-[1.02]">
      <div className="rounded-3xl h-full bg-black/30 backdrop-blur-md border border-white/10 p-6 group-hover:translate-y-[-2px] transition">
        <div className="flex items-center justify-between">
          <span className="inline-flex items-center gap-2 text-sm px-3 py-1 rounded-full bg-fuchsia-400/10 text-fuchsia-300 ring-1 ring-fuchsia-400/30">
            <span className="w-2 h-2 rounded-full bg-fuchsia-300" /> Research
          </span>
          {/* search icon */}
          <svg width="28" height="28" viewBox="0 0 24 24" className="text-fuchsia-300">
            <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="2"/>
            <path d="M21 21l-4.35-4.35" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </div>
        <h3 className="mt-4 text-xl font-semibold">Track narratives & synthesize</h3>
        <p className="mt-2 text-white/80">Compare themes, surface top tweets, and export snapshots for weekly notes.</p>
        <ul className="mt-4 text-sm text-white/70 space-y-2">
          <li>• Theme leaderboards</li>
          <li>• Topic & hashtag highlights</li>
          <li>• One-click report snapshot</li>
        </ul>
        <div className="mt-6">
          <Link to="/cryptotrend" className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 transition">
            Explore Themes
          </Link>
        </div>
      </div>
    </article>

    {/* Fund */}
    <article className="group relative rounded-3xl p-[1px] bg-gradient-to-br from-emerald-400/40 via-white/10 to-transparent hover:from-emerald-400/70 hover:via-white/20 transition">
      <div className="rounded-3xl h-full bg-black/30 backdrop-blur-md border border-white/10 p-6 group-hover:translate-y-[-2px] transition">
        <div className="flex items-center justify-between">
          <span className="inline-flex items-center gap-2 text-sm px-3 py-1 rounded-full bg-emerald-400/10 text-emerald-300 ring-1 ring-emerald-400/30">
            <span className="w-2 h-2 rounded-full bg-emerald-300" /> Fund
          </span>
          {/* building icon */}
          <svg width="28" height="28" viewBox="0 0 24 24" className="text-emerald-300">
            <path d="M3 21h18M5 21V9l7-4 7 4v12M9 21v-6h6v6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <h3 className="mt-4 text-xl font-semibold">Monitor sectors at a glance</h3>
        <p className="mt-2 text-white/80">Build watchlists, track sector flows, and generate LP updates instantly.</p>
        <ul className="mt-4 text-sm text-white/70 space-y-2">
          <li>• Sector & basket views</li>
          <li>• Watchlists with alerts (soon)</li>
          <li>• Auto-generated portfolio notes</li>
        </ul>
        <div className="mt-6">
          <Link to="/videogen" className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/90 text-black font-semibold hover:bg-emerald-400 transition">
            Create Update
          </Link>
        </div>
      </div>
    </article>
  </div>
</section>


    {/* SECURITY & COMPLIANCE (icons, no emojis) */}
<section className="relative z-10 max-w-6xl mx-auto px-6 py-12">
  <h2 className="text-2xl md:text-3xl font-bold mb-6">Security & compliance</h2>

  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
    {/* Column 1 */}
    <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
      <ul className="space-y-3 text-white/80">
        <li className="flex items-start gap-3">
          <span className="mt-1 text-cyan-300">
            {/* shield-check */}
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22c-4-1.5-8-4.5-8-9V6l8-4 8 4v7c0 4.5-4 7.5-8 9z" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </span>
          <span>Non-custodial: we never hold your funds.</span>
        </li>

        <li className="flex items-start gap-3">
          <span className="mt-1 text-cyan-300">
            {/* lock */}
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="5" y="11" width="14" height="10" rx="2" />
              <path d="M7 11V8a5 5 0 0 1 10 0v3" strokeLinecap="round" />
            </svg>
          </span>
          <span>API keys are stored securely (env) and never logged.</span>
        </li>

        <li className="flex items-start gap-3">
          <span className="mt-1 text-cyan-300">
            {/* users (RBAC) */}
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="8" cy="8" r="3" />
              <circle cx="16" cy="8" r="3" />
              <path d="M2 20a6 6 0 0 1 12 0" />
              <path d="M14 20a6 6 0 0 1 8 0" />
            </svg>
          </span>
          <span>Role-based access planned for team workspaces.</span>
        </li>
      </ul>
    </div>

    {/* Column 2 */}
    <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
      <ul className="space-y-3 text-white/80">
        <li className="flex items-start gap-3">
          <span className="mt-1 text-cyan-300">
            {/* clipboard-check (audits) */}
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 3h6a2 2 0 0 1 2 2v0a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2v0a2 2 0 0 1 2-2z" />
              <path d="M9 7H7a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-2" />
              <path d="M9 14l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span>Audits: integrations reviewed; formal audits planned.</span>
        </li>

        <li className="flex items-start gap-3">
          <span className="mt-1 text-cyan-300">
            {/* scale (legal/disclaimer) */}
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 3v4m-7 4h14M6 11l-3 5h6l-3-5zm12 0l-3 5h6l-3-5z" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 7l-2 4h4l-2-4zM12 21V11" strokeLinecap="round"/>
            </svg>
          </span>
          <span>Disclaimers: TrendAI provides analytics only, not financial advice.</span>
        </li>

        <li className="flex items-start gap-3">
          <span className="mt-1 text-cyan-300">
            {/* shield (privacy) */}
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22c-4-1.5-8-4.5-8-9V6l8-4 8 4v7c0 4.5-4 7.5-8 9z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </span>
          <span>Privacy: personal data minimized by design.</span>
        </li>
      </ul>
    </div>
  </div>
</section>


      {/* SOCIAL PROOF */}
<section className="relative z-10 max-w-6xl mx-auto px-6 py-12 space-y-8">
  {/* Numbers: full-width card */}
  <div className="rounded-3xl p-8 md:p-10 bg-gradient-to-br from-cyan-500/10 via-white/5 to-transparent border border-white/10 shadow-md">
    <h3 className="text-xl font-semibold mb-6 text-white/90">By the numbers</h3>
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 md:gap-10">
      <div className="text-center">
        <div className="text-6xl md:text-7xl font-extrabold tracking-tight text-cyan-300">50+</div>
        <div className="mt-2 text-sm md:text-base text-white/70">AI tokens tracked</div>
      </div>
      <div className="text-center">
        <div className="text-6xl md:text-7xl font-extrabold tracking-tight text-cyan-300">24/7</div>
        <div className="mt-2 text-sm md:text-base text-white/70">Fresh trend scans</div>
      </div>
      <div className="text-center">
        <div className="text-6xl md:text-7xl font-extrabold tracking-tight text-cyan-300">⟨10s⟩</div>
        <div className="mt-2 text-sm md:text-base text-white/70">From prompt to post</div>
      </div>
    </div>
  </div>

  {/* What early users say */}
  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
    <figure className="rounded-2xl p-6 bg-white/5 border border-white/10">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-black font-bold">A</div>
        <div>
          <div className="font-semibold">Amir · Swing trader</div>
          <div className="text-white/60 text-sm">@alphaamir</div>
        </div>
      </div>
      <blockquote className="text-white/85 leading-relaxed">
        “Checked the dashboard during London open—saw <span className='text-cyan-300'>RNDR</span> heating up.
        The agent drafted a tweet in seconds. Easiest engagement I’ve had this month.”
      </blockquote>
    </figure>

    <figure className="rounded-2xl p-6 bg-white/5 border border-white/10">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-fuchsia-400 to-cyan-500 flex items-center justify-center text-black font-bold">L</div>
        <div>
          <div className="font-semibold">Lina · Research analyst</div>
          <div className="text-white/60 text-sm">@linaresearch</div>
        </div>
      </div>
      <blockquote className="text-white/85 leading-relaxed">
        “Themes are the cheat code. I filtered ‘Agentic AI’ + ‘Infra’ and pulled a clean summary for my weekly note in five minutes.”
      </blockquote>
    </figure>

    <figure className="rounded-2xl p-6 bg-white/5 border border-white/10 md:col-span-2">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-400 to-sky-500 flex items-center justify-center text-black font-bold">M</div>
        <div>
          <div className="font-semibold">Marco · Fund PM</div>
          <div className="text-white/60 text-sm">@marcopm</div>
        </div>
      </div>
      <blockquote className="text-white/85 leading-relaxed">
        “We set a token watchlist, the agent generated posts for updates, and the team stopped doom-scrolling. It’s just faster.”
      </blockquote>
    </figure>
  </div>
</section>


      {/* Your existing Feature Cards */}
      <section className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-10 max-w-6xl mx-auto px-6 pb-12">
        <Link
          to="/cryptotrend"
          className="bg-white/10 p-6 rounded-2xl shadow-md backdrop-blur-md hover:scale-105 transition-all duration-300 hover:bg-cyan-400/20"
        >
          <h2 className="text-xl font-bold mb-2 text-cyan-300">Crypto Trend</h2>
          <p className="text-gray-300 text-sm">
            Analyze real-time crypto market trends using our AI-powered insight engine.
          </p>
        </Link>

        <Link
          to="/videogen"
          className="bg-white/10 p-6 rounded-2xl shadow-md backdrop-blur-md hover:scale-105 transition-all duration-300 hover:bg-cyan-400/20"
        >
          <h2 className="text-xl font-bold mb-2 text-cyan-300">Video Generator</h2>
          <p className="text-gray-300 text-sm">
            Turn prompts into engaging short videos powered by generative AI tools.
          </p>
        </Link>

        <Link
          to="/imagegen"
          className="bg-white/10 p-6 rounded-2xl shadow-md backdrop-blur-md hover:scale-105 transition-all duration-300 hover:bg-cyan-400/20"
        >
          <h2 className="text-xl font-bold mb-2 text-cyan-300">Image Generator</h2>
          <p className="text-gray-300 text-sm">
            Generate futuristic visuals from text using advanced AI image models.
          </p>
        </Link>
      </section>

      {/* Interactive Map */}
      <section className="relative z-10 max-w-6xl mx-auto px-6 pb-24">
        <CryptoMap />
      </section>

      {/* FOOTER */}
      <footer className="border-t border-white/10 bg-black/30">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-white/80">© {new Date().getFullYear()} TrendAI · All rights reserved</div>
          <nav className="flex items-center gap-6 text-white/70">
            <Link to="/" className="hover:text-white">Home</Link>
            <Link to="/cryptotrend" className="hover:text-white">Dashboard</Link>
            <Link to="/videogen" className="hover:text-white">Video Gen</Link>
            <Link to="/imagegen" className="hover:text-white">Img Gen</Link>
          </nav>
        </div>
      </footer>
    </div>
  );
};

export default Home;
