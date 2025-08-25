// pages/Home.jsx
import React, { useState } from "react";
import Banner3DSlider from "../components/Banner3DSlider";

const API_URL = import.meta?.env?.VITE_RAG_API_URL || "http://localhost:8000/api/ask";
// Mini RAG ask widget (TOP-LEVEL — do not nest inside another component)
// put this at top-level (not nested inside another component/file scope)


export function AskTrendAIMini() {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [ans, setAns] = useState("");
  const [sources, setSources] = useState([]);

  const ask = async (e) => {
    e.preventDefault();
    if (!q.trim()) return;

    setLoading(true);
    setAns("");
    setSources([]);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();

      // ensure answer is a string
      setAns(typeof data?.answer === "string" ? data.answer : JSON.stringify(data?.answer || ""));

      // normalize sources: allow strings or objects
// normalize to { title, preview } but accept old shapes too
const normalized = Array.isArray(data?.sources)
  ? data.sources.map((s, idx) => {
      if (typeof s === "string") {
        return { title: `Document ${idx + 1}`, preview: s };
      }
      if (s && typeof s === "object") {
        return {
          title: s.title ?? s.source ?? `Document ${idx + 1}`,
          preview: s.preview ?? s.snippet ?? s.text ?? "",
        };
      }
      return { title: `Document ${idx + 1}`, preview: "" };
    })
  : [];
setSources(normalized);

    } catch (err) {
      setAns("Sorry, something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 rounded-xl bg-black/40 border border-white/10 p-4">
      <form onSubmit={ask} className="flex gap-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask about features, pricing, alerts…"
          className="w-full rounded-lg bg-black/50 border border-white/10 px-3 py-2 outline-none focus:border-cyan-400"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-lg px-4 py-2 text-black font-semibold disabled:opacity-60"
          style={{ background: "var(--grad-brand)" }}
        >
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      {/* Answer */}
      {ans && (
        <div className="mt-4">
          <p className="text-sm text-white/60">Answer</p>
          <div className="mt-1 text-sm whitespace-pre-wrap">{ans}</div>
        </div>
      )}

      {/* Sources */}
{/* Sources */}
{sources.length > 0 && (
  <div className="mt-4">
    <p className="text-sm text-white/60">Sources</p>
    <ul className="mt-2 space-y-2">
      {sources.map((s, i) => (
        <li key={i} className="rounded-lg bg-white/5 border border-white/10 p-3">
          <div className="text-xs font-semibold text-white/80">
            {s.title || `Document ${i + 1}`}
          </div>
          {s.preview && (
            <div className="mt-1 text-xs text-white/60">
              {s.preview}
            </div>
          )}
        </li>
      ))}
    </ul>
  </div>
)}

    </div>
  );
}



// --- New, modern pricing (drop-in) ---
const PricingSection = () => {
  const [annual, setAnnual] = useState(true);
  const price = (monthly, yearly) => (annual ? yearly : monthly);

  const tiers = [
    {
      name: "Free",
      price: "$0",
      suffix: "",
      note: "read-only demo · 1 agent · 200 drafts/mo",
      cta: "Start free",
      href: "#waitlist",
      featured: false,
      features: [
        "Demo feeds & basic insights",
        "200 drafts / mo",
        "Community support",
      ],
    },
    {
      name: "Pro",
      price: `$${price(39, 31)}`,
      suffix: "/mo",
      note: "live feed · alerts · 5 agents",
      cta: "Go Pro",
      href: "#waitlist",
      featured: true, // highlighted middle card
      features: [
        "Live momentum & sentiment",
        "Alerts & watchlists",
        "5 agents included",
      ],
    },
    {
      name: "Team",
      price: `$${price(129, 103)}`,
      suffix: "/mo",
      note: "SSO · roles · audit · webhook/API",
      cta: "Contact sales",
      href: "#waitlist",
      featured: false,
      features: [
        "Team workspaces & roles",
        "Audit trail & SSO",
        "Webhook / API access",
      ],
    },
  ];


  
  return (
    <section id="pricing" className="relative z-10 max-w-6xl mx-auto px-6 py-20 scroll-mt-24">
      {/* header */}
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <p className="uppercase tracking-[0.18em] text-white/50 text-xs">Simple & scalable</p>
          <h2 className="mt-2 text-2xl md:text-3xl font-semibold tracking-tight">Pricing</h2>
          <p className="mt-2 text-white/70 text-sm">Choose a plan that grows with your desk or team.</p>
        </div>

        {/* billing toggle */}
        <div className="flex items-center gap-3">
          <span className={`text-sm ${annual ? "text-white/60" : "text-white"}`}>Monthly</span>
          <button
            onClick={() => setAnnual(a => !a)}
            aria-label="Toggle billing"
            className="relative inline-flex h-9 w-20 items-center rounded-full bg-white/10 border border-white/10"
          >
            <span
              className="absolute h-7 w-7 rounded-full transition-transform"
              style={{
                background: "var(--grad-brand)",
                transform: annual ? "translateX(48px)" : "translateX(4px)",
              }}
            />
          </button>
          <span className={`text-sm ${annual ? "text-white" : "text-white/60"}`}>
            Annual <span className="text-[color:var(--brand-cyan)]">(save ~20%)</span>
          </span>
        </div>
      </div>

      {/* cards in 12-col, featured center */}
      <div className="mt-10 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {tiers.map((t, i) => {
          const baseCard =
            "relative rounded-3xl p-6 border bg-white/[0.04] border-white/10 backdrop-blur-md";
          const featuredWrap = "relative p-[1px] rounded-3xl";
          const featuredRing = { background: "var(--grad-brand)" };
          const colSpan = "lg:col-span-4";
          const colStart = i === 0 ? "lg:col-start-1" : i === 1 ? "lg:col-start-5" : "lg:col-start-9";

          return (
            <div key={t.name} className={`${colStart} ${colSpan}`}>
              {t.featured ? (
                <div className={featuredWrap} style={featuredRing}>
                  <article className={`${baseCard} bg-[#0d1126]`}>
                    <span
                      className="absolute -top-3 right-4 px-2 py-1 rounded-full text-[10px] font-semibold text-black"
                      style={{ background: "var(--grad-brand)" }}
                    >
                      Most popular
                    </span>

                    <header className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold">{t.name}</h3>
                      <span className="w-2.5 h-2.5 rounded-full bg-white/70 shadow-[0_0_12px_rgba(255,255,255,.6)]" />
                    </header>

                    <div className="mt-3">
                      <div className="text-4xl font-extrabold leading-none">
                        {t.price} <span className="text-base font-normal text-white/70">{t.suffix}</span>
                      </div>
                      <p className="text-white/70 text-sm mt-1">{t.note}</p>
                    </div>

                    <ul className="mt-5 text-white/85 text-sm space-y-2">
                      {t.features.map((f) => (
                        <li key={f} className="flex items-start gap-2">
                          <svg width="16" height="16" viewBox="0 0 24 24" className="mt-0.5 opacity-90">
                            <path d="M20 6 9 17l-5-5" fill="none" stroke="currentColor" strokeWidth="2" />
                          </svg>
                          <span>{f}</span>
                        </li>
                      ))}
                    </ul>

                    <a
                      href={t.href}
                      className="mt-6 inline-flex w-full items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-black font-medium hover:opacity-95 transition"
                      style={{ background: "var(--grad-brand)" }}
                    >
                      {t.cta}
                    </a>

                    <p className="mt-3 text-[11px] text-white/50">Integrations & scheduling: coming soon.</p>
                  </article>
                </div>
              ) : (
                <article className={`${baseCard} hover:-translate-y-[2px] transition`}>
                  <header className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">{t.name}</h3>
                  </header>

                  <div className="mt-3">
                    <div className="text-3xl font-extrabold leading-none">
                      {t.price} {t.suffix && <span className="text-sm font-normal text-white/70">{t.suffix}</span>}
                    </div>
                    <p className="text-white/70 text-sm mt-1">{t.note}</p>
                  </div>

                  <ul className="mt-5 text-white/85 text-sm space-y-2">
                    {t.features.map((f) => (
                      <li key={f} className="flex items-start gap-2">
                        <svg width="16" height="16" viewBox="0 0 24 24" className="mt-0.5 opacity-90">
                          <path d="M20 6 9 17l-5-5" fill="none" stroke="currentColor" strokeWidth="2" />
                        </svg>
                        <span>{f}</span>
                      </li>
                    ))}
                  </ul>

                  <a
                    href={t.href}
                    className="mt-6 inline-flex w-full items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 text-sm"
                  >
                    {t.cta}
                  </a>
                </article>
              )}
            </div>
          );
        })}
      </div>

      {/* compact compare row */}
      <div className="mt-10 rounded-2xl overflow-hidden border border-white/10">
        <div className="grid grid-cols-5 bg-white/5 text-[13px] font-medium">
          <div className="px-4 py-3">Compare</div>
          <div className="px-4 py-3">Free</div>
          <div className="px-4 py-3">Pro</div>
          <div className="px-4 py-3">Team</div>
          <div className="px-4 py-3">Notes</div>
        </div>
        {[
          ["Agents", "1", "5", "Unlimited", ""],
          ["Data window", "24h demo", "30d", "90d+", "configurable"],
          ["Alerts", "—", "✔", "✔", "Slack/Email/Webhook"],
          ["Integrations", "—", "X (coming)", "X + API (coming)", "Webhook on Team"],
        ].map((row, i) => (
          <div key={i} className="grid grid-cols-5 text-sm odd:bg-white/[0.03]">
            {row.map((cell, j) => (
              <div key={j} className="px-4 py-2 border-t border-white/10 text-white/80">{cell}</div>
            ))}
          </div>
        ))}
      </div>

      <p className="mt-4 text-xs text-white/50">
        * Prices shown are {annual ? "annual" : "monthly"} billing equivalents. Taxes may apply.
      </p>
    </section>
  );
};



const Home = () => {
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


  // ----- helpers (place in Home.jsx file) -----




  return (
    // CHANGED: use brand background tokens
    <div className="min-h-screen text-white bg-ink-900 bg-night-sky">
      {/* Overview / Top */}
      <div id="top" className="pt-24 scroll-mt-24">
        <Banner3DSlider
          images={carouselImages}
          title=""
          subtitle=""
          authorTitle=""
          authorRole=""
          authorText=""
          modelImage={"/images/final.png"}
        />

        {/* Intro */}
        <section
          id="intro"
          className="relative z-10 max-w-4xl mx-auto px-6 pb-12 text-center scroll-mt-24"
        >
          <p className="uppercase tracking-[0.18em] text-white/60 text-xs md:text-sm">
            AI-crypto radar
          </p>

          <h2 className="mt-3 text-2xl md:text-4xl font-semibold">
            Ride signals, not noise.
          </h2>

          <p className="mt-3 text-base md:text-lg text-white/75">
            Spot momentum early, track narratives, and publish in minutes.
          </p>

          <div className="mt-6 flex items-center justify-center gap-3">
            {/* CHANGED: primary CTA uses brand gradient */}
            <a
              href="#waitlist"
              className="px-5 py-2.5 rounded-xl font-semibold text-black bg-brand-gradient shadow-md hover:opacity-95 focus:outline-none focus:ring-2 focus:ring-brand-cyan/60 transition"
            >
              Get Early Access
            </a>
            <a
              href="#how"
              className="px-5 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 font-semibold transition"
            >
              See How it Works
            </a>
          </div>
        </section>
      </div>
{/* ASK TRENDAI (RAG) — mini box */}
<section id="ask-trendai" className="relative z-10 max-w-6xl mx-auto px-6 py-12 scroll-mt-24">
  <div className="rounded-2xl border border-white/10 bg-white/[0.05] backdrop-blur-md p-5">
    <div className="flex items-center justify-between">
      <div>
        <p className="uppercase tracking-[0.18em] text-white/50 text-xs">Ask TrendAI</p>
        <h3 className="mt-1 text-xl font-semibold tracking-tight">Have a question about TrendAI?</h3>
      </div>
      <span className="hidden md:inline-flex items-center px-3 py-1 rounded-full text-[11px] font-medium text-black"
        style={{ background: "var(--grad-brand)" }}>
        Assistant • RAG
      </span>
    </div>

    {/* mini box */}
    <AskTrendAIMini />
  </div>
</section>

      {/* HOW IT WORKS */}
<section id="how" className="relative z-10 max-w-6xl mx-auto px-6 py-12 scroll-mt-24">
  <h2 className="text-2xl md:text-3xl font-bold mb-6">How it works</h2>

  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
    {/* Ingest */}
    <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
      <div className="flex items-center gap-3">
        <span className="w-9 h-9 rounded-xl bg-brand-cyan/15 flex items-center justify-center text-brand-cyan">
          {/* cloud arrow */}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25" />
            <path d="M12 12v9" /><path d="M8 16l4 4 4-4" />
          </svg>
        </span>
        <h3 className="text-lg font-semibold">Ingest</h3>
      </div>
      <p className="mt-3 text-white/80">
        Stream newest AI-crypto tweets & on-chain/news signals.
      </p>
    </div>

    {/* Understand */}
    <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
      <div className="flex items-center gap-3">
        <span className="w-9 h-9 rounded-xl bg-[color:var(--brand-violet)]/15 flex items-center justify-center text-[color:var(--brand-violet)]">
          {/* brain */}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M7 8a4 4 0 1 1 4-4" /><path d="M17 8a4 4 0 1 0-4-4" />
            <path d="M7 8v8a4 4 0 0 0 4 4" /><path d="M17 8v8a4 4 0 0 1-4 4" />
          </svg>
        </span>
        <h3 className="text-lg font-semibold">Understand</h3>
      </div>
      <p className="mt-3 text-white/80">
        Sarcasm-aware sentiment + theme/topic clustering + trend labels.
      </p>
    </div>

    {/* Act */}
    <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
      <div className="flex items-center gap-3">
        <span className="w-9 h-9 rounded-xl bg-[color:var(--brand-pink)]/15 flex items-center justify-center text-[color:var(--brand-pink)]">
          {/* rocket */}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 15a7 7 0 0 0 4 4" />
            <path d="M6 9l8-7 3 3-7 8-4 1 1-4z" />
            <path d="M14 4l6 6" />
          </svg>
        </span>
        <h3 className="text-lg font-semibold">Act</h3>
      </div>
      <p className="mt-3 text-white/80">
        Agentic post generator & scheduled threads—review, edit, publish.
      </p>
    </div>
  </div>
</section>


{/* CORE FEATURES (uniform typography) */}
<section
  id="features"
  className="relative z-10 max-w-6xl mx-auto px-6 py-16 scroll-mt-24"
>
  {/* soft section aura */}
  <div
    aria-hidden
    className="pointer-events-none absolute inset-x-0 -top-10 h-[240px] blur-3xl opacity-60"
    style={{
      background:
        "radial-gradient(600px 180px at 20% 0%, color-mix(in oklab, var(--brand-cyan) 22%, transparent) 0%, transparent 70%), radial-gradient(600px 180px at 80% 0%, color-mix(in oklab, var(--brand-pink) 18%, transparent) 0%, transparent 70%)",
    }}
  />

  <div className="flex items-center justify-between gap-4">
    <div>
      <p className="uppercase tracking-[0.18em] text-white/50 text-xs">
        What you get
      </p>
      <h2 className="mt-2 text-2xl md:text-3xl font-semibold tracking-tight">
        Core features
      </h2>
    </div>
    <span
      className="hidden md:inline-flex items-center px-3 py-1 rounded-full text-xs font-medium text-black"
      style={{ background: "var(--grad-brand)" }}
    >
      Built for AI-crypto speed
    </span>
  </div>

  <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    {/* 1) Live Narrative Radar */}
    <article className="group relative rounded-3xl p-[1px] bg-gradient-to-br from-[color:var(--brand-cyan)] via-[color:var(--brand-violet)] to-[color:var(--brand-pink)]">
      <div className="rounded-3xl h-full bg-black/40 backdrop-blur-md border border-white/10 p-6 group-hover:translate-y-[-2px] transition">
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center">
            {/* radar icon */}
            <svg width="20" height="20" viewBox="0 0 24 24" className="opacity-90">
              <path d="M12 20a8 8 0 1 0-8-8" fill="none" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 6a6 6 0 1 1-6 6" fill="none" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 12l7-7" fill="none" stroke="currentColor" strokeWidth="2" />
            </svg>
          </span>
          <h3 className="text-base md:text-lg font-semibold tracking-tight">Live Narrative Radar</h3>
        </div>
        <ul className="mt-4 text-white/80 text-sm leading-relaxed space-y-2">
          <li>• Streams tweets plus key news/on-chain signals in near real time</li>
          <li>• Alias normalization and spam/dup filtering for clean token feeds</li>
          <li>• Real-time prices and 24h/7d changes from CoinMarketCap and CoinGecko</li>
        </ul>
        <p className="mt-3 text-xs text-white/50">Data sources: CoinMarketCap · CoinGecko</p>
      </div>
    </article>

    {/* 2) Signal Engine */}
    <article className="group relative rounded-3xl p-[1px] bg-gradient-to-br from-[color:var(--brand-violet)] via-[color:var(--brand-purple)] to-[color:var(--brand-cyan)]">
      <div className="rounded-3xl h-full bg-black/40 backdrop-blur-md border border-white/10 p-6 group-hover:translate-y-[-2px] transition">
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center">
            {/* brain icon */}
            <svg width="20" height="20" viewBox="0 0 24 24" className="opacity-90">
              <path d="M8 7a4 4 0 1 1 4-4" fill="none" stroke="currentColor" strokeWidth="2"/>
              <path d="M16 7a4 4 0 1 0-4-4" fill="none" stroke="currentColor" strokeWidth="2"/>
              <path d="M8 7v9a4 4 0 0 0 4 4" fill="none" stroke="currentColor" strokeWidth="2"/>
              <path d="M16 7v9a4 4 0 0 1-4 4" fill="none" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </span>
          <h3 className="text-base md:text-lg font-semibold tracking-tight">Signal Engine</h3>
        </div>
        <ul className="mt-4 text-white/80 text-sm leading-relaxed space-y-2">
          <li>• Sarcasm-aware sentiment and theme/topic clustering</li>
          <li>• Token-level trend labels: Good / Bad / Neutral</li>
          <li>• Narrative maps and word clouds per token</li>
        </ul>
      </div>
    </article>

    {/* 3) Viral Post Studio */}
    <article className="group relative rounded-3xl p-[1px] bg-gradient-to-br from-[color:var(--brand-pink)] via-[color:var(--brand-violet)] to-[color:var(--brand-cyan)]">
      <div className="rounded-3xl h-full bg-black/40 backdrop-blur-md border border-white/10 p-6 group-hover:translate-y-[-2px] transition">
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center">
            {/* megaphone */}
            <svg width="20" height="20" viewBox="0 0 24 24" className="opacity-90">
              <path d="M3 11l14-5v12L3 13z" fill="none" stroke="currentColor" strokeWidth="2"/>
              <path d="M7 14v4a2 2 0 0 0 2 2h1" fill="none" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </span>
          <h3 className="text-base md:text-lg font-semibold tracking-tight">Viral Post Studio</h3>
        </div>
        <ul className="mt-4 text-white/80 text-sm leading-relaxed space-y-2">
          <li>• Drafts tweets/threads from positive-momentum tokens</li>
          <li>• Smart hashtag suggestions and hook ideas</li>
          <li>• Image and short-video generation to match the post’s angle</li>
        </ul>
      </div>
    </article>

    {/* 4) Meme Lab */}
    <article className="group relative rounded-3xl p-[1px] bg-gradient-to-br from-[color:var(--brand-cyan)] via-[color:var(--brand-pink)] to-[color:var(--brand-purple)]">
      <div className="rounded-3xl h-full bg-black/40 backdrop-blur-md border border-white/10 p-6 group-hover:translate-y-[-2px] transition">
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center">
            {/* image/smile */}
            <svg width="20" height="20" viewBox="0 0 24 24" className="opacity-90">
              <rect x="3" y="5" width="18" height="14" rx="2" fill="none" stroke="currentColor" strokeWidth="2"/>
              <path d="M8 13s1.5 2 4 2 4-2 4-2" fill="none" stroke="currentColor" strokeWidth="2"/>
              <circle cx="9" cy="10" r="1" /><circle cx="15" cy="10" r="1" />
            </svg>
          </span>
          <h3 className="text-base md:text-lg font-semibold tracking-tight">Meme Lab</h3>
        </div>
        <ul className="mt-4 text-white/80 text-sm leading-relaxed space-y-2">
          <li>• Upload an image to generate a timely crypto meme</li>
          <li>• Fetches what’s trending now to shape the joke</li>
          <li>• Agent returns the edited meme image and caption</li>
        </ul>
      </div>
    </article>
  </div>
</section>
{/* INSIGHTS GALLERY — SPLIT HERO + STACKED SIDEBAR (larger hero type) */}
<section id="insights" className="relative z-10 max-w-6xl mx-auto px-6 py-16 scroll-mt-24">
  {/* soft aura, unique to insights */}
  <div
    aria-hidden
    className="pointer-events-none absolute inset-x-0 -top-10 h-[240px] blur-3xl"
    style={{
      background:
        "radial-gradient(600px 200px at 20% 0%, color-mix(in oklab, var(--brand-cyan) 24%, transparent) 0%, transparent 70%), radial-gradient(600px 200px at 80% 0%, color-mix(in oklab, var(--brand-violet) 20%, transparent) 0%, transparent 70%)",
      opacity: .65
    }}
  />

  <div className="flex items-center justify-between gap-4">
    <div>
      <p className="uppercase tracking-[0.18em] text-white/50 text-xs">Recent narratives</p>
      <h2 className="mt-2 text-2xl md:text-3xl font-semibold tracking-tight">Insights gallery</h2>
    </div>
    <span
      className="hidden md:inline-flex items-center px-3 py-1 rounded-full text-xs font-medium text-black"
      style={{ background: "var(--grad-brand)" }}
    >
      Auto-refreshed snapshots
    </span>
  </div>

  {/* Split layout: hero left, two stacked cards right */}
  <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-6">
    {/* HERO (left) — bigger type */}
    <article className="lg:col-span-7 relative rounded-3xl border border-white/10 bg-white/[0.06] backdrop-blur-xl p-7 shadow-[0_0_0_1px_rgba(255,255,255,0.05)_inset]">
      <div aria-hidden className="absolute left-0 right-0 -top-px h-[2px]" style={{ background: "var(--grad-brand)" }} />
      <div className="flex items-center justify-between">
        <h3 className="text-xl md:text-2xl lg:text-3xl font-semibold tracking-tight">NEAR narrative spike</h3>
        <svg width="240" height="56" viewBox="0 0 240 56" className="opacity-90">
          <path d="M2,40 L40,34 L78,44 L116,24 L154,20 L192,28 L216,18 238,24" fill="none" stroke="currentColor" strokeWidth="2"/>
        </svg>
      </div>
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <ul className="text-sm md:text-base text-white/80 space-y-1.5">
          <li>• Mentions +240% in 6h</li>
          <li>• Sentiment flips to Good</li>
        </ul>
        <ul className="text-sm md:text-base text-white/80 space-y-1.5">
          <li>• Velocity ↑; #AI + #infra cluster</li>
          <li>• Influencer overlap rising</li>
        </ul>
      </div>
      <div className="mt-7 flex items-center gap-5">
        <a href="#how" className="text-sm md:text-base text-brand-cyan hover:underline">See method →</a>
        <a href="#waitlist" className="text-sm md:text-base text-white/70 hover:text-white">Recreate analysis</a>
      </div>
    </article>

    {/* RIGHT SIDEBAR: two stacked tiles (regular type) */}
    <div className="lg:col-span-5 grid grid-cols-1 gap-6">
      <article className="relative rounded-3xl border border-white/10 bg-white/[0.05] backdrop-blur-lg p-5 hover:translate-y-[-2px] transition">
        <div aria-hidden className="absolute left-0 right-0 -top-px h-[2px]" style={{ background: "var(--grad-brand)" }} />
        <div className="flex items-center justify-between">
          <h3 className="font-medium tracking-tight">RNDR funding chatter</h3>
          <svg width="140" height="36" viewBox="0 0 120 36" className="opacity-80">
            <path d="M2,20 L22,18 L42,16 L62,22 L82,26 L102,24" fill="none" stroke="currentColor" strokeWidth="2"/>
          </svg>
        </div>
        <ul className="mt-3 text-sm text-white/80 space-y-1">
          <li>• VC mentions +3 sources</li>
          <li>• Neutral drifting toward Good</li>
        </ul>
        <a href="#waitlist" className="mt-3 inline-block text-sm text-brand-cyan hover:underline">Open snapshot →</a>
      </article>

      <article className="relative rounded-3xl border border-white/10 bg-white/[0.05] backdrop-blur-lg p-5 hover:translate-y-[-2px] transition">
        <div aria-hidden className="absolute left-0 right-0 -top-px h-[2px]" style={{ background: "var(--grad-brand)" }} />
        <div className="flex items-center justify-between">
          <h3 className="font-medium tracking-tight">FET controversy cool-off</h3>
          <svg width="140" height="36" viewBox="0 0 120 36" className="opacity-80">
            <path d="M2,10 L22,16 L42,22 L62,24 L82,28 L102,30" fill="none" stroke="currentColor" strokeWidth="2"/>
          </svg>
        </div>
        <ul className="mt-3 text-sm text-white/80 space-y-1">
          <li>• Negativity −35% w/w</li>
          <li>• Mentions normalizing to baseline</li>
        </ul>
        <a href="#waitlist" className="mt-3 inline-block text-sm text-brand-cyan hover:underline">Open snapshot →</a>
      </article>
    </div>
  </div>
</section>


{/* USE CASES — SOLID (2 + 1 inverted pyramid) */}
<section id="use-cases" className="relative z-10 max-w-6xl mx-auto px-6 py-16 scroll-mt-24">
  <div className="flex items-center justify-between gap-4">
    <h2 className="text-2xl md:text-3xl font-semibold tracking-tight">Use cases</h2>
    <span className="hidden md:inline-flex items-center px-3 py-1 rounded-full text-xs font-medium text-white/80 border border-white/10">
      Solo • Desk • Fund
    </span>
  </div>

  <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-6">

    {/* Trader (same style) */}
    <article className="lg:col-start-2 lg:col-span-5 relative rounded-3xl border border-white/5 bg-[#0e1120] p-6 shadow-xl shadow-black/30">
      {/* left rail accent */}
      <div aria-hidden className="absolute left-0 top-0 bottom-0 w-1.5" style={{ background: "var(--grad-brand)" }} />
      {/* subtle hatch */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.06]"
        style={{ backgroundImage: "repeating-linear-gradient(135deg,#fff 0 1px,transparent 1px 10px)" }}
      />
      <h3 className="text-lg font-semibold">Trader</h3>
      <p className="mt-1 text-white/80 font-medium">Catch momentum early</p>
      <p className="mt-2 text-sm text-white/75">Spot moves fast, filter by tokens/themes, and react in minutes.</p>
      <ul className="mt-4 text-sm text-white/80 space-y-2">
        <li>• Token feed by momentum</li>
        <li>• Quick filters: AI / infra / memes</li>
        <li>• One-click snapshot export</li>
      </ul>
    </article>

    {/* Fund (same style) */}
    <article className="lg:col-start-7 lg:col-span-5 relative rounded-3xl border border-white/5 bg-[#0e1120] p-6 shadow-xl shadow-black/30">
      <div aria-hidden className="absolute left-0 top-0 bottom-0 w-1.5" style={{ background: "var(--grad-brand)" }} />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.06]"
        style={{ backgroundImage: "repeating-linear-gradient(135deg,#fff 0 1px,transparent 1px 10px)" }}
      />
      <h3 className="text-lg font-semibold">Fund</h3>
      <p className="mt-1 text-white/80 font-medium">Monitor sectors at a glance</p>
      <p className="mt-2 text-sm text-white/75">Build watchlists, track sector flows, and generate LP updates instantly.</p>
      <ul className="mt-4 text-sm text-white/80 space-y-2">
        <li>• Sector watchlists</li>
        <li>• Narrative heatmaps</li>
        <li>• LP-ready weekly summaries</li>
      </ul>
    </article>

    {/* Research (now same style as 1 & 2) */}
    <article className="lg:col-start-3 lg:col-span-8 relative rounded-3xl border border-white/5 bg-[#0e1120] p-6 shadow-xl shadow-black/30">
      <div aria-hidden className="absolute left-0 top-0 bottom-0 w-1.5" style={{ background: "var(--grad-brand)" }} />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.06]"
        style={{ backgroundImage: "repeating-linear-gradient(135deg,#fff 0 1px,transparent 1px 10px)" }}
      />
      <h3 className="text-lg font-semibold">Research</h3>
      <p className="mt-1 text-white/80 font-medium">Track narratives & synthesize</p>
      <p className="mt-2 text-sm text-white/75">Compare themes, surface top tweets, and export snapshots for weekly notes.</p>
      <ul className="mt-4 text-sm text-white/80 space-y-2">
        <li>• Side-by-side narrative compare</li>
        <li>• Top tweets & sources surfaced</li>
        <li>• Export research packs</li>
      </ul>
    </article>

  </div>
</section>
{/* PRICING */}
<section id="pricing">
<PricingSection />
</section>

      {/* SECURITY */}
      <section id="security" className="relative z-10 max-w-6xl mx-auto px-6 py-12">
        <h2 className="text-2xl md:text-3xl font-bold mb-6">Security & compliance</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
            <ul className="space-y-3 text-white/80">
              <li className="flex items-start gap-3">
                <span className="mt-1 text-brand-cyan">
                  {/* shield-check */}
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 22c-4-1.5-8-4.5-8-9V6l8-4 8 4v7c0 4.5-4 7.5-8 9z" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </span>
                <span>Non-custodial: we never hold your funds.</span>
              </li>

              <li className="flex items-start gap-3">
                <span className="mt-1 text-brand-cyan">
                  {/* lock */}
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="5" y="11" width="14" height="10" rx="2" />
                    <path d="M7 11V8a5 5 0 0 1 10 0v3" strokeLinecap="round" />
                  </svg>
                </span>
                <span>API keys are stored securely (env) and never logged.</span>
              </li>

              <li className="flex items-start gap-3">
                <span className="mt-1 text-brand-cyan">
                  {/* users */}
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

          <div className="rounded-2xl p-6 bg-white/5 border border-white/10">
            <ul className="space-y-3 text-white/80">
              <li className="flex items-start gap-3">
                <span className="mt-1 text-brand-cyan">
                  {/* clipboard-check */}
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 3h6a2 2 0 0 1 2 2v0a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2v0a2 2 0 0 1 2-2z" />
                    <path d="M9 7H7a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-2" />
                    <path d="M9 14l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
                <span>Audits: integrations reviewed; formal audits planned.</span>
              </li>

              <li className="flex items-start gap-3">
                <span className="mt-1 text-brand-cyan">
                  {/* scale */}
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 3v4m-7 4h14M6 11l-3 5h6l-3-5zm12 0l-3 5h6l-3-5z" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M12 7l-2 4h4l-2-4zM12 21V11" strokeLinecap="round"/>
                  </svg>
                </span>
                <span>Disclaimers: TrendAI provides analytics only, not financial advice.</span>
              </li>

              <li className="flex items-start gap-3">
                <span className="mt-1 text-brand-cyan">
                  {/* shield */}
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
      





     {/* FAQ */}
<section id="faq" className="relative z-10 max-w-6xl mx-auto px-6 py-12 scroll-mt-24">
  <h2 className="text-2xl md:text-3xl font-bold mb-6">FAQ</h2>

  <div className="space-y-4 text-white/85">
    <details className="bg-white/5 border border-white/10 rounded-xl p-4">
      <summary className="cursor-pointer font-semibold">What is TrendAI?</summary>
      <div className="mt-2 text-white/80 space-y-3">
        <p>
          TrendAI is an AI-crypto radar: it ingests tweets plus key news/on-chain signals,
          understands the narrative behind them, and helps you act—fast.
        </p>
        <ul className="list-disc list-inside text-white/75 text-sm space-y-1">
          <li><span className="text-white">Ingest:</span> near-real-time streams for AI/crypto tickers, projects, and themes.</li>
          <li><span className="text-white">Understand:</span> sarcasm-aware sentiment, topic clustering, trend labels, and narrative maps.</li>
          <li><span className="text-white">Act:</span> agentic post & thread drafts, memes, and scheduled publishing.</li>
        </ul>
        <p className="text-sm text-white/70">
          Built for traders, funds, and researchers who want signal over noise.
        </p>
      </div>
    </details>

    <details className="bg-white/5 border border-white/10 rounded-xl p-4">
      <summary className="cursor-pointer font-semibold">How often is data refreshed?</summary>
      <div className="mt-2 text-white/80 space-y-3">
        <p>
          Feeds update continuously; most surfaces refresh throughout the day.
          Derived metrics (sentiment, clusters, trend labels) recompute on rolling windows.
        </p>
        <ul className="list-disc list-inside text-white/75 text-sm space-y-1">
          <li><span className="text-white">Tweets/news ingest:</span> streaming (seconds to minutes).</li>
          <li><span className="text-white">Dashboards:</span> refreshed periodically as new data lands.</li>
          <li><span className="text-white">Lookback window:</span> varies by plan (Free: 24h demo · Pro: ~30d · Team: 90d+).</li>
        </ul>
        <p className="text-sm text-white/70">
          Exact cadence can vary with API limits and source availability.
        </p>
      </div>
    </details>

    <details className="bg-white/5 border border-white/10 rounded-xl p-4">
      <summary className="cursor-pointer font-semibold">Do you support watchlists and alerts?</summary>
      <div className="mt-2 text-white/80 space-y-3">
        <p>
          Yes—watchlists and alerts are rolling out with early access. Pro and Team tiers
          get prioritised access to real-time alerts and integrations.
        </p>
        <ul className="list-disc list-inside text-white/75 text-sm space-y-1">
          <li><span className="text-white">Watchlists:</span> track tokens, themes, influencers, and keywords.</li>
          <li><span className="text-white">Alerts:</span> momentum & sentiment flips, narrative spikes, volume/velocity changes.</li>
          <li><span className="text-white">Delivery:</span> in-app; Team adds Slack/Email/Webhook.</li>
        </ul>
        <p className="text-sm text-white/70">
          Request access via the waitlist to be included in the next rollout.
        </p>
      </div>
    </details>
  </div>
</section>


      {/* CONTACT / WAITLIST */}
      <section id="contact" className="relative z-10 max-w-5xl mx-auto px-6 pb-24 scroll-mt-24">
        <div id="waitlist" className="rounded-2xl p-6 bg-white/5 border border-white/10 text-center">
          <h2 className="text-2xl md:text-3xl font-bold mb-3">Join the waitlist</h2>
          <p className="text-white/80 mb-6">Be the first to try new features and dashboards.</p>

          <form onSubmit={(e) => e.preventDefault()} className="mx-auto max-w-md flex flex-col sm:flex-row gap-3">
            <input
              type="email"
              placeholder="you@example.com"
              className="w-full rounded-xl bg-black/40 border border-white/10 px-4 py-3 outline-none focus:border-brand-cyan"
              required
            />
            <button
              type="submit"
              className="rounded-xl bg-brand-gradient text-black font-semibold px-5 py-3 hover:opacity-95 transition"
            >
              Request Invite
            </button>
          </form>

          <div className="mt-4 text-white/70">
            Prefer email?{" "}
            <a className="text-brand-cyan hover:opacity-90 underline" href="mailto:xpteam.tn@gmail.com?subject=Waitlist">
              xpteam.tn@gmail.com
            </a>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-white/10 bg-black/30">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-white/80">© {new Date().getFullYear()} TrendAI · All rights reserved</div>
          <nav className="flex items-center gap-6 text-white/70">
            <a href="#top" className="hover:text-white">Overview</a>
            <a href="#how" className="hover:text-white">How it works</a>
            <a href="#use-cases" className="hover:text-white">Use cases</a>
            <a href="#security" className="hover:text-white">Security</a>
            <a href="#faq" className="hover:text-white">FAQ</a>
            <a href="#contact" className="hover:text-white">Contact</a>
          </nav>
        </div>
      </footer>
    </div>
  );
};

export default Home;
