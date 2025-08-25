// src/components/PricingTAI.jsx
import React, { useState } from "react";

export default function PricingTAI() {
  const [annual, setAnnual] = useState(false);

  const tiers = [
    {
      name: "Pay-as-you-go",
      price: "10 TAI",
      suffix: "/meme (example)",
      note: "On-demand actions: memes, reports, API calls",
      cta: "Top-up with TAI",
      href: "#waitlist",
      features: [
        "Micro-transactions in TAI",
        "No subscription required",
        "0.2% burn per tx (example)",
      ],
      highlighted: false,
    },
    {
      name: "Pro",
      price: annual ? "54,000 TAI" : "5,000 TAI",
      suffix: annual ? "/year" : "/month",
      note: "Advanced dashboards • API (medium rate)",
      cta: annual ? "Choose Pro (Annual –10%)" : "Choose Pro (Monthly)",
      href: "#waitlist",
      features: [
        "Advanced dashboards & analytics",
        "100 AI content generations / month",
        "API access (medium rate)",
        "Staking rewards boost",
      ],
      highlighted: true,
    },
    {
      name: "Enterprise",
      price: "Custom",
      suffix: " (TAI-based)",
      note: "Unlimited + white-label + priority support",
      cta: "Contact Sales",
      href: "mailto:xpteam.tn@gmail.com?subject=TrendAI%20Enterprise",
      features: [
        "Unlimited dashboards & reports",
        "Unlimited AI content generation",
        "Dedicated API & SLA",
        "White-label reports • Priority support",
      ],
      highlighted: false,
    },
  ];

  return (
    <section className="relative max-w-6xl mx-auto px-6">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <p className="uppercase tracking-[0.18em] text-white/50 text-xs">Pricing in TAI</p>
          <h2 className="mt-2 text-2xl md:text-3xl font-semibold tracking-tight">Simple & flexible</h2>
          <p className="mt-2 text-white/70 text-sm">
            Subscriptions are TAI-denominated and may float with market value. Annual prepay gets a discount.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className={`text-sm ${annual ? "text-white/60" : "text-white"}`}>Monthly</span>
          <button
            onClick={() => setAnnual(a => !a)}
            aria-label="Toggle monthly/annual"
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
            Annual <span className="text-[color:var(--brand-cyan)]">(save ~10%)</span>
          </span>
        </div>
      </div>

      <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {tiers.map((t, i) => {
          const colSpan = "lg:col-span-4";
          const colStart = i === 0 ? "lg:col-start-1" : i === 1 ? "lg:col-start-5" : "lg:col-start-9";
          return (
            <div key={t.name} className={`${colStart} ${colSpan}`}>
              <article className={`relative rounded-3xl p-6 border bg-white/[0.04] border-white/10 backdrop-blur-md ${t.highlighted ? "bg-[#0d1126]" : ""}`}>
                {t.highlighted && (
                  <span
                    className="absolute -top-3 right-4 px-2 py-1 rounded-full text-[10px] font-semibold text-black"
                    style={{ background: "var(--grad-brand)" }}
                  >
                    Most popular
                  </span>
                )}
                <header className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">{t.name}</h3>
                </header>
                <div className="mt-3">
                  <div className="text-3xl md:text-4xl font-extrabold leading-none">
                    {t.price} <span className="text-sm md:text-base font-normal text-white/70">{t.suffix}</span>
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
                  className={`mt-6 inline-flex w-full items-center justify-center gap-2 px-4 py-2.5 rounded-xl ${t.highlighted ? "text-black" : "bg-white/10 border border-white/10"} font-medium hover:opacity-95 transition`}
                  style={t.highlighted ? { background: "var(--grad-brand)" } : {}}
                >
                  {t.cta}
                </a>
              </article>
            </div>
          );
        })}
      </div>

      <p className="mt-4 text-xs text-white/55">
        Example on-demand: 10&nbsp;TAI per meme • 0.2% burn per transaction. Pro Annual example: 54,000&nbsp;TAI (10% off 12×5,000).
        NFT-based VIP passes (permanent or time-locked) planned for future phases.
      </p>
    </section>
  );
}
