import { useEffect, useRef, useState } from "react";
import "./landing.css";

// Inline SVG icons (no dependency). lucide-style 24x24 stroke paths.
const Svg = ({ children, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{children}</svg>
);
const Pulse = (p) => <Svg {...p}><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></Svg>;
const Layers = (p) => <Svg {...p}><path d="M12 2 2 7l10 5 10-5-10-5Z" /><path d="m2 17 10 5 10-5" /><path d="m2 12 10 5 10-5" /></Svg>;
const Sliders = (p) => <Svg {...p}><line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" /><line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" /><line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" /><line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line x1="17" y1="16" x2="23" y2="16" /></Svg>;
const Shield = (p) => <Svg {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" /><path d="m9 12 2 2 4-4" /></Svg>;
const Zap = (p) => <Svg {...p}><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z" /></Svg>;
const Arrow = (p) => <Svg {...p}><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></Svg>;

const features = [
  { icon: Layers, title: "Three live sources", body: "Every question fans out to PubMed, OpenAlex, and ClinicalTrials.gov in parallel — ~170 de-duplicated studies per query, never a stale index." },
  { icon: Sliders, title: "Domain-tuned ranking", body: "BM25 + PubMedBERT embeddings fused by RRF, re-ranked by NCBI's MedCPT cross-encoder, then source-balanced — the right studies rise to the top." },
  { icon: Shield, title: "Cite or abstain", body: "Every claim carries its title, authors, year, and a supporting snippet. If the evidence isn't there, Curalink says so instead of inventing it." },
  { icon: Zap, title: "Real-time streaming", body: "Watch the pipeline work — expansion, retrieval, ranking, reasoning — and read the answer token by token as it's written." },
];

const steps = [
  { n: "01", title: "Ask", body: "Set the disease and intent once, then ask in plain language." },
  { n: "02", title: "Retrieve & rank", body: "Three sources fetched live, de-duplicated, and ranked by a medical-domain funnel." },
  { n: "03", title: "Grounded answer", body: "A structured, source-cited answer streams straight back to you." },
];

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Fades + rises its children into view the first time they're scrolled to.
function Reveal({ children, className = "" }) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (prefersReducedMotion()) { setShown(true); return; }
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setShown(true); io.disconnect(); } },
      { threshold: 0.15, rootMargin: "0px 0px -8% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return <div ref={ref} className={`l-reveal ${shown ? "is-visible" : ""} ${className}`}>{children}</div>;
}

const DEMO_Q = "Latest clinical trials for Parkinson's?";
const DEMO_A = [
  "Overview — several Phase II/III trials are recruiting, spanning deep brain stimulation and disease-modifying therapies.",
  "🧪 NCT05… · Deep Brain Stimulation — Recruiting · Toronto, CA",
  "📄 GLP-1 agonists show neuroprotective signals — cited, PubMed 2024.",
].join("\n");

// Types the question, pauses, streams the answer, then loops.
function ChatDemo() {
  const [q, setQ] = useState("");
  const [a, setA] = useState("");
  const [phase, setPhase] = useState("typing"); // typing | thinking | streaming | done

  useEffect(() => {
    if (prefersReducedMotion()) { setQ(DEMO_Q); setA(DEMO_A); setPhase("done"); return; }
    let cancelled = false;
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    const run = async () => {
      while (!cancelled) {
        setQ(""); setA(""); setPhase("typing");
        for (let i = 1; i <= DEMO_Q.length && !cancelled; i++) { setQ(DEMO_Q.slice(0, i)); await wait(40); }
        if (cancelled) return;
        setPhase("thinking"); await wait(850);
        if (cancelled) return;
        setPhase("streaming");
        for (let i = 1; i <= DEMO_A.length && !cancelled; i++) { setA(DEMO_A.slice(0, i)); await wait(14); }
        if (cancelled) return;
        setPhase("done"); await wait(3000);
      }
    };
    run();
    return () => { cancelled = true; };
  }, []);

  const lines = a.split("\n");
  return (
    <div className="l-mock-body">
      <div className="l-msg l-msg-user">
        {q || " "}
        {phase === "typing" && <span className="l-caret" />}
      </div>
      {phase !== "typing" && (
        <div className="l-msg l-msg-bot">
          {phase === "thinking" ? (
            <span className="l-thinking">Searching PubMed, OpenAlex, ClinicalTrials<span className="l-caret" /></span>
          ) : (
            lines.map((line, i) => {
              const isLast = i === lines.length - 1;
              return (
                <div key={i} className={i === 0 ? undefined : "l-prod"}>
                  {line}
                  {isLast && phase === "streaming" && <span className="l-caret" />}
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

export default function LandingPage({ onGetStarted, onSignIn }) {
  return (
    <div className="landing">
      <nav className="l-nav">
        <div className="l-container l-nav-inner">
          <button className="l-brand" onClick={onGetStarted}>
            <Pulse size={18} />
            <span>Curalink</span>
          </button>
          <div className="l-nav-actions">
            <button className="l-btn l-btn-ghost" onClick={onSignIn}>Sign in</button>
            <button className="l-btn l-btn-primary" onClick={onGetStarted}>Get started</button>
          </div>
        </div>
      </nav>

      <header className="l-container l-hero">
        <div className="l-eyebrow">AI MEDICAL RESEARCH ASSISTANT</div>
        <h1 className="l-display">Research answers,<br />backed by sources.</h1>
        <p className="l-lead">
          Ask about any disease. Curalink retrieves live studies from PubMed, OpenAlex, and
          ClinicalTrials.gov, reasons over them, and streams back structured, source-cited
          answers — or abstains rather than guess.
        </p>
        <div className="l-hero-cta">
          <button className="l-btn l-btn-primary l-btn-lg" onClick={onGetStarted}>
            Get started <Arrow size={16} />
          </button>
          <button className="l-btn l-btn-secondary l-btn-lg" onClick={onSignIn}>Sign in</button>
        </div>

        <div className="l-mock">
          <div className="l-mock-bar">
            <span className="l-dot" /><span className="l-dot" /><span className="l-dot" />
            <span className="l-mock-title">Curalink Research Assistant</span>
          </div>
          <ChatDemo />
        </div>
      </header>

      <section className="l-container l-section">
        <Reveal>
          <div className="l-section-head">
            <div className="l-eyebrow">WHY IT'S DIFFERENT</div>
            <h2 className="l-h2">Not a chatbot — a research system.</h2>
          </div>
        </Reveal>
        <Reveal>
          <div className="l-grid">
            {features.map((f) => (
              <div className="l-card" key={f.title}>
                <div className="l-card-icon"><f.icon size={18} /></div>
                <h3 className="l-card-title">{f.title}</h3>
                <p className="l-card-body">{f.body}</p>
              </div>
            ))}
          </div>
        </Reveal>
      </section>

      <section className="l-container l-section">
        <Reveal>
          <div className="l-section-head">
            <div className="l-eyebrow">HOW IT WORKS</div>
            <h2 className="l-h2">Three steps, one question.</h2>
          </div>
        </Reveal>
        <Reveal>
          <div className="l-steps">
            {steps.map((s) => (
              <div className="l-step" key={s.n}>
                <div className="l-step-n">{s.n}</div>
                <h3 className="l-card-title">{s.title}</h3>
                <p className="l-card-body">{s.body}</p>
              </div>
            ))}
          </div>
        </Reveal>
      </section>

      <section className="l-container l-cta-wrap">
        <Reveal>
          <div className="l-cta">
            <h2 className="l-h2">Start researching.</h2>
            <p className="l-lead l-cta-lead">Create an account and ask your first question in seconds.</p>
            <button className="l-btn l-btn-primary l-btn-lg" onClick={onGetStarted}>
              Get started <Arrow size={16} />
            </button>
          </div>
        </Reveal>
      </section>

      <footer className="l-footer">
        <div className="l-container l-footer-inner">
          <span className="l-foot-meta">
            © {new Date().getFullYear()} Curalink · Research information, not medical advice.
          </span>
        </div>
      </footer>
    </div>
  );
}
