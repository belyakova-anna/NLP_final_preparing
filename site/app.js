/* ============================================================
   NLP Tickets — Reader app
   ============================================================ */

const TICKETS = [
  { n: 1,  q1: "KV Cache",                                  q2: "Mixture of Experts" },
  { n: 2,  q1: "BERT and BERT-like language models",        q2: "RLHF" },
  { n: 3,  q1: "Rotary positional embeddings",              q2: "GPT and GPT-like language models" },
  { n: 4,  q1: "Grouped-query attention",                   q2: "RETRO algorithm" },
  { n: 5,  q1: "FlashAttention",                            q2: "Architecture of agent systems: planner, executor, orchestrator" },
  { n: 6,  q1: "Main components of an LLM agent",           q2: "GPT: architecture, training" },
  { n: 7,  q1: "In-context learning: Few-shot, CoT",        q2: "Scaled dot-product, Bahdanau, Luong attention" },
  { n: 8,  q1: "Proximal Policy Optimization (PPO)",        q2: "Self-, cross-, multi-head attention" },
  { n: 9,  q1: "Language models as world models",           q2: "Direct Preference Optimization (DPO)" },
  { n: 10, q1: "Quantization",                              q2: "RAG. Rewriting, reranking. Agentic RAG" },
  { n: 11, q1: "Role of skills in agentic systems",         q2: "LLaMA architecture" },
  { n: 12, q1: "Definition of an LLM agent",                q2: "(IA)³, A3" },
  { n: 13, q1: "Types of LLM agent actions",                q2: "BERT architecture and training" },
  { n: 14, q1: "RETRO algorithm",                           q2: "Generated-Knowledge Prompting, Self-Consistency" },
  { n: 15, q1: "Rotary positional embeddings",              q2: "Scaling laws of LLMs" },
  { n: 16, q1: "Thought–Action–Observation workflow",       q2: "Sampling: Beam-search, Top-K, Temperature" },
  { n: 17, q1: "Key components of a reinforcement learning system", q2: "Alignment vs. Instruction Tuning vs. Fine-tuning" },
  { n: 18, q1: "Planning in agents",                        q2: "KNN-LM algorithm" },
  { n: 19, q1: "Model Context Protocol (MCP) for agent tools", q2: "PEFT. Low-Rank Adaptation (LoRA)" },
  { n: 20, q1: "LLM-as-judge",                              q2: "Group Relative Policy Optimization (GRPO)" },
];

const pad2 = (n) => String(n).padStart(2, "0");
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));

// ------------------------------------------------------------
// Build sidebar list
// ------------------------------------------------------------
function buildRail() {
  const list = $("#ticketList");
  list.innerHTML = TICKETS.map(t => `
    <a class="rail-item" href="#/${pad2(t.n)}" data-n="${pad2(t.n)}">
      <span class="rail-num">${pad2(t.n)}</span>
      <span class="rail-titles">
        <span class="rail-q">${escapeHtml(t.q1)}</span>
        <span class="rail-q">${escapeHtml(t.q2)}</span>
      </span>
    </a>
  `).join("");
}

// ------------------------------------------------------------
// Build home grid
// ------------------------------------------------------------
function buildGrid(filter = "") {
  const grid = $("#grid");
  const q = filter.trim().toLowerCase();
  const items = TICKETS.filter(t =>
    !q ||
    t.q1.toLowerCase().includes(q) ||
    t.q2.toLowerCase().includes(q) ||
    String(t.n).includes(q)
  );
  if (items.length === 0) {
    grid.innerHTML = `<li class="grid-empty" style="padding:40px 22px;color:var(--muted);font-family:var(--mono);font-size:13px;">No tickets match “${escapeHtml(filter)}”.</li>`;
    return;
  }
  grid.innerHTML = items.map(t => `
    <li>
      <a class="grid-card" href="#/${pad2(t.n)}">
        <span class="grid-num"><em>№</em>${pad2(t.n)}</span>
        <span class="grid-q"><span class="qlabel">Q1</span>${escapeHtml(t.q1)}</span>
        <span class="grid-q"><span class="qlabel">Q2</span>${escapeHtml(t.q2)}</span>
      </a>
    </li>
  `).join("");
}

// ------------------------------------------------------------
// Article rendering
// ------------------------------------------------------------
async function loadTicket(n) {
  const num = pad2(n);
  const article = $("#article");
  const home = $("#home");
  const pager = $("#pager");

  document.body.dataset.view = "article";
  home.hidden = true;
  article.hidden = false;
  pager.hidden = false;

  $("#crumbs").innerHTML = `
    <a href="#/">Index</a>
    <span class="sep">/</span>
    <span>Ticket ${num}</span>
  `;

  $("#progress").textContent = `${num} / 20`;
  setActiveRail(num);

  // Show skeleton
  article.innerHTML = `<div class="ticket-eyebrow">Ticket № ${num}</div><h1>Loading…</h1>`;

  try {
    const res = await fetch(`./tickets/ticket-${num}.md`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const md = await res.text();

    const html = marked.parse(md, {
      gfm: true,
      breaks: false,
      headerIds: true,
    });

    const t = TICKETS[parseInt(num, 10) - 1];
    article.innerHTML = `
      <div class="ticket-eyebrow">Ticket № ${num} · ${escapeHtml(t.q1)} & ${escapeHtml(t.q2)}</div>
      ${html}
    `;

    // Render math
    if (window.renderMathInElement) {
      renderMathInElement(article, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$",  right: "$",  display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        strict: false,
      });
    }

    buildPager(parseInt(num, 10));
    article.scrollIntoView({ block: "start", behavior: "instant" });
    window.scrollTo({ top: 0 });
    document.title = `№${num} · ${t.q1} — NLP Tickets`;
  } catch (err) {
    article.innerHTML = `
      <div class="ticket-eyebrow" style="color:var(--crimson)">Error</div>
      <h1>Couldn’t load ticket ${num}</h1>
      <p>${escapeHtml(err.message)}</p>
      <p><a href="#/">← Back to index</a></p>
    `;
  }
}

function buildPager(n) {
  const prev = TICKETS[n - 2];
  const next = TICKETS[n];
  const prevEl = $("#prevLink");
  const nextEl = $("#nextLink");

  if (prev) {
    prevEl.classList.remove("disabled");
    prevEl.href = `#/${pad2(prev.n)}`;
    prevEl.querySelector(".pager-title").textContent = `№${pad2(prev.n)} · ${prev.q1}`;
  } else {
    prevEl.classList.add("disabled");
    prevEl.href = "#";
    prevEl.querySelector(".pager-title").textContent = "—";
  }

  if (next) {
    nextEl.classList.remove("disabled");
    nextEl.href = `#/${pad2(next.n)}`;
    nextEl.querySelector(".pager-title").textContent = `№${pad2(next.n)} · ${next.q1}`;
  } else {
    nextEl.classList.add("disabled");
    nextEl.href = "#";
    nextEl.querySelector(".pager-title").textContent = "—";
  }
}

function showHome() {
  document.body.dataset.view = "home";
  $("#home").hidden = false;
  $("#article").hidden = true;
  $("#pager").hidden = true;
  $("#crumbs").innerHTML = `<a href="#/">Index</a>`;
  $("#progress").textContent = `— / 20`;
  setActiveRail(null);
  document.title = `NLP — Final Exam Tickets · 2026`;
  window.scrollTo({ top: 0 });
}

function setActiveRail(num) {
  $$(".rail-item").forEach(a => {
    if (num && a.dataset.n === num) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
}

// ------------------------------------------------------------
// Routing
// ------------------------------------------------------------
function route() {
  const hash = location.hash.replace(/^#\/?/, "");
  if (!hash) { showHome(); return; }
  const n = parseInt(hash, 10);
  if (Number.isInteger(n) && n >= 1 && n <= 20) {
    loadTicket(n);
  } else {
    showHome();
  }
}

// ------------------------------------------------------------
// Keyboard nav
// ------------------------------------------------------------
function onKey(e) {
  if (e.target.matches("input, textarea")) return;
  const hash = location.hash.replace(/^#\/?/, "");
  const cur = parseInt(hash, 10);

  if (e.key === "ArrowRight" && Number.isInteger(cur) && cur < 20) {
    location.hash = `#/${pad2(cur + 1)}`;
  } else if (e.key === "ArrowLeft" && Number.isInteger(cur) && cur > 1) {
    location.hash = `#/${pad2(cur - 1)}`;
  } else if (e.key === "g" || e.key === "G") {
    location.hash = "#/";
  } else if (e.key === "/") {
    if (document.body.dataset.view === "home") {
      e.preventDefault();
      $("#search").focus();
    } else {
      location.hash = "#/";
      setTimeout(() => $("#search").focus(), 50);
    }
  } else if (e.key === "Escape") {
    if (document.body.classList.contains("rail-open")) {
      document.body.classList.remove("rail-open");
    }
  }
}

// ------------------------------------------------------------
// Utils
// ------------------------------------------------------------
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ------------------------------------------------------------
// Init
// ------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  buildRail();
  buildGrid();

  $("#search").addEventListener("input", (e) => buildGrid(e.target.value));

  $("#randomTicket").addEventListener("click", (e) => {
    e.preventDefault();
    const r = 1 + Math.floor(Math.random() * 20);
    location.hash = `#/${pad2(r)}`;
  });

  $("#railToggle").addEventListener("click", () => {
    document.body.classList.toggle("rail-open");
    const open = document.body.classList.contains("rail-open");
    $("#railToggle").setAttribute("aria-expanded", String(open));
  });

  // close rail on nav (mobile)
  $("#rail").addEventListener("click", (e) => {
    if (e.target.closest(".rail-item, .brand")) {
      document.body.classList.remove("rail-open");
    }
  });

  window.addEventListener("hashchange", route);
  window.addEventListener("keydown", onKey);

  route();
});
