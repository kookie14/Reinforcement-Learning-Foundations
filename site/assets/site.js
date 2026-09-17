/* Progressive enhancement for the static blog: nothing here is required to read a post. */
(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- mermaid diagrams, themed from the CSS tokens ---------- */
  function renderDiagrams() {
    const nodes = document.querySelectorAll("pre.mermaid");
    if (!nodes.length || !window.mermaid) return;
    const css = getComputedStyle(document.documentElement);
    const v = name => css.getPropertyValue(name).trim();
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      fontFamily: v("--font-display"),
      themeVariables: {
        darkMode: false, background: v("--surface"), primaryColor: v("--accent-soft"),
        primaryTextColor: v("--ink"), primaryBorderColor: v("--accent"), lineColor: v("--muted"),
        textColor: v("--ink"), edgeLabelBackground: v("--surface"), fontSize: "15px"
      }
    });
    mermaid.run({ nodes }).catch(() => {});
  }

  /* ---------- reading progress + active table-of-contents entry ---------- */
  function mountReading() {
    const prose = document.getElementById("prose");
    const progress = document.getElementById("progress");
    if (!prose) return;
    const tocLinks = [...document.querySelectorAll(".toc a")];
    const headings = tocLinks.map(a => document.getElementById(a.hash.slice(1))).filter(Boolean);

    function onScroll() {
      const rect = prose.getBoundingClientRect();
      const total = rect.height - window.innerHeight;
      if (progress) progress.style.width = (total > 0 ? Math.min(1, Math.max(0, -rect.top / total)) * 100 : 100) + "%";
      let current = headings[0];
      for (const h of headings) if (h.getBoundingClientRect().top < 140) current = h;
      tocLinks.forEach(a => a.classList.toggle("active", !!current && a.hash === "#" + current.id));
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    for (const a of document.querySelectorAll(".toc-mobile a")) {
      a.addEventListener("click", () => a.closest("details").removeAttribute("open"));
    }
  }

  /* ---------- home page: value iteration on a deterministic 4×6 grid ---------- */
  function mountGridWorld() {
    const grid = document.getElementById("gw");
    if (!grid) return;
    const label = document.getElementById("gw-sweep");
    const R = 4, C = 6, GAMMA = 0.9;
    const goal = "0,5", pit = "1,5", walls = new Set(["1,1", "2,3"]);
    const moves = [[-1, 0, "↑"], [0, 1, "→"], [1, 0, "↓"], [0, -1, "←"]];
    const key = (r, c) => r + "," + c;
    const terminal = k => k === goal || k === pit;

    function sweep(V) {
      const next = V.map(row => row.slice());
      const policy = V.map(row => row.map(() => ""));
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
        const k = key(r, c);
        if (walls.has(k) || terminal(k)) continue;
        let best = -Infinity, arrow = "";
        for (const [dr, dc, a] of moves) {
          let nr = r + dr, nc = c + dc;
          if (nr < 0 || nr >= R || nc < 0 || nc >= C || walls.has(key(nr, nc))) { nr = r; nc = c; }
          const nk = key(nr, nc);
          const reward = nk === goal ? 1 : nk === pit ? -1 : 0;
          const q = reward + GAMMA * (terminal(nk) ? 0 : V[nr][nc]);
          if (q > best + 1e-9) { best = q; arrow = a; }
        }
        next[r][c] = best; policy[r][c] = arrow;
      }
      return { V: next, policy };
    }

    const history = [];
    let V = Array.from({ length: R }, () => Array(C).fill(0));
    history.push({ V, policy: V.map(row => row.map(() => "")) });
    for (let i = 0; i < 30; i++) {
      const step = sweep(V);
      const delta = Math.max(...step.V.flat().map((x, j) => Math.abs(x - V.flat()[j])));
      history.push(step); V = step.V;
      if (delta < 1e-6) break;
    }

    const cells = [];
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const el = document.createElement("div");
      const k = key(r, c);
      el.className = "cell" + (walls.has(k) ? " wall" : k === goal ? " goal" : k === pit ? " pit" : "");
      if (k === goal) el.textContent = "+1";
      if (k === pit) el.textContent = "−1";
      grid.appendChild(el); cells.push({ el, r, c, k });
    }

    function show(i) {
      const { V, policy } = history[i];
      label.textContent = i === history.length - 1 ? `sweep ${i - 1} · converged` : `sweep ${i}`;
      for (const { el, r, c, k } of cells) {
        if (walls.has(k) || terminal(k)) continue;
        const v = V[r][c];
        const mix = Math.round(Math.max(0, v) * 78);
        el.style.setProperty("--mix", mix + "%");
        el.classList.toggle("hot", mix > 46);
        el.innerHTML = policy[r][c]
          ? `<span class="arrow">${policy[r][c]}</span><span class="val">${v.toFixed(2)}</span>`
          : `<span class="val">0.00</span>`;
      }
    }

    let timer = null;
    show(history.length - 1);
    document.getElementById("gw-replay").addEventListener("click", () => {
      clearInterval(timer);
      if (reduceMotion) { show(history.length - 1); return; }
      let i = 0; show(0);
      timer = setInterval(() => { i++; show(i); if (i >= history.length - 1) clearInterval(timer); }, 520);
    });
  }

  mountGridWorld();
  mountReading();
  renderDiagrams();
})();
