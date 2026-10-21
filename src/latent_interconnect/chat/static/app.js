// The chat UI of latent_interconnect.chat.
/* AI Mindmeld (chat) frontend. No build step, no dependencies.
 * Contract: examples/guides/chat.md, "HTTP API". The server is stateless about
 * conversations: every turn we send the full message list plus every placeholder it references.
 */
'use strict';
(() => {
  // ------------------------------------------------------------------ utils
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const MARKER_RE = /\[\[([A-Za-z0-9_]{1,16})\]\]/g;
  const now = () => Date.now();
  const uid = () => Math.random().toString(36).slice(2, 10) + now().toString(36);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } },
  };
  const KEY = { convs: 'aimm-chat.v1.convs', current: 'aimm-chat.v1.current', settings: 'aimm-chat.v1.settings', theme: 'aimm-chat.v1.theme', lastConnector: 'aimm-chat.v1.lastConnector', yesNoNoticeOff: 'aimm-chat.v1.yesNoNoticeOff' };

  // Browser state saved under the earlier key names (plachat.v1.*) is copied once to the current names, on this origin.
  // A group is copied only when every one of its new keys is absent (getItem returns null): a present new value, even
  // an empty or malformed one, is never overwritten. Values are copied as stored (no re-encoding) and the old keys are
  // kept. Conversations and the current-conversation pointer form one group, so the pointer always refers to the list
  // it was saved with. The marker is set only after every planned copy succeeded; a failed write removes the new keys
  // this attempt created, leaving them absent, so the next load tries again. Denied storage leaves everything as is.
  const LEGACY_PREFIX = 'plachat.v1.';
  const MIGRATED_KEY = 'aimm-chat.v1.migrated';
  const MIGRATION_GROUPS = [['convs', 'current'], ['settings'], ['theme'], ['lastConnector']];
  function migrateLegacyStorage() {
    let ls;
    try { ls = window.localStorage; if (!ls || ls.getItem(MIGRATED_KEY) !== null) return; } catch (e) { return; }
    const copies = [];
    try {
      for (const group of MIGRATION_GROUPS) {
        if (group.some((n) => ls.getItem(KEY[n]) !== null)) continue;
        for (const n of group) {
          const v = ls.getItem(LEGACY_PREFIX + n);
          if (v !== null) copies.push([KEY[n], v]);
        }
      }
    } catch (e) { return; }
    const written = [];
    try {
      for (const [k, v] of copies) { ls.setItem(k, v); written.push(k); }
      ls.setItem(MIGRATED_KEY, JSON.stringify(now()));
    } catch (e) {
      for (const k of written) { try { ls.removeItem(k); } catch (e2) { /* the next load copies what is still absent */ } }
    }
  }

  const INPUT_TYPE_LABEL = { sequence: 'Sequence', uniprot: 'UniProt accession', smiles: 'SMILES', chebi: 'ChEBI id' };
  const EXAMPLES = {
    sequence: { value: 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL', label: 'lysozyme C' },
    uniprot: { value: 'P69905', label: 'HBA human' },
    smiles: { value: 'CC(=O)Oc1ccccc1C(=O)O', label: 'aspirin' },
    chebi: { value: 'CHEBI:15365', label: 'aspirin' },
  };
  const STAGES = [
    ['encoder_start', 'encoder'],
    ['embedding', 'embed'],
    ['prefill', 'prefill'],
    ['generating', 'generate'],
  ];
  const STAGE_TEXT = {
    encoder_start: 'Revving up encoder',
    embedding: 'Embedding inputs and applying connectors',
    prefill: 'Prefilling the conversation',
    generating: 'Generating',
  };

  // ------------------------------------------------------------------ state
  const state = {
    config: null,
    convs: [],
    cur: null,
    settings: null,
    streaming: null, // {controller, requestId, msg, el, stopRequested}
    editing: false,
    saveTimer: null,
    storageWarned: false,
    phDialog: null, // {editId, controller}
    healthTimer: null,
    models: null, // the door's /door/api/models (the model picker), or null without a door
    modelsTimer: null,
    switching: false, // the label of the model a page switch is loading
    lastSwitchError: null,
    pickerKey: null,
  };

  // ------------------------------------------------------------------ api
  class AuthError extends Error {}
  function errText(data, status) {
    if (!data) return status ? `HTTP ${status}` : 'request failed';
    if (typeof data === 'string') return data;
    if (typeof data.error === 'string' && data.error) return data.error;
    if (typeof data.message === 'string' && data.message) return data.message;
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    return status ? `HTTP ${status}` : 'request failed';
  }
  async function api(path, { method = 'GET', body, signal } = {}) {
    const opt = { method, headers: {}, credentials: 'same-origin', signal };
    if (body !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
    const r = await fetch(path, opt);
    if (r.status === 401) { showLogin(); throw new AuthError('Sign-in required'); }
    return r;
  }
  async function apiJSON(path, o) {
    const r = await api(path, o);
    let data = null;
    try { data = await r.json(); } catch (e) { /* non-JSON */ }
    if (!r.ok) { const e = new Error(errText(data, r.status)); e.data = data; e.status = r.status; throw e; }
    return data;
  }

  // ------------------------------------------------------------------ toasts
  function toast(msg, kind = '') {
    const t = document.createElement('div');
    t.className = 'toast ' + kind;
    t.textContent = msg;
    const host = $('#toasts');
    // anchor bottom-right, just above the composer (clear of the header and the Thinking switch)
    const cw = document.querySelector('.composer-wrap');
    if (cw) {
      const r = cw.getBoundingClientRect();
      host.style.setProperty('--toast-bottom', `${Math.max(16, Math.round(window.innerHeight - r.top + 10))}px`);
      host.style.setProperty('--toast-right', `${Math.max(16, Math.round(window.innerWidth - r.right))}px`);
    }
    host.appendChild(t);
    while (host.children.length > 2) host.firstElementChild.remove();
    setTimeout(() => t.remove(), kind === 'error' || kind === 'warn' ? 6000 : 3200);
  }

  // ------------------------------------------------------------------ theme
  function applyTheme() {
    const t = store.get(KEY.theme, 'auto');
    if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
    else delete document.documentElement.dataset.theme;
    const label = { auto: 'Theme: follows system', light: 'Theme: light', dark: 'Theme: dark' }[t] || 'Theme';
    const b = $('#btn-theme'); b.title = label + ' (click to change)'; b.setAttribute('aria-label', label);
  }
  function cycleTheme() {
    const order = ['auto', 'light', 'dark'];
    const t = store.get(KEY.theme, 'auto');
    store.set(KEY.theme, order[(order.indexOf(t) + 1) % order.length]);
    applyTheme();
    toast($('#btn-theme').getAttribute('aria-label'));
  }

  // ------------------------------------------------------------------ config + connectors
  function connectors() { return (state.config && state.config.connectors) || []; }
  function connById(id) { return connectors().find((c) => c.id === id) || null; }
  function domainOf(ph) {
    const c = ph && connById(ph.connector);
    return (c && c.domain) || (ph && ph.domain) || 'other';
  }
  async function loadConfig() {
    const cfg = await apiJSON('api/config');
    state.config = cfg;
    if (!state.settings) state.settings = store.get(KEY.settings, null) || { ...defaultsFromConfig() };
    if (state.settings.thinkingPolicy !== 2) {               // thinking is off by default now: reset once
      state.settings.thinking = false; state.settings.thinkingPolicy = 2; store.set(KEY.settings, state.settings);
    }
    renderHeader();
    renderConnectors();
    syncSettingsUI();
    renderAll();
    document.documentElement.dataset.ready = '1';
    return cfg;
  }
  function defaultsFromConfig() {
    const d = (state.config && state.config.defaults) || {};
    return {
      thinking: d.thinking !== undefined ? !!d.thinking : false,
      temperature: d.temperature ?? 0.6,
      max_tokens: d.max_tokens ?? 2048,
      system: d.system ?? '',
    };
  }
  function maxTokensCap() {
    const d = (state.config && state.config.defaults) || {};
    return d.max_tokens_cap || 16384;
  }
  async function refreshConfigQuiet() {
    try {
      const cfg = await apiJSON('api/config');
      state.config = cfg; renderConnectors(); renderHeader();
    } catch (e) { /* ignore */ }
  }
  // --- start-up progress: the UI is usable while the model loads; only Send waits for it ---------------------
  function chatOpen() {
    const h = state.health;
    if (!h) return true;                                      // older servers / not polled yet: let the server decide
    return h.startup ? !!h.startup.chat_open : !!h.ready;
  }
  function fmtDur(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    const m = Math.floor(sec / 60), s = sec % 60;
    return m ? `${m}:${String(s).padStart(2, '0')} min` : `${s} s`;
  }
  // the mind-meld animation (mindmeld.js) beside the phase list: a persistent <pre>, only the text part is rewritten
  function startupAnim(on) {
    if (!on) { if (state.mindmeld) { state.mindmeld.stop(); state.mindmeld = null; } return null; }
    if (!state.mindmeld && window.Mindmeld) state.mindmeld = window.Mindmeld.mount($('#mindmeld'));
    return state.mindmeld;
  }
  function startupProgress(phases) {         // elapsed fraction of the summed expected phase times
    let total = 0, done = 0;
    for (const p of phases) {
      const e = Math.max(1, p.expected_s || 0);
      total += e;
      if (p.status === 'done' || p.status === 'skipped' || p.status === 'failed') done += e;
      else if (p.status === 'running') done += Math.min(0.97 * e, p.elapsed_s || 0);
    }
    return total ? done / total : 0;
  }
  function renderStartup() {
    const box = $('#startup');
    const h = state.health;
    const st = h && h.startup;
    const phases = (st && st.phases) || [];
    if (!phases.length || (h.ready && !st.failed && !state.showReadyFlash)) { box.hidden = true; startupAnim(false); return; }
    box.hidden = false;
    box.className = 'startup' + (st.failed ? ' failed' : h.ready ? ' done' : '');
    const anim = startupAnim(!st.failed);
    if (anim) anim.setProgress(h.ready ? 1 : startupProgress(phases));
    const icon = { done: '✓', running: '', pending: '', failed: '✕', skipped: '–' };
    const rows = phases.map((p) => {
      let right = '';
      if (p.status === 'running') right = `${fmtDur(p.elapsed_s)} <span class="muted">of ~${fmtDur(p.expected_s)}</span>`;
      else if (p.status === 'done' || p.status === 'failed') right = fmtDur(p.elapsed_s);
      else right = `<span class="muted">~${fmtDur(p.expected_s)}</span>`;
      const pct = p.status === 'running' ? Math.min(95, 100 * (p.elapsed_s || 0) / Math.max(1, p.expected_s)) : (p.status === 'done' ? 100 : 0);
      return `<li class="ph ${esc(p.status)}">
        <span class="ph-icon" aria-hidden="true">${icon[p.status] || ''}</span>
        <span class="ph-label">${esc(p.label)}${p.detail ? `<span class="ph-detail">${esc(p.detail)}</span>` : ''}</span>
        <span class="ph-time">${right}</span>
        <span class="ph-bar" aria-hidden="true"><span style="width:${pct.toFixed(1)}%"></span></span>
      </li>`;
    }).join('');
    const encs = Object.entries(h.encoders || {});
    const warm = encs.filter(([, v]) => v === 'warm').length;
    const encLine = encs.length ? `<span class="enc">${warm}/${encs.length} encoders ready</span>` : '';
    let head;
    if (st.failed) head = `<strong>Start-up failed.</strong> ${esc(st.failed)} Sending is disabled for this run.`;
    else if (h.ready) head = '<strong>Ready.</strong> Everything is loaded; Send is open.';
    else head = `<strong>${esc(state.config?.model || 'The model')} is loading</strong> (${fmtDur(st.uptime_s)} so far). You can write messages and add placeholders now; Send opens when it is up.`;
    $('#startup-text').innerHTML = `<div class="startup-head"><span>${head}</span>${encLine}</div><ol class="phases">${rows}</ol>`;
  }
  async function checkHealth() {
    const el = $('#health');
    try {
      const r = await fetch('api/health', { credentials: 'same-origin' });
      const h = await r.json();
      const wasReady = state.health && state.health.ready;
      if (state.config && h.receiver && state.config.receiver && h.receiver !== state.config.receiver) {
        state.switching = false;                             // a new model answers: its connectors, its label
        loadConfig().catch(() => {}); loadModels();
      }
      state.health = h;
      const ready = !!h.ready, failed = !!(h.startup && h.startup.failed);
      el.className = 'health ' + (failed ? 'down' : ready ? 'ready' : 'loading');
      el.textContent = failed ? 'start-up failed' : ready ? 'ready' : (chatOpen() ? 'loading encoders' : 'loading');
      el.title = failed ? h.startup.failed : ready ? 'Everything is loaded' : ((h.startup && h.startup.current) || 'starting');
      if (ready && !wasReady && state.healthPolled) {
        state.showReadyFlash = true; setTimeout(() => { state.showReadyFlash = false; renderStartup(); }, 6000);
        renderStartup(); if (state.mindmeld) state.mindmeld.finale();   // one big collapse + burst: the model is up
      }
      state.healthPolled = true;
      renderStartup();
      updateComposer();
      clearTimeout(state.healthTimer);
      if (!ready || failed) state.healthTimer = setTimeout(checkHealth, 3000);
      if (ready && !wasReady) refreshConfigQuiet();           // encoder status dots in the connector list
    } catch (e) {
      clearTimeout(state.healthTimer);
      if (state.switching) {                                  // between the old server and the new one
        el.className = 'health loading'; el.textContent = `switching to ${state.switching}`;
        state.healthTimer = setTimeout(checkHealth, 3000);
      } else {
        el.className = 'health down'; el.textContent = 'server unreachable';
        state.healthTimer = setTimeout(checkHealth, 10000);
      }
    }
  }
  // ------------------------------------------------------------------ model picker (the front door's /door/api/)
  // The front door answers door/api/models and door/api/switch itself (a server reached without the door: 404, no
  // picker). Only the workstation's own tunnel (a loopback client) may switch the local Gemma model; Kimi is listed
  // as information. A switch restarts the server: the page then shows the start-up progress of the new model.
  async function loadModels() {
    let m = null;
    try {
      const r = await fetch('door/api/models', { credentials: 'same-origin' });
      if (r.ok) m = await r.json();
    } catch (e) { /* no door */ }
    state.models = m;
    const s = m && m.switch;
    if (s && !s.running && s.ok === false && s.message && state.lastSwitchError !== s.message) {
      state.lastSwitchError = s.message; state.switching = false;
      toast(`Model switch failed: ${s.message}`, 'error');
    }
    renderPicker();
    clearTimeout(state.modelsTimer);
    if (s && s.running) state.modelsTimer = setTimeout(loadModels, 3000);
    return m;
  }
  function renderPicker() {
    const sel = $('#model-picker');
    const m = state.models;
    if (!m || !Array.isArray(m.models)) { sel.hidden = true; return; }
    sel.hidden = false;
    const live = m.live || (state.config && state.config.receiver) || null;
    const busy = !!state.streaming;
    const key = JSON.stringify([m.models, live, m.can_switch, m.reason, busy]);
    if (key === state.pickerKey) return;                      // unchanged: never rebuild an open dropdown
    state.pickerKey = key;
    const opts = m.models.map((x) => `<option value="${esc(x.id)}"${x.receiver === live ? ' selected' : ''}${x.switchable ? '' : ' disabled'}>${esc(x.label)}${x.receiver === live ? ' (live)' : ''}</option>`);
    if (!m.models.some((x) => x.receiver === live)) opts.push('<option value="" selected disabled>No model live</option>');
    sel.innerHTML = opts.join('');
    sel.disabled = !m.can_switch || busy;
    sel.title = m.reason || (busy ? 'Wait for the reply to finish' : 'Switch the local model (it reloads: about 3 to 6 minutes)');
  }
  async function switchModel() {
    const sel = $('#model-picker');
    const model = sel.value;
    const row = (state.models && state.models.models.find((x) => x.id === model)) || { label: model };
    state.pickerKey = null;                                   // any answer below re-renders the picker
    if (!window.confirm(`Switch to ${row.label}? The current model stops and ${row.label} loads (about 3 to 6 minutes). Your conversations stay.`)) { state.pickerKey = null; renderPicker(); return; }
    sel.disabled = true;
    try {
      const r = await fetch('door/api/switch', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-AIMM-Chat': 'switch' }, body: JSON.stringify({ model }),
      });
      let data = null;
      try { data = await r.json(); } catch (e) { /* non-JSON */ }
      if (r.status === 401) { showLogin(); return; }
      if (!r.ok) { toast(errText(data, r.status), 'error'); await loadModels(); return; }
      toast((data && data.message) || `Switching to ${row.label}`);
      state.switching = row.label; state.lastSwitchError = null;
      const el = $('#health'); el.className = 'health loading'; el.textContent = `switching to ${row.label}`;
      await loadModels();
      clearTimeout(state.healthTimer); state.healthTimer = setTimeout(checkHealth, 3000);
    } catch (e) { toast(`Switch failed: ${e.message}`, 'error'); state.pickerKey = null; renderPicker(); }
  }
  function renderHeader() {
    const cfg = state.config || {};
    const model = cfg.model || 'the model';
    $$('.model-name-slot').forEach((e) => { e.textContent = model; });
    document.title = `AI Mindmeld · ${model}`;
  }
  function renderConnectors() {
    const host = $('#conn-list');
    const list = connectors();
    if (!list.length) { host.innerHTML = '<p class="ph-empty">No connectors reported by the server.</p>'; return; }
    const groups = groupConnectors(list);
    host.innerHTML = groups.map(([dom, cs]) => `
      <div class="conn-group">
        <div class="conn-group-h dom-${esc(dom)}">${esc(domainTitle(dom))}</div>
        ${cs.map((c) => `
          <div class="conn" title="${esc(`${c.label} · encoder ${c.encoder || '?'} · inputs: ${(c.input_types || []).join(', ')} · status ${c.status || '?'}${c.error ? ` · ${c.error}` : ''}`)}">
            <span class="sdot ${esc(c.status || '')}" aria-label="${esc(c.status || 'unknown')}"></span>
            <span class="conn-name">${esc(c.label || c.id)}</span>
            <span class="conn-k">${c.k && c.k > 1 ? `k=${esc(c.k)}` : ''}</span>
          </div>`).join('')}
      </div>`).join('') + `
      <div class="conn-legend"><span><i class="sdot"></i>cold</span><span><i class="sdot starting"></i>starting</span><span><i class="sdot warm"></i>warm</span><span><i class="sdot error"></i>error</span></div>`;
  }
  function groupConnectors(list) {
    const order = ['protein', 'molecule'];
    const map = new Map();
    for (const c of list) { const d = c.domain || 'other'; if (!map.has(d)) map.set(d, []); map.get(d).push(c); }
    return Array.from(map.entries()).sort((a, b) => (order.indexOf(a[0]) + 99 * (order.indexOf(a[0]) < 0)) - (order.indexOf(b[0]) + 99 * (order.indexOf(b[0]) < 0)));
  }
  function domainTitle(d) { return d === 'protein' ? 'Protein' : d === 'molecule' ? 'Small molecule' : d.charAt(0).toUpperCase() + d.slice(1); }

  // ------------------------------------------------------------------ conversations
  function newConv() {
    return { id: uid(), title: '', created: now(), updated: now(), messages: [], placeholders: {}, nextPid: 1 };
  }
  function loadConvs() {
    const list = store.get(KEY.convs, []);
    state.convs = Array.isArray(list) ? list.filter((c) => c && c.id && Array.isArray(c.messages)) : [];
    // a reply that was streaming when the page closed can't resume
    for (const c of state.convs) for (const m of c.messages) if (m.status === 'streaming') { m.status = 'stopped'; m.note = 'Interrupted (page closed during generation).'; }
    const curId = store.get(KEY.current, null);
    state.cur = state.convs.find((c) => c.id === curId) || null;
    if (!state.cur) { state.cur = newConv(); }
  }
  function saveNow() {
    clearTimeout(state.saveTimer); state.saveTimer = null;
    const keep = state.convs.filter((c) => c.messages.length || Object.keys(c.placeholders).length);
    const ok = store.set(KEY.convs, keep) && store.set(KEY.current, state.cur.id);
    if (!ok && !state.storageWarned) { state.storageWarned = true; toast('Could not save to browser storage (full or blocked). Export the conversation to keep it.', 'error'); }
  }
  function saveSoon() { if (!state.saveTimer) state.saveTimer = setTimeout(saveNow, 900); }
  function touchConv() {
    const c = state.cur;
    c.updated = now();
    if (!state.convs.includes(c)) state.convs.unshift(c);
    if (!c.title) { const first = c.messages.find((m) => m.role === 'user'); if (first) c.title = titleFrom(first.content); }
  }
  function titleFrom(text) {
    const t = text.replace(MARKER_RE, (_, id) => { const ph = state.cur.placeholders[id]; return ph && ph.label ? ph.label : id; }).replace(/\s+/g, ' ').trim();
    return t.length > 56 ? t.slice(0, 54) + '…' : t || 'Untitled';
  }
  function switchConv(id) {
    if (state.streaming) { toast('Stop the current reply first.'); return; }
    const c = state.convs.find((x) => x.id === id); if (!c) return;
    state.cur = c; state.editing = false; $('#input').value = '';
    store.set(KEY.current, c.id);
    renderAll(); closeDrawers(); updateComposer();
  }
  function startNewChat() {
    if (state.streaming) { toast('Stop the current reply first.'); return; }
    if (!state.cur.messages.length && !Object.keys(state.cur.placeholders).length) { closeDrawers(); $('#input').focus(); return; }
    state.cur = newConv(); state.editing = false; $('#input').value = '';
    renderAll(); closeDrawers(); updateComposer(); $('#input').focus();
  }
  function deleteConv(id) {
    const c = state.convs.find((x) => x.id === id); if (!c) return;
    if (!confirm(`Delete “${c.title || 'Untitled'}”? This only removes it from this browser.`)) return;
    state.convs = state.convs.filter((x) => x.id !== id);
    if (state.cur.id === id) { state.cur = state.convs[0] || newConv(); state.editing = false; }
    saveNow(); renderAll(); updateComposer();
  }
  function renderConvList() {
    const host = $('#conv-list');
    const list = [...state.convs].sort((a, b) => b.updated - a.updated);
    if (!list.length) { host.innerHTML = '<li class="side-empty">No saved conversations yet.</li>'; return; }
    host.innerHTML = list.map((c) => {
      const turns = c.messages.filter((m) => m.role === 'user').length;
      const nph = Object.keys(c.placeholders).length;
      return `<li class="conv-item ${c.id === state.cur.id ? 'active' : ''}" data-id="${esc(c.id)}">
        <button type="button" class="conv-open" ${c.id === state.cur.id ? 'aria-current="true"' : ''}>
          <span class="conv-title">${esc(c.title || 'Untitled')}</span>
          <span class="conv-meta">${esc(relTime(c.updated))} · ${turns} turn${turns === 1 ? '' : 's'}${nph ? ` · ${nph} ph` : ''}</span>
        </button>
        <button type="button" class="conv-del" aria-label="Delete conversation ${esc(c.title || 'Untitled')}"><svg><use href="#i-close"/></svg></button>
      </li>`;
    }).join('');
  }
  function relTime(t) {
    const s = (now() - t) / 1000;
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    const d = new Date(t);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }

  // ------------------------------------------------------------------ placeholders
  function usedIds(conv = state.cur) {
    const used = new Map();
    for (const m of conv.messages) if (m.role === 'user') for (const mm of m.content.matchAll(MARKER_RE)) used.set(mm[1], (used.get(mm[1]) || 0) + 1);
    return used;
  }
  function nextId() {
    const c = state.cur;
    let n = c.nextPid || 1;
    while (c.placeholders['P' + n]) n++;
    return 'P' + n;
  }
  function preview(ph, n = 10) {
    const v = (ph.value || '').replace(/\s+/g, '');
    if (ph.input_type === 'uniprot' || ph.input_type === 'chebi') return v;
    return v.length > n ? v.slice(0, n) + '…' : v;
  }
  function infoText(info, ph) {
    if (!info) return '';
    const parts = [];
    if (info.length != null) parts.push(`${info.length} ${ph && ph.input_type === 'smiles' ? 'chars' : 'aa'}`);
    if (info.atoms != null) parts.push(`${info.atoms} atoms`);
    if (info.tokens != null) parts.push(`${info.tokens} tokens`);
    if (info.source) parts.push(info.source === 'cache' ? `cached embedding${info.set ? ` (${info.set})` : ''}` : info.source === 'live' ? 'live encoder' : info.source);
    return parts.join(' · ');
  }
  function chipHTML(id, ph) {
    if (!ph) return `<span class="chip unknown" title="Undefined placeholder ${esc(id)}">[[${esc(id)}]]</span>`;
    const c = connById(ph.connector);
    const dom = domainOf(ph);
    const title = `${id}${ph.label ? ' · ' + ph.label : ''} — ${(c && c.label) || ph.connector} · ${INPUT_TYPE_LABEL[ph.input_type] || ph.input_type}. Click for details.`;
    return `<button type="button" class="chip dom-${esc(dom)}" data-ph="${esc(id)}" title="${esc(title)}"><span class="chip-id">${esc(id)}</span>${ph.label ? `<span class="chip-label">${esc(ph.label)}</span>` : ''}<span class="chip-prev">${esc(preview(ph))}</span></button>`;
  }
  function renderPhList() {
    const host = $('#ph-list');
    const phs = state.cur.placeholders;
    const ids = Object.keys(phs).sort((a, b) => idNum(a) - idNum(b));
    $('#ph-count').textContent = ids.length;
    if (!ids.length) { host.innerHTML = '<li class="ph-empty">None yet. Add one to refer to a protein or a molecule.</li>'; return; }
    const used = usedIds();
    host.innerHTML = ids.map((id) => {
      const ph = phs[id]; const c = connById(ph.connector); const dom = domainOf(ph);
      const u = used.get(id) || 0;
      return `<li class="ph-card dom-${esc(dom)}" data-ph-card="${esc(id)}" tabindex="0" role="button" aria-label="Insert ${esc(id)} at the cursor">
        <div class="ph-head"><span class="ph-id">${esc(id)}</span><span class="ph-label">${esc(ph.label || '')}</span>${u ? `<span class="ph-tag used" title="Referenced in ${u} sent message(s)">sent ×${u}</span>` : '<span class="ph-tag">draft</span>'}</div>
        <div class="ph-sub">${esc((c && c.label) || ph.connector)} · ${esc(INPUT_TYPE_LABEL[ph.input_type] || ph.input_type)}${ph.info ? ' · ' + esc(infoText(ph.info, ph)) : ''}</div>
        <div class="ph-prev">${esc(preview(ph, 40))}</div>
        <div class="ph-actions">
          <button type="button" class="act" data-ph-act="insert" data-id="${esc(id)}">Insert</button>
          <button type="button" class="act" data-ph-act="details" data-id="${esc(id)}">Details</button>
          <button type="button" class="act" data-ph-act="edit" data-id="${esc(id)}">${u ? 'Edit as new' : 'Edit'}</button>
          ${u ? '' : `<button type="button" class="act danger" data-ph-act="delete" data-id="${esc(id)}">Delete</button>`}
        </div>
      </li>`;
    }).join('');
  }
  const idNum = (id) => { const m = /(\d+)$/.exec(id); return m ? +m[1] : 1e9; };

  function insertMarker(id) {
    const ta = $('#input');
    const marker = `[[${id}]]`;
    const v = ta.value;
    let s = ta.selectionStart ?? v.length, e = ta.selectionEnd ?? v.length;
    if (document.activeElement !== ta && state._lastSel) { s = state._lastSel[0]; e = state._lastSel[1]; }
    s = clamp(s, 0, v.length); e = clamp(e, s, v.length);
    const before = v.slice(0, s), after = v.slice(e);
    const pre = before && !/\s$/.test(before) ? ' ' : '';
    const post = !after || !/^[\s.,;:!?)]/.test(after) ? ' ' : '';
    ta.value = before + pre + marker + post + after;
    const pos = (before + pre + marker + post).length;
    ta.focus(); ta.setSelectionRange(pos, pos);
    state._lastSel = [pos, pos];
    updateComposer(); closeDrawers();
  }

  // ----- validation (light; the server has the final word)
  function normalizeValue(type, raw) {
    let v = String(raw || '');
    if (type === 'sequence') {
      v = v.split(/\r?\n/).filter((l) => !/^\s*>/.test(l)).join('');
      return v.replace(/\s+/g, '').toUpperCase();
    }
    if (type === 'uniprot') return v.trim().toUpperCase();
    if (type === 'chebi') { v = v.trim(); const m = /^(?:chebi:?\s*)?(\d+)$/i.exec(v); return m ? `CHEBI:${m[1]}` : v; }
    return v.trim();
  }
  function validate(type, raw) {
    const v = normalizeValue(type, raw);
    if (!v) return { level: 'empty', msg: '', v, len: '' };
    if (type === 'sequence') {
      const bad = Array.from(new Set(v.replace(/[A-Z*]/g, '')));
      if (bad.length) return { level: 'error', msg: `Not amino-acid letters: ${bad.slice(0, 8).join(' ')}`, v, len: `${v.length} aa` };
      const odd = Array.from(new Set(v.replace(/[ACDEFGHIKLMNPQRSTVWY]/g, '')));
      if (odd.length) return { level: 'warn', msg: `Non-standard residues: ${odd.join(' ')}`, v, len: `${v.length} aa` };
      return { level: 'ok', msg: 'Amino-acid sequence', v, len: `${v.length} aa` };
    }
    if (type === 'uniprot') {
      const ok = /^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$/.test(v);
      return ok ? { level: 'ok', msg: 'Looks like a UniProt accession', v, len: '' } : { level: 'warn', msg: 'Does not look like a UniProt accession (e.g. P69905)', v, len: '' };
    }
    if (type === 'smiles') {
      if (/\s/.test(v)) return { level: 'error', msg: 'SMILES cannot contain spaces', v, len: `${v.length} chars` };
      const bad = Array.from(new Set(v.replace(/[A-Za-z0-9@+\-[\]()=#$%/\\.:~*&!]/g, '')));
      if (bad.length) return { level: 'error', msg: `Unexpected characters: ${bad.join(' ')}`, v, len: `${v.length} chars` };
      let depthP = 0, depthB = 0, unbalanced = false;
      for (const ch of v) {
        if (ch === '(') depthP++; else if (ch === ')') depthP--;
        if (ch === '[') depthB++; else if (ch === ']') depthB--;
        if (depthP < 0 || depthB < 0) unbalanced = true;
      }
      if (unbalanced || depthP || depthB) return { level: 'warn', msg: 'Unbalanced brackets or parentheses', v, len: `${v.length} chars` };
      return { level: 'ok', msg: 'SMILES characters OK', v, len: `${v.length} chars` };
    }
    if (type === 'chebi') {
      return /^CHEBI:\d+$/.test(v) ? { level: 'ok', msg: 'ChEBI id', v, len: '' } : { level: 'error', msg: 'Expected CHEBI:<number>, e.g. CHEBI:15365', v, len: '' };
    }
    return { level: 'ok', msg: '', v, len: `${v.length} chars` };
  }

  // ----- add / edit dialog
  function openPhDialog({ editId = null, domain = null } = {}) {
    if (!state.config) { toast('Server configuration not loaded yet.', 'error'); return; }
    if (!connectors().length) { toast('The server reports no connectors.', 'error'); return; }
    const dlg = $('#dlg-ph');
    const src = editId ? state.cur.placeholders[editId] : null;
    const used = editId ? (usedIds().get(editId) || 0) : 0;
    const targetId = editId && !used ? editId : nextId();
    state.phDialog = { editId, targetId, controller: null, used };
    $('#ph-dlg-verb').textContent = editId ? (used ? `Edit ${editId} as new placeholder` : 'Edit placeholder') : 'New placeholder';
    $('#ph-dlg-id').textContent = targetId;
    const note = $('#ph-edit-note');
    note.hidden = !(editId && used);
    note.textContent = editId && used ? `${editId} is already used in sent messages, so it stays unchanged. Saving creates ${targetId} and inserts it at the cursor.` : '';

    // connector select, grouped by domain
    const sel = $('#ph-connector');
    sel.innerHTML = groupConnectors(connectors()).map(([dom, cs]) => `<optgroup label="${esc(domainTitle(dom))}">${cs.map((c) => `<option value="${esc(c.id)}">${esc(c.label || c.id)}${c.k > 1 ? ` (k=${esc(c.k)})` : ''}${c.status === 'warm' ? ' · warm' : c.status === 'error' ? ' · error' : ''}</option>`).join('')}</optgroup>`).join('');
    const last = store.get(KEY.lastConnector, null);
    let pick = src ? src.connector : null;
    if (!pick && domain) pick = (connectors().find((c) => c.id === last && c.domain === domain) || connectors().find((c) => c.domain === domain) || {}).id;
    if (!pick) pick = connById(last) ? last : connectors()[0].id;
    sel.value = pick;
    renderTypes(src ? src.input_type : null);
    $('#ph-value').value = src ? src.value : '';
    $('#ph-label').value = src ? (src.label || '') : '';
    $('#ph-result').hidden = true;
    $('#ph-save').disabled = false;
    $('#ph-save').textContent = 'Check & insert';
    onPhValueInput();
    if (!dlg.open) dlg.showModal();
    setTimeout(() => (src ? $('#ph-value') : sel).focus(), 0);
  }
  function currentType() { const r = $('input[name="ph-type"]:checked'); return r ? r.value : null; }
  function renderTypes(prefer) {
    const c = connById($('#ph-connector').value);
    const types = (c && c.input_types && c.input_types.length) ? c.input_types : ['sequence'];
    const prev = prefer || currentType();
    const chosen = types.includes(prev) ? prev : types[0];
    $('#ph-types').innerHTML = types.map((t) => `<label><input type="radio" name="ph-type" value="${esc(t)}" ${t === chosen ? 'checked' : ''}><span>${esc(INPUT_TYPE_LABEL[t] || t)}</span></label>`).join('');
    const dom = (c && c.domain) || 'other';
    $('#ph-dlg-id').className = `ph-id-badge dom-${dom}`;
    $('#ph-conn-note').textContent = c ? `${domainTitle(dom)} connector · encoder ${c.encoder || '?'}${c.k ? ` · ${c.k} slot token${c.k > 1 ? 's' : ''}` : ''} · ${statusWord(c.status)}${c.error ? ` · ${c.error}` : ''}` : '';
    updateValueLabel();
  }
  function statusWord(s) { return { cold: 'encoder cold (first use starts it)', starting: 'encoder starting', warm: 'encoder warm', error: 'encoder error' }[s] || (s ? `status ${s}` : 'status unknown'); }
  function updateValueLabel() {
    const t = currentType();
    $('#ph-value-label').textContent = INPUT_TYPE_LABEL[t] || 'Value';
    const ta = $('#ph-value');
    ta.rows = t === 'sequence' ? 6 : t === 'smiles' ? 3 : 1;
    ta.placeholder = { sequence: 'Paste an amino-acid sequence (FASTA header lines are ignored)', uniprot: 'e.g. P69905', smiles: 'e.g. CC(=O)Oc1ccccc1C(=O)O', chebi: 'e.g. CHEBI:15365' }[t] || '';
    $('#ph-example').hidden = !EXAMPLES[t];
  }
  function onPhValueInput() {
    const res = validate(currentType(), $('#ph-value').value);
    const el = $('#ph-valid');
    el.className = 'valid ' + (res.level === 'empty' ? '' : res.level);
    el.textContent = res.msg;
    $('#ph-len').textContent = res.len;
    $('#ph-result').hidden = true;
    return res;
  }
  async function savePlaceholder(ev) {
    ev.preventDefault();
    const d = state.phDialog;
    if (!d) { setCheck('error', 'This form lost its state; close it and try again.'); return; }
    const connector = $('#ph-connector').value;
    const input_type = currentType();
    const res = onPhValueInput();
    if (res.level === 'empty') { setCheck('error', 'Enter a value first.'); $('#ph-value').focus(); return; }
    if (res.level === 'error') { setCheck('error', res.msg); $('#ph-value').focus(); return; }
    const value = res.v;
    const label = $('#ph-label').value.trim();
    const conn = connById(connector);
    setCheck('pending', conn && conn.status !== 'warm' ? `Checking… the ${conn.label} encoder may need to start first (this can take a while).` : 'Checking and embedding…');
    $('#ph-save').disabled = true;
    d.controller = new AbortController();
    let out;
    try {
      out = await apiJSON('api/placeholder/check', { method: 'POST', body: { connector, input_type, value }, signal: d.controller.signal });
    } catch (e) {
      $('#ph-save').disabled = false;
      if (e.name === 'AbortError') return;
      if (e instanceof AuthError) { $('#dlg-ph').close(); return; }
      setCheck('error', `Check failed: ${e.message}`);
      return;
    } finally { d.controller = null; }
    $('#ph-save').disabled = false;
    if (!out || !out.ok) { setCheck('error', (out && out.error) || 'The server rejected this input.'); refreshConfigQuiet(); return; }
    const id = d.targetId;
    const ph = { connector, input_type, value, label, key: out.key || null, info: out.info || null, created: now() };
    if (conn) ph.domain = conn.domain;
    state.cur.placeholders[id] = ph;
    if (!d.editId || d.used) state.cur.nextPid = Math.max(state.cur.nextPid || 1, idNum(id) + 1);
    store.set(KEY.lastConnector, connector);
    touchConv(); saveNow();
    $('#dlg-ph').close();
    state.phDialog = null;
    const draft = $('#input').value;
    if (!(d.editId && !d.used && draft.includes(`[[${id}]]`))) insertMarker(id);
    renderPhList(); updateComposer(); refreshConfigQuiet();
    if (d.editId && !d.used) renderThread();
    const info = infoText(ph.info, ph);
    toast(`${id} ready${info ? ' · ' + info : ''}`);
  }
  function setCheck(kind, text) {
    const r = $('#ph-result');
    r.hidden = false; r.className = 'check-result ' + kind; r.textContent = text;
  }
  function deletePlaceholder(id) {
    if (usedIds().get(id)) { toast(`${id} is used in sent messages and cannot be deleted.`); return; }
    if ($('#input').value.includes(`[[${id}]]`) && !confirm(`${id} is referenced in your draft. Delete it anyway?`)) return;
    delete state.cur.placeholders[id];
    touchConv(); saveNow(); renderPhList(); updateComposer();
  }

  // ----- details dialog
  function lastInjection(id) {
    const msgs = state.cur.messages;
    for (let i = msgs.length - 1; i >= 0; i--) { const inj = msgs[i].injections; if (inj && inj[id]) return inj[id]; }
    return null;
  }
  function seqView(ph) {
    const v = ph.value || '';
    if (ph.input_type === 'sequence') {
      const lines = [];
      const width = String(v.length).length;
      for (let i = 0; i < v.length; i += 50) {
        const row = v.slice(i, i + 50).match(/.{1,10}/g).join(' ');
        lines.push(`<span class="pos">${String(i + 1).padStart(width, ' ')}</span>  ${esc(row)}`);
      }
      return `<div class="seqview" tabindex="0" aria-label="Sequence">${lines.join('\n')}</div>`;
    }
    return `<div class="seqview wrap" tabindex="0">${esc(v)}</div>`;
  }
  function showDetails(id) {
    const ph = state.cur.placeholders[id];
    if (!ph) { toast(`${id} is not defined in this conversation.`, 'error'); return; }
    const c = connById(ph.connector); const dom = domainOf(ph);
    const used = usedIds().get(id) || 0;
    const inj = lastInjection(id);
    $('#dlg-detail-title').innerHTML = `<span class="ph-id-badge dom-${esc(dom)}">${esc(id)}</span> ${esc(ph.label || 'Placeholder')}`;
    $('#detail-body').innerHTML = `
      <dl class="kv">
        <dt>Connector</dt><dd>${esc((c && c.label) || ph.connector)} <span class="opt">(${esc(ph.connector)}, ${esc(domainTitle(dom))})</span></dd>
        <dt>Encoder</dt><dd>${esc((c && c.encoder) || '?')} · ${esc(statusWord(c && c.status))}</dd>
        <dt>Input type</dt><dd>${esc(INPUT_TYPE_LABEL[ph.input_type] || ph.input_type)}</dd>
        ${ph.info ? `<dt>Check</dt><dd>${esc(infoText(ph.info, ph))}</dd>` : ''}
        ${ph.key ? `<dt>Cache key</dt><dd class="mono" title="${esc(ph.key)}">${esc(String(ph.key).slice(0, 16))}…</dd>` : ''}
        <dt>Used</dt><dd>${used ? `in ${used} sent message${used > 1 ? 's' : ''} (frozen)` : 'not sent yet'}</dd>
        ${inj ? `<dt>Last injection</dt><dd class="mono">layers ${esc((inj.layers || []).join(', '))}${inj.k ? ` · k=${esc(inj.k)}` : ''}</dd>` : ''}
      </dl>
      ${seqView(ph)}`;
    $('#detail-actions').innerHTML = `
      <button type="button" class="btn ghost" data-d="copy">Copy value</button>
      <button type="button" class="btn ghost" data-d="edit">${used ? 'Edit as new' : 'Edit'}</button>
      <button type="button" class="btn primary" data-d="insert">Insert ${esc(id)}</button>`;
    $('#detail-actions').dataset.id = id;
    const dlg = $('#dlg-detail'); if (!dlg.open) dlg.showModal();
  }

  // ------------------------------------------------------------------ markdown (escape first, then a small block/inline grammar)
  function renderMarkdown(src, phs) {
    const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
    return blocks(lines, phs);
  }
  const LIST_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
  const FENCE_RE = /^\s{0,3}(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;
  const TABLE_SEP_RE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
  function isBlockStart(line, next) {
    return FENCE_RE.test(line) || /^#{1,6}\s/.test(line) || /^\s{0,3}>/.test(line) || LIST_RE.test(line) || /^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line) || (line.includes('|') && next !== undefined && TABLE_SEP_RE.test(next));
  }
  function blocks(lines, phs) {
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      let m;
      if ((m = FENCE_RE.exec(line))) {
        const fence = m[1], lang = m[2]; const buf = []; i++;
        while (i < lines.length && !lines[i].trim().startsWith(fence)) { buf.push(lines[i]); i++; }
        i++;
        out.push(`<pre class="code"><code${lang ? ` data-lang="${esc(lang)}"` : ''}>${esc(buf.join('\n'))}</code></pre>`);
        continue;
      }
      if (/^\s*$/.test(line)) { i++; continue; }
      if ((m = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line))) { const lv = Math.min(6, m[1].length + 2); out.push(`<h${lv}>${inline(m[2], phs)}</h${lv}>`); i++; continue; }
      if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) { out.push('<hr>'); i++; continue; }
      if (/^\s{0,3}>/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s{0,3}>/.test(lines[i])) { buf.push(lines[i].replace(/^\s{0,3}>\s?/, '')); i++; }
        out.push(`<blockquote>${blocks(buf, phs)}</blockquote>`);
        continue;
      }
      if (line.includes('|') && i + 1 < lines.length && TABLE_SEP_RE.test(lines[i + 1])) {
        const cells = (l) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
        const head = cells(line); i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
        out.push(`<table><thead><tr>${head.map((h) => `<th>${inline(h, phs)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${head.map((_, j) => `<td>${inline(r[j] || '', phs)}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
        continue;
      }
      if ((m = LIST_RE.exec(line))) {
        const baseIndent = m[1].length;
        const ordered = /\d/.test(m[2]);
        const start = ordered ? parseInt(m[2], 10) : 1;
        const items = [];
        let loose = false;
        while (i < lines.length) {
          const lm = LIST_RE.exec(lines[i]);
          if (lm && lm[1].length === baseIndent && (/\d/.test(lm[2]) === ordered)) {
            const item = [lm[3]]; i++;
            while (i < lines.length) {
              const l = lines[i];
              if (/^\s*$/.test(l)) {
                const nx = lines[i + 1];
                if (nx !== undefined && (/^\s+/.test(nx) && (nx.match(/^\s*/)[0].length > baseIndent))) { item.push(''); i++; loose = true; continue; }
                break;
              }
              const ind = l.match(/^\s*/)[0].length;
              const lm2 = LIST_RE.exec(l);
              if (lm2 && lm2[1].length <= baseIndent) break;
              if (ind > baseIndent || lm2) { item.push(l.slice(Math.min(ind, baseIndent + 2 + (lm2 ? 0 : 1)))); i++; continue; }
              if (isBlockStart(l, lines[i + 1])) break;
              item.push(l); i++; // lazy continuation
            }
            items.push(item);
            const nx = i < lines.length && /^\s*$/.test(lines[i]) ? LIST_RE.exec(lines[i + 1] || '') : null;
            if (nx && nx[1].length === baseIndent && /\d/.test(nx[2]) === ordered) { loose = true; i++; }
          } else break;
        }
        const lis = items.map((it) => {
          let html = blocks(it, phs);
          if (!loose) html = html.replace(/^<p>([\s\S]*?)<\/p>/, '$1');
          return `<li>${html}</li>`;
        }).join('');
        out.push(ordered ? `<ol${start !== 1 ? ` start="${start}"` : ''}>${lis}</ol>` : `<ul>${lis}</ul>`);
        continue;
      }
      const buf = [];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !(buf.length && isBlockStart(lines[i], lines[i + 1]))) { buf.push(lines[i]); i++; }
      out.push(`<p>${buf.map((l) => inline(l.trim(), phs)).join('<br>')}</p>`);
    }
    return out.join('\n');
  }
  function inline(text, phs) {
    const out = [];
    const re = /(`+)([\s\S]*?[^`]|[^`])\1(?!`)/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      out.push(fmt(text.slice(last, m.index), phs));
      out.push(`<code>${esc(m[2].replace(/^ (.*) $/, '$1'))}</code>`);
      last = re.lastIndex;
    }
    out.push(fmt(text.slice(last), phs));
    return out.join('');
  }
  function fmt(t, phs) {
    if (!t) return '';
    const saved = [];
    const keep = (html) => `\u0000${saved.push(html) - 1}\u0000`;
    let s = t.replace(/\u0000/g, '');
    s = s.replace(MARKER_RE, (all, id) => (phs && phs[id] ? keep(chipHTML(id, phs[id])) : all));
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => keep(`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`));
    s = esc(s);
    s = s.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?=[^\w]|$)/g, '$1<strong>$2</strong>');
    s = s.replace(/(^|[^*\w])\*(?=[^\s*])([^*]*?[^\s*])\*(?!\*)/g, '$1<em>$2</em>');
    s = s.replace(/(^|[^*\w])\*([^\s*])\*(?!\*)/g, '$1<em>$2</em>');
    s = s.replace(/(^|[^\w])_(?=[^\s_])([^_]*?[^\s_]|[^\s_])_(?=[^\w]|$)/g, '$1<em>$2</em>');
    s = s.replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, n) => saved[+n]);
  }
  function userHTML(text, phs) {
    let last = 0, out = '';
    for (const m of text.matchAll(MARKER_RE)) {
      out += esc(text.slice(last, m.index)) + chipHTML(m[1], phs[m[1]]);
      last = m.index + m[0].length;
    }
    return out + esc(text.slice(last));
  }

  // ------------------------------------------------------------------ thread rendering
  function renderAll() {
    renderConvList(); renderPhList(); renderThread();
  }
  function renderThread() {
    const host = $('#entries');
    const msgs = state.cur.messages;
    $('#empty').hidden = msgs.length > 0;
    host.innerHTML = '';
    let turn = 0;
    msgs.forEach((m, i) => {
      if (m.role === 'user') turn++;
      host.appendChild(entryEl(m, i, turn));
    });
    markLast();
    scrollToBottom(true);
  }
  function entryEl(m, idx, turn) {
    const el = document.createElement('article');
    el.className = `entry ${m.role}`;
    el.dataset.idx = idx;
    const modelName = (state.config && state.config.model) || 'the model';
    el.innerHTML = `<div class="gutter"><span class="role">${m.role === 'user' ? 'You' : esc(modelName.split(' ')[0])}</span><span class="turn">§${turn}</span></div><div class="body"></div>`;
    fillEntry(el, m);
    return el;
  }
  function fillEntry(el, m) {
    const body = el.querySelector('.body');
    const phs = state.cur.placeholders;
    if (m.role === 'user') {
      const idx = +el.dataset.idx;
      const msgs = state.cur.messages;
      const unanswered = isFailedReply(msgs[idx + 1]);
      el.classList.toggle('unanswered', unanswered);
      body.innerHTML = `<div class="user-text">${userHTML(m.content, phs)}</div>` +
        (unanswered ? `<div class="msg-note">No reply; this turn is left out of the conversation sent to the model. <button type="button" class="act" data-act="resend" data-idx="${idx}">Retry</button></div>` : '') +
        '<div class="entry-actions"></div>';
      return;
    }
    const live = m.status === 'streaming';
    if (m.readout === 'yes_no') {                            // a score, not a reply: its card
      let yn = live ? stagesHTML(m) : '';
      if (m.yesNo) yn += yesNoHTML(m.yesNo);
      if (m.error) yn += `<div class="msg-error" role="alert">${esc(m.error)}</div>`;
      if (m.note) yn += `<div class="msg-note">${esc(m.note)}</div>`;
      if (!live) yn += metaHTML(m);
      body.innerHTML = yn + '<div class="entry-actions"></div>';
      return;
    }
    const hasReasoning = !!(m.reasoning && m.reasoning.length);
    // the thinking box scrolls on its own and this re-render recreates it: keep its position (follow the bottom while
    // streaming unless the user scrolled up to read), otherwise every update snapped it back to the top
    const oldTb = body.querySelector('.think-body');
    const tbPrev = oldTb ? oldTb.scrollTop : null;
    const tbFollow = !oldTb || oldTb.scrollHeight - oldTb.scrollTop - oldTb.clientHeight < 24;
    let html = '';
    if (live) html += stagesHTML(m);
    if (hasReasoning || (live && m.thinking && !m.content && m.stage === 'generating')) {
      const secs = m.thinkMs ? ` · ${(m.thinkMs / 1000).toFixed(1)} s` : '';
      const label = live && !m.content && m.stage === 'generating' ? 'Thinking…' : 'Thinking';
      html += `<details class="think ${live && !m.content ? 'live' : ''}" ${m._thinkOpen ? 'open' : ''}>
        <summary><span class="think-label">${label}</span><span class="think-meta">${hasReasoning ? `${m.reasoning.length.toLocaleString()} chars${secs}` : ''}</span></summary>
        <div class="think-body md">${hasReasoning ? renderMarkdown(m.reasoning, phs) : ''}</div></details>`;
    }
    const contentHTML = m.content ? renderMarkdown(m.content, phs) : (live ? '' : (m.status === 'error' ? '' : '<p class="content-empty">(no content)</p>'));
    html += `<div class="content md ${live && m.content ? 'streaming' : ''}">${contentHTML}</div>`;
    if (m.error) html += `<div class="msg-error" role="alert">${esc(m.error)}</div>`;
    if (m.note) html += `<div class="msg-note">${esc(m.note)}</div>`;
    if (!live) html += metaHTML(m);
    html += '<div class="entry-actions"></div>';
    body.innerHTML = html;
    const tb = body.querySelector('.think-body');
    if (tb) {
      if (live && tbFollow) tb.scrollTop = tb.scrollHeight;
      else if (tbPrev !== null) tb.scrollTop = tbPrev;
    }
    const det = body.querySelector('details.think');
    if (det) det.addEventListener('toggle', () => { m._thinkOpen = det.open; m._thinkUserToggled = true; });
  }
  // ----- the Yes/No card: the event of a readout 'yes_no' request (null: a log-probability that is not finite)
  const sig = (x, n = 6) => String(Number(x.toPrecision(n)));
  function yesNoNumbers(r) {
    const pY = r.logp_yes == null ? 0 : Math.exp(r.logp_yes), pN = r.logp_no == null ? 0 : Math.exp(r.logp_no);
    let score = r.score == null ? '–' : sig(r.score);
    if (r.score == null && (r.logp_yes == null) !== (r.logp_no == null)) score = r.logp_yes == null ? '−∞' : '+∞';
    return { pY, pN, score, lY: r.logp_yes == null ? '−∞' : sig(r.logp_yes, 4), lN: r.logp_no == null ? '−∞' : sig(r.logp_no, 4) };
  }
  function yesNoHTML(r) {
    const { pY, pN, score, lY, lN } = yesNoNumbers(r);
    const pct = r.pair_normalized == null ? null : 100 * r.pair_normalized;
    const after = r.prefix ? `, next token after "${esc(r.prefix)}"` : ', next token';
    return `<div class="yn-card" role="group" aria-label="Yes/No probability">
      <div class="yn-head"><span class="yn-title">Yes/No probability</span><span class="yn-tag">scored, not generated · thinking off</span></div>
      <div class="yn-main"><span class="yn-big">${pct == null ? '–' : r.pair_normalized.toFixed(3)}</span><span class="yn-big-label">P(Yes) of the two answers, pair-normalized: P(Yes) / (P(Yes) + P(No))</span></div>
      ${pct == null ? '' : `<div class="yn-bar" role="img" aria-label="pair-normalized P(Yes) ${pct.toFixed(1)}%"><span class="y" style="width:${pct.toFixed(2)}%"></span><span class="n" style="width:${(100 - pct).toFixed(2)}%"></span></div>
      <div class="yn-scale"><span>Yes ${pct.toFixed(1)}%</span><span>No ${(100 - pct).toFixed(1)}%</span></div>`}
      <dl class="yn-rows">
        <dt>P(Yes), full softmax</dt><dd>${sig(pY)} <small>(log ${lY})</small></dd>
        <dt>P(No), full softmax</dt><dd>${sig(pN)} <small>(log ${lN})</small></dd>
        <dt>every other token</dt><dd>${sig(Math.max(0, 1 - pY - pN), 3)}</dd>
        <dt>log-odds score</dt><dd>${score} <small>log P(Yes) − log P(No), the suites’ score</small></dd>
      </dl>
      <div class="yn-tokens">read: '${esc(r.yes)}' (id ${esc(r.yes_id)}) · '${esc(r.no)}' (id ${esc(r.no_id)})${after} · appended: "${esc(r.instruction)}"</div>
    </div>`;
  }
  function yesNoText(r) {
    const { pY, pN, score } = yesNoNumbers(r);
    return [
      `P(Yes) = ${sig(pY)}  (full softmax; token '${r.yes}', id ${r.yes_id})`,
      `P(No) = ${sig(pN)}  (full softmax; token '${r.no}', id ${r.no_id})`,
      `log P(Yes) - log P(No) = ${score}  (the suites' score)`,
      `pair-normalized P(Yes) / (P(Yes) + P(No)) = ${r.pair_normalized == null ? '–' : sig(r.pair_normalized)}`,
    ].join('\n');
  }
  function stagesHTML(m) {
    const reached = STAGES.findIndex(([s]) => s === m.stage);
    const pills = STAGES.map(([s, short], i) => {
      let cls = '';
      if (i < reached) cls = (s === 'encoder_start' && !m.sawEncoder) ? 'skipped' : 'done';
      else if (i === reached) cls = 'active';
      return `<span class="stage ${cls}" title="${esc(STAGE_TEXT[s])}">${short}</span>`;
    }).join('');
    const label = m.stage ? (m.stageDetail || STAGE_TEXT[m.stage] || m.stage) : 'Sending…';
    const el = ((now() - (m.started || now())) / 1000).toFixed(1);
    return `<div class="stages" role="status" aria-live="polite">${pills}<span class="stage-detail">${esc(m.stage === 'encoder_start' && m.stageDetail ? 'Revving up encoder: ' + m.stageDetail : label)}<span class="elapsed">${el} s</span></span></div>`;
  }
  function metaHTML(m) {
    const bits = [];
    if (m.status === 'stopped') bits.push('<span>stopped</span>');
    if (m.usage) {
      if (m.usage.prompt_tokens != null) bits.push(`<span title="Prompt tokens (whole conversation is re-prefilled every turn)">${esc(m.usage.prompt_tokens)} prompt</span>`);
      if (m.usage.completion_tokens != null) bits.push(`<span>${esc(m.usage.completion_tokens)} completion</span>`);
    }
    if (m.wallMs) bits.push(`<span title="Wall time measured in the browser">${(m.wallMs / 1000).toFixed(1)} s</span>`);
    if (m.finishReason && m.finishReason !== 'stop') bits.push(`<span class="warn-text" title="finish_reason from the engine">${m.finishReason === 'length' ? 'hit max tokens' : 'finish: ' + esc(m.finishReason)}</span>`);
    if (m.usage && m.usage.completion_tokens && m.genMs && !(m.timings && m.timings.tok_per_s)) bits.push(`<span title="Completion tokens / time since first streamed token">${(m.usage.completion_tokens / (m.genMs / 1000)).toFixed(1)} tok/s</span>`);
    if (m.timings && typeof m.timings === 'object') {
      const t = Object.entries(m.timings).filter(([, v]) => typeof v === 'number').map(([k, v]) => `${k} ${v < 10 ? v.toFixed(2) : v.toFixed(1)}`).join(' · ');
      if (t) bits.push(`<span title="Server timings">${esc(t)}</span>`);
    }
    if (m.thinking === false) bits.push('<span>thinking off</span>');
    if (m.injections && Object.keys(m.injections).length) {
      const phs = state.cur.placeholders;
      bits.push(`<span class="inj" title="Placeholders injected this turn (number of layers written, slot tokens k)"><span>injected</span>${Object.entries(m.injections).map(([id, v]) => `<span class="inj-item">${chipHTML(id, phs[id])}<span>${esc((v.layers || []).length)} layers${v.k > 1 ? ` · k=${esc(v.k)}` : ''}</span></span>`).join('')}</span>`);
    }
    return bits.length ? `<div class="meta">${bits.join('')}</div>` : '';
  }
  function markLast() {
    const entries = $$('#entries .entry');
    entries.forEach((e) => { e.classList.remove('last'); const a = e.querySelector('.entry-actions'); if (a) a.innerHTML = ''; });
    if (state.streaming) return;
    const msgs = state.cur.messages;
    const lastIdx = msgs.length - 1;
    if (lastIdx < 0) return;
    let lastUser = -1;
    for (let i = lastIdx; i >= 0; i--) if (msgs[i].role === 'user') { lastUser = i; break; }
    const elLast = entries[lastIdx];
    if (msgs[lastIdx].role === 'assistant' && elLast) {
      elLast.classList.add('last');
      const scored = msgs[lastIdx].readout === 'yes_no';
      elLast.querySelector('.entry-actions').innerHTML = `<button type="button" class="act" data-act="regen">${scored ? 'Score again' : 'Regenerate'}</button><button type="button" class="act" data-act="copy" data-idx="${lastIdx}">${scored ? 'Copy numbers' : 'Copy'}</button>`;
    } else if (msgs[lastIdx].role === 'user' && elLast) {
      elLast.classList.add('last');
    }
    if (lastUser >= 0 && entries[lastUser]) {
      const a = entries[lastUser].querySelector('.entry-actions');
      a.innerHTML = `<button type="button" class="act" data-act="edit">Edit &amp; resend</button>` + (msgs[lastIdx].role === 'user' ? '<button type="button" class="act" data-act="retry">Send again</button>' : '');
      entries[lastUser].classList.add('last');
    }
  }
  function nearBottom() { const t = $('#thread'); return t.scrollHeight - t.scrollTop - t.clientHeight < 120; }
  function scrollToBottom(force) { const t = $('#thread'); if (force || nearBottom()) t.scrollTop = t.scrollHeight; }

  // ------------------------------------------------------------------ composer
  function updateComposer() {
    if (state.models) renderPicker();                         // no switch while a reply streams
    const ta = $('#input');
    const text = ta.value;
    const phs = state.cur.placeholders;
    // backdrop highlighting (same metrics as the textarea; only backgrounds change)
    let html = '', last = 0;
    for (const m of text.matchAll(MARKER_RE)) {
      const ph = phs[m[1]];
      html += esc(text.slice(last, m.index)) + `<mark class="${ph ? 'dom-' + esc(domainOf(ph)) : 'unknown'}">${esc(m[0])}</mark>`;
      last = m.index + m[0].length;
    }
    html += esc(text.slice(last)) + (text.endsWith('\n') ? ' ' : '');
    $('#input-backdrop').innerHTML = html;
    // autosize
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight + 2, window.innerHeight * 0.38) + 'px';
    $('#input-backdrop').scrollTop = ta.scrollTop;
    // chips row
    const ids = Array.from(new Set(Array.from(text.matchAll(MARKER_RE), (m) => m[1])));
    const row = $('#draft-chips');
    row.hidden = !ids.length;
    row.innerHTML = ids.length ? `<span class="lead">in this message</span>${ids.map((id) => chipHTML(id, phs[id])).join('')}` : '';
    const busy = !!state.streaming;
    $('#btn-send').hidden = busy;
    $('#btn-stop').hidden = !busy;
    $('#btn-send').disabled = !text.trim() || !state.config || !chatOpen();
    $('#btn-send').title = chatOpen() ? `${yesNoOn() ? 'Score' : 'Send'} (Enter) · Shift+Enter for a new line` : `${(state.config && state.config.model) || 'The model'} is still loading; Send opens when it is up (progress under the header)`;
    $('#edit-banner').hidden = !state.editing;
    syncYesNo();
  }
  // The Yes/No switch, shown when the server offers the readout (/api/config yes_no: the answer convention). While it
  // is on, Thinking shows off and is disabled (the readout is defined with thinking off); its setting is kept.
  function yesNoOn() { return !!(state.config && state.config.yes_no && state.settings && state.settings.yesNo); }
  function syncYesNo() {
    const convention = state.config && state.config.yes_no;
    const on = yesNoOn();
    $('#yesno-switch').hidden = !convention;
    $('#yesno-toggle').checked = on;
    $('#yesno-state').textContent = on ? 'on' : 'off';
    const th = $('#thinking-toggle');
    th.disabled = on;
    th.closest('.switch').classList.toggle('disabled', on);
    th.closest('.switch').title = on ? 'Thinking is off while Yes/No probability is on: the readout is defined with thinking off' : 'Reasoning (thinking) on or off';
    th.checked = !on && !!(state.settings && state.settings.thinking);
    $('#thinking-state').textContent = th.checked ? 'on' : 'off';
    $('#send-label').textContent = on ? 'Score' : 'Send';
    $('#input').placeholder = on ? 'Ask a Yes/No question about a placeholder…' : 'Write a message…';
  }
  // The notice shown each time the Yes/No switch is turned on, unless this browser chose "Don't show this again"
  // (KEY.yesNoNoticeOff, written only when that box is ticked and OK pressed; absent or unreadable storage shows it).
  // The switch is already on when the notice opens; no storage or dialog error changes that.
  function showYesNoNotice() {
    const convention = state.config && state.config.yes_no;
    if (!convention || !yesNoOn() || store.get(KEY.yesNoNoticeOff, null) === true) return;
    const dlg = $('#dlg-yesno');
    if (dlg.open) return;
    $('#yesno-note').innerHTML = `Your message is scored on its own (earlier turns, the system prompt and the sampling settings are not sent), with “${esc(convention.instruction)}” appended${convention.prefix ? ` and the answer read after “${esc(convention.prefix)}”` : ''}. Write the question without an answer instruction.`;
    $('#yesno-notice-off').checked = false;
    dlg.returnValue = '';
    try { dlg.showModal(); } catch (e) { return; }
    dlg.addEventListener('close', () => { if (dlg.returnValue === 'ok' && $('#yesno-notice-off').checked) store.set(KEY.yesNoNoticeOff, true); }, { once: true });
  }
  function setThinking(on) {
    if (!state.settings) state.settings = defaultsFromConfig();
    state.settings.thinking = !!on;
    $('#thinking-toggle').checked = !!on;
    $('#thinking-state').textContent = on ? 'on' : 'off';
    store.set(KEY.settings, state.settings);
  }

  // ------------------------------------------------------------------ settings
  function syncSettingsUI() {
    const s = state.settings;
    for (const k of ['temperature']) { $(`#set-${k}`).value = s[k]; $(`#set-${k}-n`).value = s[k]; }
    $('#set-max_tokens').max = maxTokensCap();
    if (s.max_tokens > maxTokensCap()) { s.max_tokens = maxTokensCap(); store.set(KEY.settings, s); }
    $('#set-max_tokens').value = s.max_tokens;
    $('#set-system').value = s.system || '';
    setThinking(s.thinking);
  }
  function bindSettings() {
    for (const k of ['temperature']) {
      const r = $(`#set-${k}`), n = $(`#set-${k}-n`);
      r.addEventListener('input', () => { n.value = r.value; state.settings[k] = +r.value; store.set(KEY.settings, state.settings); });
      n.addEventListener('change', () => { const v = clamp(+n.value || 0, +n.min, +n.max); n.value = v; r.value = v; state.settings[k] = v; store.set(KEY.settings, state.settings); });
    }
    $('#set-max_tokens').addEventListener('change', (e) => { const v = clamp(Math.round(+e.target.value || 1), 1, maxTokensCap()); e.target.value = v; state.settings.max_tokens = v; store.set(KEY.settings, state.settings); });
    $('#set-system').addEventListener('input', (e) => { state.settings.system = e.target.value; store.set(KEY.settings, state.settings); });
    $('#btn-reset-settings').addEventListener('click', () => { state.settings = { ...defaultsFromConfig() }; store.set(KEY.settings, state.settings); syncSettingsUI(); toast('Settings reset to server defaults'); });
  }

  // ------------------------------------------------------------------ chat
  // An assistant turn that failed or was stopped before any content: its user turn is "unanswered". Both are left out
  // of the history sent to the server (the server requires alternating turns), and the user turn is shown grayed with
  // a Retry action. A Yes/No turn and its card are left out of later history too (history.js).
  const isFailedReply = window.AimmHistory.isFailedReply;
  function buildPayload(messages) {
    const s = state.settings;
    const { messages: msgs, readout, thinking } = window.AimmHistory.request(messages, { thinking: s.thinking, yesNo: yesNoOn() });
    const placeholders = {};
    const missing = [];
    for (const m of msgs) if (m.role === 'user') for (const mm of m.content.matchAll(MARKER_RE)) {
      const id = mm[1]; const ph = state.cur.placeholders[id];
      if (!ph) { missing.push(id); continue; }
      placeholders[id] = { connector: ph.connector, input_type: ph.input_type, value: ph.value };
      if (ph.label) placeholders[id].label = ph.label;
    }
    const payload = readout === 'yes_no'                     // the server reads no other setting for a score
      ? { messages: msgs, placeholders, readout }
      : { messages: msgs, placeholders, thinking, system: s.system || '', sampling: { temperature: +s.temperature, max_tokens: Math.min(+s.max_tokens, maxTokensCap()) } };
    return { payload, missing: Array.from(new Set(missing)) };
  }
  function send() {
    if (state.streaming) return;
    if (!chatOpen()) { toast(`${(state.config && state.config.model) || 'The model'} is still loading; your message stays here. Send opens when it is up.`, 'warn'); return; }
    const ta = $('#input');
    const text = ta.value.replace(/\s+$/, '');
    if (!text.trim()) return;
    const unknown = Array.from(new Set(Array.from(text.matchAll(MARKER_RE), (m) => m[1]).filter((id) => !state.cur.placeholders[id])));
    if (unknown.length) { toast(`Undefined placeholder${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}. Add ${unknown.length > 1 ? 'them' : 'it'} or remove the marker.`, 'error'); return; }
    const c = state.cur;
    if (state.editing) {
      let lastUser = -1;
      for (let i = c.messages.length - 1; i >= 0; i--) if (c.messages[i].role === 'user') { lastUser = i; break; }
      if (lastUser >= 0) c.messages.splice(lastUser);
      state.editing = false;
    }
    c.messages.push(yesNoOn() ? { role: 'user', content: text, ts: now(), readout: 'yes_no' } : { role: 'user', content: text, ts: now() });
    ta.value = ''; state._lastSel = null;
    touchConv(); saveNow();
    renderAll(); updateComposer();
    runChat();
  }
  async function runChat() {
    const c = state.cur;
    const { payload, missing } = buildPayload(c.messages);
    if (missing.length) { toast(`Undefined placeholder(s): ${missing.join(', ')}`, 'error'); return; }
    const scoring = payload.readout === 'yes_no';
    const msg = { role: 'assistant', content: '', reasoning: '', status: 'streaming', thinking: scoring ? false : payload.thinking, started: now(), stage: null, ts: now() };
    if (scoring) msg.readout = 'yes_no';
    c.messages.push(msg);
    const controller = new AbortController();
    const st = { controller, requestId: null, msg, conv: c, stopRequested: false, firstTokenAt: 0, thinkStart: 0 };
    state.streaming = st;
    renderThread(); updateComposer();
    const el = () => $(`#entries .entry[data-idx="${c.messages.indexOf(msg)}"]`);
    let raf = 0;
    const paint = () => { raf = 0; const e = el(); if (e && state.cur === c) { fillEntry(e, msg); scrollToBottom(false); } };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(paint); };
    const ticker = setInterval(() => { if (!msg.content) schedule(); }, 250);
    let gotDone = false;

    const handle = (ev) => {
      if (!ev || typeof ev !== 'object') return;
      const rid = ev.request_id ?? ev.requestId ?? ev.id;
      if (rid && !st.requestId && typeof rid !== 'object') st.requestId = String(rid);
      switch (ev.type) {
        case 'status':
          msg.stage = ev.stage || msg.stage; msg.stageDetail = ev.detail || '';
          if (ev.stage === 'encoder_start') { msg.sawEncoder = true; refreshConfigQuiet(); }
          if (ev.stage === 'generating' && msg.thinking && !msg.content) msg._thinkOpen = msg._thinkUserToggled ? msg._thinkOpen : true;
          break;
        case 'reasoning':
          if (!st.thinkStart) st.thinkStart = now();
          if (!st.firstTokenAt) st.firstTokenAt = now();
          msg.stage = 'generating';
          msg.reasoning += ev.delta || '';
          if (!msg._thinkUserToggled) msg._thinkOpen = true;
          break;
        case 'content':
          if (!st.firstTokenAt) st.firstTokenAt = now();
          msg.stage = 'generating';
          if (!msg.content && st.thinkStart) msg.thinkMs = now() - st.thinkStart;
          if (!msg.content && !msg._thinkUserToggled) msg._thinkOpen = false; // collapse once the answer starts
          msg.content += ev.delta || '';
          break;
        case 'done':
          gotDone = true;
          msg.usage = ev.usage || null; msg.timings = ev.timings || null; msg.injections = ev.injections || null;
          msg.finishReason = ev.finish_reason || null;
          break;
        case 'yes_no':
          msg.yesNo = ev;
          break;
        case 'error':
          msg.error = ev.message || 'Server error';
          break;
        default: break;
      }
      schedule();
      saveSoon();
    };

    try {
      const r = await api('api/chat', { method: 'POST', body: payload, signal: controller.signal });
      const ctype = r.headers.get('content-type') || '';
      if (!r.ok || !ctype.includes('text/event-stream')) {
        let data = null; try { data = await r.json(); } catch (e) { /* ignore */ }
        throw new Error(r.ok ? `Unexpected response (${ctype || 'no content type'})` : errText(data, r.status));
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true }).replace(/\r\n?/g, '\n');
        let cut;
        while ((cut = buf.indexOf('\n\n')) >= 0) {
          const raw = buf.slice(0, cut); buf = buf.slice(cut + 2);
          const data = raw.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).replace(/^ /, '')).join('\n');
          if (!data || data === '[DONE]') continue;
          try { handle(JSON.parse(data)); } catch (e) { /* skip malformed event */ }
        }
      }
      if (!gotDone && !msg.error && !st.stopRequested) msg.error = 'The stream ended before the reply was complete.';
      if (gotDone && scoring && !msg.yesNo && !msg.error && !st.stopRequested) msg.error = 'The score did not arrive.';
    } catch (e) {
      if (e.name === 'AbortError' || st.stopRequested) { /* stopped by user */ }
      else if (e instanceof AuthError) msg.error = 'Signed out. Sign in and press Regenerate.';
      else msg.error = `Request failed: ${e.message}`;
    } finally {
      clearInterval(ticker);
      if (raf) cancelAnimationFrame(raf);
      msg.wallMs = now() - msg.started;
      if (st.firstTokenAt) msg.genMs = now() - st.firstTokenAt;
      if (st.thinkStart && !msg.thinkMs) msg.thinkMs = now() - st.thinkStart;
      msg.status = st.stopRequested ? 'stopped' : msg.error ? 'error' : 'done';
      if (!msg._thinkUserToggled) msg._thinkOpen = false;
      delete msg.stage; delete msg.stageDetail; delete msg.sawEncoder;
      state.streaming = null;
      touchConv(); saveNow();
      if (state.cur === c) { renderThread(); }
      renderConvList(); updateComposer();
      refreshConfigQuiet();
    }
  }
  async function stop() {
    const st = state.streaming; if (!st) return;
    st.stopRequested = true;
    $('#btn-stop').disabled = true;
    try {
      if (st.requestId) await apiJSON('api/stop', { method: 'POST', body: { request_id: st.requestId } });
    } catch (e) { /* abort below regardless */ }
    // give the server a moment to close the stream with a final event, then cut it
    setTimeout(() => { if (state.streaming === st) st.controller.abort(); }, st.requestId ? 1500 : 0);
    setTimeout(() => { $('#btn-stop').disabled = false; }, 1600);
  }
  function resendAt(idx) {
    // Retry an unanswered user turn: remove it and its failed reply, then ask it again at the end of the conversation.
    if (state.streaming) return;
    const msgs = state.cur.messages;
    const m = msgs[idx];
    if (!m || m.role !== 'user' || !isFailedReply(msgs[idx + 1])) return;
    msgs.splice(idx, 2);
    if (msgs.length && msgs[msgs.length - 1].role === 'user' && !isFailedReply(msgs[msgs.length])) {
      toast('Answer or remove the last message first.', 'error'); msgs.splice(idx, 0, m, { role: 'assistant', content: '', status: 'error', ts: now() }); return;
    }
    msgs.push(m.readout ? { role: 'user', content: m.content, ts: now(), readout: m.readout } : { role: 'user', content: m.content, ts: now() });
    touchConv(); saveNow(); renderThread(); runChat();
  }
  function regenerate() {
    if (state.streaming) return;
    const msgs = state.cur.messages;
    if (msgs.length && msgs[msgs.length - 1].role === 'assistant') msgs.pop();
    if (!msgs.length || msgs[msgs.length - 1].role !== 'user') return;
    renderThread(); runChat();
  }
  function editLast() {
    if (state.streaming) return;
    const msgs = state.cur.messages;
    let lastUser = -1;
    for (let i = msgs.length - 1; i >= 0; i--) if (msgs[i].role === 'user') { lastUser = i; break; }
    if (lastUser < 0) return;
    state.editing = true;
    const ta = $('#input');
    ta.value = msgs[lastUser].content;
    updateComposer(); ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
  }

  // ------------------------------------------------------------------ export
  function exportConv() {
    const c = state.cur;
    if (!c.messages.length && !Object.keys(c.placeholders).length) { toast('Nothing to export yet.'); return; }
    const data = {
      format: 'aimm-chat/conversation@1',
      exported_at: new Date().toISOString(),
      model: state.config && state.config.model, engine: state.config && state.config.engine,
      title: c.title, created: new Date(c.created).toISOString(), updated: new Date(c.updated).toISOString(),
      settings: { ...state.settings },
      placeholders: c.placeholders,
      messages: c.messages.map((m) => { const o = { ...m }; delete o._thinkOpen; delete o._thinkUserToggled; return o; }),
      next_request: buildPayload(c.messages.filter((m) => m.status !== 'streaming')).payload,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    const slug = (c.title || 'chat').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'chat';
    a.href = URL.createObjectURL(blob);
    a.download = `chat-${new Date().toISOString().slice(0, 10)}-${slug}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ------------------------------------------------------------------ about
  function renderAbout() {
    const REPO = 'https://github.com/PolymathicAI/AI-Mindmeld';
    const link = (href, text) => `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    $('#about-body').innerHTML = `
      <p><strong>AI Mindmeld: A Latent Interconnect Protocol</strong></p>
      <p>${link(REPO, 'Repository')} · ${link(`${REPO}/tree/main/src/latent_interconnect/chat`, 'Code of this chat')}</p>
      <p>Please cite the paper first, then the repository:</p>
      <p>Siavash Golkar, Jake Kovalic, and Shirley Ho. <em>${link('https://polymathic-ai.org/papers/aimm.pdf', 'AI Mindmeld: A Latent Interconnect Protocol.')}</em> 2026. Manuscript.</p>
      <p>Siavash Golkar and the AI Mindmeld contributors. <em>AI Mindmeld</em> ${link(REPO, 'software repository')}.</p>`;
  }

  // ------------------------------------------------------------------ login
  function showLogin() {
    $('#login').hidden = false;
    if (!state.loginAnim && window.Mindmeld) state.loginAnim = window.Mindmeld.mount($('#login-mindmeld'), { fit: true, progress: 0.6 });
    $('#login-err').hidden = true;
    setTimeout(() => $('#login-token').focus(), 0);
  }
  async function doLogin(ev) {
    ev.preventDefault();
    const token = $('#login-token').value;
    const err = $('#login-err');
    err.hidden = true;
    try {
      const r = await fetch('api/login', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token }) });
      if (r.status === 401) { err.textContent = 'That token was not accepted.'; err.hidden = false; return; }
      if (!r.ok) { err.textContent = `Sign-in failed (HTTP ${r.status}).`; err.hidden = false; return; }
      $('#login').hidden = true; $('#login-token').value = '';
      if (state.loginAnim) { state.loginAnim.stop(); state.loginAnim = null; }
      await loadConfig();
      loadModels();
      checkHealth();
    } catch (e) {
      if (!(e instanceof AuthError)) { err.textContent = `Sign-in failed: ${e.message}`; err.hidden = false; }
    }
  }

  // ------------------------------------------------------------------ drawers
  function openDrawer(which) {
    const el = $(which === 'left' ? '#sidebar' : '#panel');
    el.classList.add('open');
    $('#scrim').hidden = false;
    $(which === 'left' ? '#btn-sidebar' : '#btn-panel').setAttribute('aria-expanded', 'true');
  }
  function closeDrawers() {
    $('#sidebar').classList.remove('open'); $('#panel').classList.remove('open');
    $('#scrim').hidden = true;
    $('#btn-sidebar').setAttribute('aria-expanded', 'false'); $('#btn-panel').setAttribute('aria-expanded', 'false');
  }

  // ------------------------------------------------------------------ bindings
  function bind() {
    const ta = $('#input');
    ta.addEventListener('input', updateComposer);
    ta.addEventListener('scroll', () => { $('#input-backdrop').scrollTop = ta.scrollTop; });
    ta.addEventListener('blur', () => { state._lastSel = [ta.selectionStart, ta.selectionEnd]; });
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
        e.preventDefault();
        if (state.streaming) toast('A reply is still streaming. Stop it or wait.');
        else send();
      }
    });
    window.addEventListener('resize', () => updateComposer());
    $('#btn-send').addEventListener('click', send);
    $('#btn-stop').addEventListener('click', stop);
    $('#yesno-toggle').addEventListener('change', (e) => {
      if (!state.settings) state.settings = defaultsFromConfig();
      state.settings.yesNo = e.target.checked;
      store.set(KEY.settings, state.settings);
      updateComposer();
      if (e.target.checked) showYesNoNotice();
    });
    $('#thinking-toggle').addEventListener('change', (e) => {
      if (!e.target.checked) { setThinking(false); return; }
      e.target.checked = false;                              // stays off unless the user confirms
      const dlg = $('#dlg-thinking');
      dlg.returnValue = '';
      dlg.addEventListener('close', () => setThinking(dlg.returnValue === 'proceed'), { once: true });
      dlg.showModal();
    });
    $('#btn-add-ph').addEventListener('click', () => openPhDialog());
    $('#btn-add-ph-2').addEventListener('click', () => openPhDialog());
    $('#btn-cancel-edit').addEventListener('click', () => { state.editing = false; $('#input').value = ''; updateComposer(); });
    $('#btn-new').addEventListener('click', startNewChat);
    $('#btn-export').addEventListener('click', exportConv);
    $('#btn-theme').addEventListener('click', cycleTheme);
    $('#model-picker').addEventListener('change', switchModel);
    $('#btn-settings').addEventListener('click', () => { syncSettingsUI(); $('#dlg-settings').showModal(); });
    $('#btn-about').addEventListener('click', () => { renderAbout(); $('#dlg-about').showModal(); });
    $('#btn-refresh-config').addEventListener('click', async () => { await refreshConfigQuiet(); toast('Connector status refreshed'); });
    $('#btn-sidebar').addEventListener('click', () => ($('#sidebar').classList.contains('open') ? closeDrawers() : openDrawer('left')));
    $('#btn-panel').addEventListener('click', () => ($('#panel').classList.contains('open') ? closeDrawers() : openDrawer('right')));
    $('#scrim').addEventListener('click', closeDrawers);
    $$('[data-drawer-close]').forEach((b) => b.addEventListener('click', closeDrawers));
    $('#login-form').addEventListener('submit', doLogin);

    // placeholder dialog
    $('#ph-form').addEventListener('submit', savePlaceholder);
    $('#ph-connector').addEventListener('change', () => { renderTypes(); onPhValueInput(); });
    $('#ph-types').addEventListener('change', () => { updateValueLabel(); onPhValueInput(); });
    $('#ph-value').addEventListener('input', onPhValueInput);
    $('#ph-example').addEventListener('click', () => {
      const ex = EXAMPLES[currentType()]; if (!ex) return;
      $('#ph-value').value = ex.value; if (!$('#ph-label').value) $('#ph-label').value = ex.label; onPhValueInput();
    });
    $('#dlg-ph').addEventListener('close', () => {
      // 'close' is dispatched asynchronously; if the dialog was reopened meanwhile, leave the new state alone
      if ($('#dlg-ph').open) return;
      const d = state.phDialog; if (d && d.controller) d.controller.abort(); state.phDialog = null;
    });
    $('#ph-value').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $('#ph-form').requestSubmit(); } });

    // generic close buttons; click on backdrop closes
    $$('dialog').forEach((dlg) => {
      dlg.addEventListener('click', (e) => {
        if (e.target.closest('[data-close]')) { dlg.close(); return; }
        if (e.target === dlg) { const r = dlg.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dlg.close(); }
      });
    });
    $('#detail-actions').addEventListener('click', (e) => {
      const b = e.target.closest('[data-d]'); if (!b) return;
      const id = $('#detail-actions').dataset.id;
      if (b.dataset.d === 'insert') { $('#dlg-detail').close(); insertMarker(id); }
      else if (b.dataset.d === 'edit') { $('#dlg-detail').close(); openPhDialog({ editId: id }); }
      else if (b.dataset.d === 'copy') copyText(state.cur.placeholders[id].value);
    });

    // delegated: chips, placeholder cards, entry actions, conversations, starters
    document.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip[data-ph]');
      if (chip) { e.preventDefault(); showDetails(chip.dataset.ph); return; }
      const act = e.target.closest('[data-ph-act]');
      if (act) {
        const id = act.dataset.id;
        if (act.dataset.phAct === 'insert') insertMarker(id);
        else if (act.dataset.phAct === 'details') showDetails(id);
        else if (act.dataset.phAct === 'edit') openPhDialog({ editId: id });
        else if (act.dataset.phAct === 'delete') deletePlaceholder(id);
        return;
      }
      const card = e.target.closest('[data-ph-card]');
      if (card) { insertMarker(card.dataset.phCard); return; }
      const ea = e.target.closest('[data-act]');
      if (ea) {
        if (ea.dataset.act === 'regen' || ea.dataset.act === 'retry') regenerate();
        if (ea.dataset.act === 'resend') resendAt(+ea.dataset.idx);
        else if (ea.dataset.act === 'edit') editLast();
        else if (ea.dataset.act === 'copy') { const m = state.cur.messages[+ea.dataset.idx]; copyText(m.yesNo ? yesNoText(m.yesNo) : m.content); }
        return;
      }
      const conv = e.target.closest('.conv-item');
      if (conv) {
        if (e.target.closest('.conv-del')) deleteConv(conv.dataset.id);
        else switchConv(conv.dataset.id);
        return;
      }
      const st = e.target.closest('.starter');
      if (st) {
        const ta = $('#input');
        const tmpl = st.dataset.text;
        const existing = Object.entries(state.cur.placeholders).find(([, p]) => domainOf(p) === st.dataset.domain);
        if (existing) { ta.value = tmpl.replace('{P}', `[[${existing[0]}]]`); updateComposer(); ta.focus(); return; }
        ta.value = tmpl.replace('{P}', '').replace(/\s+\?/, '?');
        const pos = tmpl.indexOf('{P}');
        state._lastSel = [pos, pos];
        updateComposer();
        openPhDialog({ domain: st.dataset.domain });
      }
    });
    document.addEventListener('keydown', (e) => {
      const card = e.target.closest && e.target.closest('[data-ph-card]');
      if (card && (e.key === 'Enter' || e.key === ' ') && e.target === card) { e.preventDefault(); insertMarker(card.dataset.phCard); }
      if (e.key === 'Escape') closeDrawers();
      if (e.altKey && (e.key === 'p' || e.key === 'P' || e.code === 'KeyP')) { e.preventDefault(); openPhDialog(); }
    });
  }
  async function copyText(t) {
    try { await navigator.clipboard.writeText(t || ''); toast('Copied'); } catch (e) { toast('Copy failed (clipboard not available)', 'error'); }
  }

  // ------------------------------------------------------------------ boot
  async function boot() {
    migrateLegacyStorage();
    applyTheme();
    loadConvs();
    state.settings = store.get(KEY.settings, null);
    bind();
    renderConvList(); renderPhList(); renderThread();
    if (state.settings) syncSettingsUI();
    updateComposer();
    checkHealth();
    try {
      await loadConfig();
      loadModels();
    } catch (e) {
      if (!(e instanceof AuthError)) toast(`Could not load server configuration: ${e.message}`, 'error');
    }
    updateComposer();
    $('#input').focus();
  }
  boot();
})();
