// The turns a chat request sends, from the conversation the browser keeps: the pairing rule is examples/guides/chat.md's
// "Yes/No probabilities" paragraph, the request body its section 4. Loaded by index.html before app.js
// (window.AimmHistory) and by the tests through Node (module.exports). No DOM.
'use strict';
(function (root) {
  // An assistant turn that failed or was stopped before any content: it and its user turn are left out.
  function isFailedReply(m) {
    return !!m && m.role === 'assistant' && !m.content && (m.status === 'error' || m.status === 'stopped');
  }
  // The turns a reply is generated from. Left out: a failed reply and its user turn (the server's templates require
  // alternating turns), and a Yes/No turn (a user turn with readout 'yes_no') together with its result card, which is a
  // score, not a reply; dropping only one of the two would leave two user turns in a row.
  function history(messages) {
    const out = [];
    let scored = false;
    for (const m of messages) {
      if (m.role === 'user') {
        scored = m.readout === 'yes_no';
        if (!scored) out.push({ role: 'user', content: m.content });
      } else if (m.role === 'assistant') {
        if (scored || m.readout === 'yes_no') { scored = false; continue; }
        if (isFailedReply(m)) { if (out.length && out[out.length - 1].role === 'user') out.pop(); continue; }
        const o = { role: 'assistant', content: m.content || '' };
        if (m.reasoning) o.reasoning = m.reasoning;
        out.push(o);
      }
    }
    return out;
  }
  // The turns a request sends for the conversation's last turn: a Yes/No turn alone (it is scored on its own), else
  // the history; the readout it asks for; and, for a generation, the thinking mode the composer shows: off while the
  // Yes/No switch is on (`yesNo`), whatever the stored Thinking setting (`thinking`).
  function request(messages, { thinking = false, yesNo = false } = {}) {
    const last = messages[messages.length - 1];
    if (last && last.role === 'user' && last.readout === 'yes_no') {
      return { messages: [{ role: 'user', content: last.content }], readout: 'yes_no' };
    }
    return { messages: history(messages), readout: 'generate', thinking: !!thinking && !yesNo };
  }
  const api = { isFailedReply, history, request };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AimmHistory = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
