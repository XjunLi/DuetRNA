/* DuetRNA site: no media requests until an explicit click; one active media node. */
(() => {
  'use strict';
  document.documentElement.classList.add('js');
  const $ = (q, scope = document) => scope.querySelector(q);
  const menu = $('.menu'), nav = $('#site-nav');
  if (menu && nav) {
    menu.addEventListener('click', () => {
      const open = menu.getAttribute('aria-expanded') !== 'true';
      menu.setAttribute('aria-expanded', String(open));
      nav.classList.toggle('is-open', open);
    });
    nav.addEventListener('click', e => {
      if (!e.target.closest('a')) return;
      menu.setAttribute('aria-expanded', 'false');
      nav.classList.remove('is-open');
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
        menu.setAttribute('aria-expanded', 'false'); nav.classList.remove('is-open'); menu.focus();
      }
    });
  }

  const progress = $('.progress');
  const sections = [...document.querySelectorAll('main section[id]')];
  const navLinks = nav ? [...nav.querySelectorAll('a[href^="#"]')] : [];
  let scrollFrame = 0;
  function updateReadingPosition() {
    scrollFrame = 0;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.transform = `scaleX(${max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0})`;
    let current = '';
    for (const section of sections) if (section.getBoundingClientRect().top <= window.innerHeight * .3) current = section.id;
    for (const link of navLinks) {
      if (link.hash === '#' + current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }
  function requestReadingUpdate() { if (!scrollFrame) scrollFrame = requestAnimationFrame(updateReadingPosition); }
  window.addEventListener('scroll', requestReadingUpdate, { passive: true });
  window.addEventListener('resize', requestReadingUpdate, { passive: true });
  window.addEventListener('load', requestReadingUpdate, { once: true });
  updateReadingPosition();

  const dialog = document.createElement('dialog');
  dialog.className = 'media-dialog'; dialog.setAttribute('aria-labelledby', 'media-title');
  dialog.innerHTML = '<div class="dialog-head"><h2 id="media-title"></h2><button class="close-dialog" type="button" aria-label="Close media">×</button></div><div class="dialog-body"></div><p class="dialog-note"></p><p class="dialog-message" role="status" aria-live="polite"></p>';
  document.body.append(dialog);
  const title = $('#media-title', dialog), body = $('.dialog-body', dialog);
  const note = $('.dialog-note', dialog), status = $('.dialog-message', dialog);
  let media = null, activeType = '', generation = 0, timer = 0, oldOverflow = '';
  function disposeMedia() {
    generation++; clearTimeout(timer);
    if (media instanceof HTMLVideoElement) { media.pause(); media.removeAttribute('src'); media.load(); }
    if (media instanceof HTMLIFrameElement) media.src = 'about:blank';
    body.replaceChildren(); media = null; activeType = ''; status.textContent = '';
  }
  function closeMedia() {
    if (!dialog.open) return;
    disposeMedia(); document.body.style.overflow = oldOverflow; dialog.close();
  }
  $('.close-dialog', dialog).addEventListener('click', closeMedia);
  dialog.addEventListener('cancel', event => { event.preventDefault(); closeMedia(); });
  // Ignore a queued close event if a new media session has already opened.
  dialog.addEventListener('close', () => { if (!dialog.open) { disposeMedia(); document.body.style.overflow = oldOverflow; } });
  dialog.addEventListener('click', e => {
    if (e.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) closeMedia();
  });
  document.addEventListener('click', e => {
    const link = e.target.closest('a[data-video],a[data-viewer],a[data-image]');
    if (!link || e.defaultPrevented || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey || typeof dialog.showModal !== 'function') return;
    e.preventDefault();
    disposeMedia();
    const token = generation;
    dialog.style.width = link.hasAttribute('data-image') ? 'min(1680px, calc(100vw - 18px))' : '';
    title.textContent = link.dataset.title || 'DuetRNA';
    note.replaceChildren(document.createTextNode(link.dataset.note || (link.hasAttribute('data-viewer') ? 'Interactive structure · Mol*. Close the viewer to release its rendering resources. ' : ' ')));
    const direct = document.createElement('a'); direct.href = link.href; direct.target = '_blank'; direct.rel = 'noopener'; direct.textContent = 'Open separately ↗';
    note.append(document.createTextNode(' '), direct);
    if (link.hasAttribute('data-video')) {
      activeType = 'video';
      const video = document.createElement('video');
      video.controls = true; video.playsInline = true; video.muted = true; video.preload = 'none';
      video.setAttribute('aria-label', title.textContent);
      video.addEventListener('loadeddata', () => { if (generation === token) status.textContent = ''; }, { once: true });
      video.addEventListener('error', () => { if (generation === token) status.textContent = 'This video could not be loaded. Open the file separately, or close and try again.'; });
      media = video; body.append(video); status.textContent = 'Loading the selected video…';
      video.src = link.href;
    } else if (link.hasAttribute('data-viewer')) {
      activeType = 'viewer';
      const frame = document.createElement('iframe'); frame.title = title.textContent;
      frame.allowFullscreen = true; frame.referrerPolicy = 'strict-origin-when-cross-origin';
      media = frame; body.append(frame); status.textContent = 'Loading the selected 3D structure…';
      frame.src = link.href;
      timer = window.setTimeout(() => { if (generation === token) status.textContent = 'The 3D renderer is taking longer than expected. The standalone viewer offers retry and structure-download controls.'; }, 30000);
    } else {
      activeType = 'image';
      const image = document.createElement('img');
      image.alt = $('img', link)?.alt || title.textContent;
      image.addEventListener('error', () => { if (generation === token) status.textContent = 'The figure could not be loaded. Use the direct link above.'; });
      media = image; body.append(image); image.src = link.href;
    }
    if (!dialog.open) { oldOverflow = document.body.style.overflow; document.body.style.overflow = 'hidden'; dialog.showModal(); }
    if (activeType === 'video') {
      media.play().catch(() => { if (generation === token) status.textContent = 'Press the player’s play button to start, or open the video separately.'; });
    }
  });
  window.addEventListener('message', e => {
    if (activeType !== 'viewer' || e.source !== media?.contentWindow || e.origin !== location.origin) return;
    if (e.data?.type === 'duetrna:ready') { clearTimeout(timer); status.textContent = ''; }
    if (e.data?.type === 'duetrna:error') { clearTimeout(timer); status.textContent = 'The 3D viewer could not finish loading. Use its Retry button or download the PDB structure.'; }
    if (e.data?.type === 'duetrna:escape') closeMedia();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && media instanceof HTMLVideoElement) media.pause();
  });
  window.addEventListener('pagehide', () => {
    if (dialog.open) dialog.close();
    disposeMedia(); document.body.style.overflow = oldOverflow;
  });

  const copy = $('#copy-citation'), bib = $('#bibtex'), copyStatus = $('#copy-status');
  if (copy && bib && copyStatus) copy.addEventListener('click', async () => {
    let success = false;
    try { await navigator.clipboard.writeText(bib.textContent); success = true; } catch (_) {
      const field = document.createElement('textarea'); field.value = bib.textContent;
      field.style.cssText = 'position:fixed;top:0;left:-9999px'; document.body.append(field); field.select();
      try { success = document.execCommand('copy'); } catch (_) { /* The citation remains selectable. */ }
      field.remove(); copy.focus();
    }
    if (success) copyStatus.textContent = 'Citation copied.';
    else {
      const range = document.createRange(); range.selectNodeContents(bib);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      copyStatus.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });

  const filters = [...document.querySelectorAll('[data-filter-length]')];
  const cards = [...document.querySelectorAll('[data-length]')];
  const count = $('#sample-count');
  for (const button of filters) button.addEventListener('click', () => {
    const length = button.dataset.filterLength;
    filters.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    let shown = 0;
    cards.forEach(card => { card.hidden = length !== 'all' && card.dataset.length !== length; if (!card.hidden) shown++; });
    if (count) count.textContent = `${shown} recorded videos${length === 'all' ? ' · all lengths' : ` · ${length} nt`}`;
    document.querySelectorAll('[data-sample-group]').forEach(group => { group.hidden = ![...group.querySelectorAll('[data-length]')].some(card => !card.hidden); });
  });
})();
