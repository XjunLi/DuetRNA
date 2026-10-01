/* Static final-frame posters; initialize only the selected inline video. */
(() => {
  'use strict';
  let active = null;
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    for (const entry of entries) if (!entry.isIntersecting && active?.card === entry.target) active.video.pause();
  }, { threshold: 0 }) : null;

  function release() {
    if (!active) return;
    const old = active; active = null;
    observer?.unobserve(old.card);
    old.video.pause(); old.video.removeAttribute('src'); old.video.load(); old.video.remove();
    old.preview.hidden = false; old.reset.hidden = true; old.message.textContent = '';
    old.card.classList.remove('is-playing');
  }
  function restore(message, focus = false) {
    if (!active) return;
    const old = active; release();
    old.message.textContent = message;
    if (focus) old.preview.focus({ preventScroll: true });
  }
  document.addEventListener('click', event => {
    const reset = event.target.closest('.film-reset');
    if (reset && active?.reset === reset) { restore('', true); return; }
    const other = event.target.closest('[data-viewer],[data-image],[data-filter-length]');
    if (other && !event.ctrlKey && !event.metaKey && event.button === 0) { release(); return; }
    const preview = event.target.closest('a[data-inline-video]');
    if (!preview || event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    release();
    const card = preview.closest('[data-film]');
    const stage = card.querySelector('.film-stage');
    const video = document.createElement('video');
    video.controls = true; video.playsInline = true; video.muted = true; video.preload = 'none';
    video.poster = preview.querySelector('img').currentSrc || preview.querySelector('img').src;
    video.tabIndex = 0; video.setAttribute('aria-label', preview.getAttribute('aria-label').replace(/^Play /, ''));
    const session = { card, preview, video, reset: card.querySelector('.film-reset'), message: card.querySelector('.film-status') };
    active = session; preview.hidden = true; session.reset.hidden = false;
    stage.append(video); card.classList.add('is-playing');
    session.message.textContent = 'Loading video…';
    video.addEventListener('loadeddata', () => { if (active === session) session.message.textContent = ''; }, { once: true });
    video.addEventListener('playing', () => { if (active === session) session.message.textContent = ''; });
    video.addEventListener('error', () => {
      if (active === session) restore('Unable to load the video. Click the preview to retry, or use ↗ to open it separately.');
    });
    observer?.observe(card);
    video.src = preview.href;
    // play() is called in the click handler so Safari retains the user gesture.
    const play = video.play();
    if (play?.catch) play.catch(error => {
      if (active !== session || error.name === 'AbortError') return;
      if (error.name === 'NotAllowedError') session.message.textContent = 'Press play in the video controls to start.';
      else restore('Unable to start the video. Click the preview to retry, or open it separately with ↗.');
    });
    video.focus({ preventScroll: true });
  }, true);
  document.addEventListener('visibilitychange', () => { if (document.hidden) active?.video.pause(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && active && !document.fullscreenElement) restore('', true);
  });
  window.addEventListener('pagehide', release);
})();
