/* Pitch deck: scale the 1920x1080 frame to the window, move with keys, clicks or swipes,
   play only the active slide's videos, keep the slide number in the URL (#1 ... #5). */
(function () {
  'use strict';
  const frame = document.getElementById('slides');
  const slides = Array.from(frame.querySelectorAll('.slide'));
  const count = document.getElementById('count');
  let cur = -1;

  function fit() {
    const s = Math.min(innerWidth / 1920, innerHeight / 1080);
    document.documentElement.style.setProperty('--s', s.toFixed(4));
  }
  addEventListener('resize', fit);
  fit();

  function go(i) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    if (i === cur) return;
    cur = i;
    slides.forEach((el, k) => {
      el.classList.toggle('on', k === i);
      el.classList.toggle('before', k < i);
      el.querySelectorAll('video').forEach((v) => {
        if (k === i) { const p = v.play(); if (p) p.catch(() => {}); } else v.pause();
      });
    });
    count.textContent = `${i + 1} / ${slides.length}`;
    frame.style.setProperty('--p', ((i + 1) / slides.length).toFixed(3));
    history.replaceState(null, '', `#${i + 1}`);
  }

  addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const k = e.key;
    if (['ArrowRight', 'PageDown', ' ', 'Enter', 'ArrowDown'].includes(k)) { e.preventDefault(); go(cur + 1); }
    else if (['ArrowLeft', 'PageUp', 'Backspace', 'ArrowUp'].includes(k)) { e.preventDefault(); go(cur - 1); }
    else if (k === 'Home') go(0);
    else if (k === 'End') go(slides.length - 1);
    else if (k === 'f' || k === 'F') toggleFull();
    else if (/^[1-9]$/.test(k)) go(parseInt(k, 10) - 1);
  });

  function toggleFull() {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
  }
  document.getElementById('prev').addEventListener('click', () => go(cur - 1));
  document.getElementById('next').addEventListener('click', () => go(cur + 1));
  document.getElementById('full').addEventListener('click', toggleFull);

  // swipe on touch screens
  let x0 = null;
  frame.addEventListener('pointerdown', (e) => { if (e.pointerType === 'touch') x0 = e.clientX; });
  frame.addEventListener('pointerup', (e) => {
    if (x0 === null) return;
    const dx = e.clientX - x0; x0 = null;
    if (Math.abs(dx) > 50) go(cur + (dx < 0 ? 1 : -1));
  });

  // show the on-screen controls while the mouse moves (and at the start)
  let idle;
  function wake() { document.body.classList.add('awake'); clearTimeout(idle); idle = setTimeout(() => document.body.classList.remove('awake'), 2500); }
  addEventListener('pointermove', wake);
  wake();

  addEventListener('hashchange', () => go(parseInt(location.hash.slice(1), 10) - 1 || 0));
  go((parseInt(location.hash.slice(1), 10) || 1) - 1);
})();
