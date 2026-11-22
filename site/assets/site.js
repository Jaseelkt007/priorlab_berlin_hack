/* Keys: 1-4 open a project, Esc goes back to the overview. (Inside a report, click the header first.) */
(function () {
  'use strict';
  const pages = ['picker.html', 'crane.html', 'drone.html', 'fly.html'];
  window.addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'Escape' && !/index\.html$|\/$/.test(location.pathname)) { location.href = 'index.html'; return; }
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= pages.length) location.href = pages[n - 1];
  });
})();
