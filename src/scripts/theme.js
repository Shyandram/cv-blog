// Run before the first paint. Base.astro approves these exact bytes in its CSP.
try {
  const theme = localStorage.getItem('theme');
  if (theme === 'dark' || theme === 'light') document.documentElement.dataset.theme = theme;
} catch {}
