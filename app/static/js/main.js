document.addEventListener('DOMContentLoaded', () => {
  const vp = document.querySelector('.tree-viewport');
  if (vp) vp.addEventListener('click', () => vp.classList.toggle('zoomed'));
});
