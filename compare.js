document.querySelectorAll('[data-comparison]').forEach(card => {
  const range = card.querySelector('input[type="range"]');
  const view = card.querySelector('.comparison');
  const reveal = value => {
    range.value = value;
    view.style.setProperty('--reveal', `${value}%`);
  };
  range.addEventListener('input', () => reveal(range.value));
  card.querySelectorAll('[data-reveal]').forEach(button => {
    button.addEventListener('click', () => reveal(button.dataset.reveal));
  });
});
