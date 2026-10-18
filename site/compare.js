document.querySelectorAll('[data-comparison]').forEach(card => {
  const view = card.querySelector('.comparison');
  let dragging = false;
  const reveal = value => {
    const position = Math.max(0, Math.min(100, value));
    view.style.setProperty('--reveal', `${position}%`);
    view.setAttribute('aria-valuenow', Math.round(position));
    view.setAttribute('aria-valuetext', `${Math.round(position)}% input`);
  };
  const fromPointer = event => {
    const bounds = view.getBoundingClientRect();
    reveal((event.clientX - bounds.left) / bounds.width * 100);
  };
  view.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button !== 0) return;
    dragging = true;
    view.setPointerCapture(event.pointerId);
    view.focus({preventScroll: true});
    fromPointer(event);
  });
  view.addEventListener('pointermove', event => {
    if (dragging && view.hasPointerCapture(event.pointerId)) fromPointer(event);
  });
  const stop = () => { dragging = false; };
  view.addEventListener('pointerup', stop);
  view.addEventListener('pointercancel', stop);
  view.addEventListener('lostpointercapture', stop);
  view.addEventListener('keydown', event => {
    const current = Number(view.getAttribute('aria-valuenow'));
    const step = event.shiftKey ? 10 : 1;
    const values = {ArrowLeft: current - step, ArrowDown: current - step,
      ArrowRight: current + step, ArrowUp: current + step, Home: 0, End: 100};
    if (Object.hasOwn(values, event.key)) {
      event.preventDefault();
      reveal(values[event.key]);
    }
  });
});
