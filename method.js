// This illustrates only which image condition is used; it does not synthesize images.
const stepControl = document.getElementById('denoising-step');
if (stepControl) {
  const status = document.getElementById('step-status');
  const explanation = document.getElementById('step-explanation');
  const updateSchedule = () => {
    const step = Number(stepControl.value);
    const early = step <= 3;
    document.querySelectorAll('[data-step]').forEach(cell => {
      cell.classList.toggle('is-current', Number(cell.dataset.step) === step);
    });
    status.textContent = `Step ${step} / 10 — ${early ? 'global guidance' : 'detail recovery'}`;
    explanation.textContent = early
      ? 'The image condition is the upsampled Stage 1 restoration, supplying global structure and color guidance.'
      : 'The image condition is the original capture, supplying texture and fine detail during the later denoising steps.';
    stepControl.setAttribute('aria-valuetext', `Step ${step}: ${early ? 'Stage 1 restoration condition' : 'original input condition'}`);
  };
  stepControl.addEventListener('input', updateSchedule);
  updateSchedule();
}
