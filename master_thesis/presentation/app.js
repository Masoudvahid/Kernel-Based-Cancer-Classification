document.addEventListener('DOMContentLoaded', () => {
  // --- Presentation State ---
  let currentSlideIndex = 0;
  const slides = document.querySelectorAll('.slide');
  const slideNumEl = document.querySelector('.slide-num');
  const sectionTagEl = document.querySelector('.slide-header .section-tag');

  // --- Active Slide Updates ---
  function updateSlide(index) {
    slides[currentSlideIndex].classList.remove('active');
    slides[index].classList.add('active');
    currentSlideIndex = index;
    
    // Update Slide count in footer
    slideNumEl.textContent = `${currentSlideIndex + 1} / ${slides.length}`;
    
    // Update Header Section Title
    const currentSection = slides[currentSlideIndex].getAttribute('data-section') || 'Introduction';
    sectionTagEl.textContent = currentSection;

    // Update Speaker Notes Panel
    const activeNotesEl = document.getElementById('active-notes');
    if (activeNotesEl) {
      const notesSource = slides[index].querySelector('.speaker-notes');
      if (notesSource) {
        activeNotesEl.innerHTML = notesSource.innerHTML;
      } else {
        activeNotesEl.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">No speaker notes for this slide.</p>';
      }
    }
  }

  // --- Keyboard navigation ---
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      if (currentSlideIndex < slides.length - 1) {
        updateSlide(currentSlideIndex + 1);
      }
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      if (currentSlideIndex > 0) {
        updateSlide(currentSlideIndex - 1);
      }
    }
  });

  // --- On-screen Navigation Controls ---
  document.getElementById('prev-slide-btn').addEventListener('click', () => {
    if (currentSlideIndex > 0) updateSlide(currentSlideIndex - 1);
  });

  document.getElementById('next-slide-btn').addEventListener('click', () => {
    if (currentSlideIndex < slides.length - 1) updateSlide(currentSlideIndex + 1);
  });

  // Add slide numbers dynamically to each slide
  slides.forEach((slide, index) => {
    const slideNumberEl = document.createElement('div');
    slideNumberEl.className = 'slide-number';
    const pageNum = index + 1;
    slideNumberEl.textContent = pageNum;
    slide.appendChild(slideNumberEl);
  });

  // Initialize
  updateSlide(0);
});
