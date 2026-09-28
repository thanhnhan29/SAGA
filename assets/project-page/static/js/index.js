/**
 * SAGA: Stable Acceleration Guidance for Autoregressive Video Generation
 * Interactive Client Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  initAbstractDisclosure();
  initComparisonTheater();
  initGalleryFilters();
  initLightbox();
  initBibtexCopy();
  initScrollspy();
});

function initAbstractDisclosure() {
  const toggle = document.getElementById('abstractToggle');
  const content = document.getElementById('abstractContent');
  const label = document.getElementById('abstractToggleLabel');
  if (!toggle || !content || !label) return;

  toggle.addEventListener('click', () => {
    const isOpen = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', String(!isOpen));
    content.hidden = isOpen;
    label.textContent = isOpen ? 'Show abstract' : 'Hide abstract';
  });
}

// ============================================================================
// 1. Comparison Theater & Synchronized Video Player
// ============================================================================

const PRESETS = [
  {
    id: 'cyberpunk_robot',
    name: 'Cyberpunk Robot',
    length: '5s',
    baseline: './static/videos/comparisons/5s/baseline/cyberpunk_robot.mp4',
    saga: './static/videos/comparisons/5s/saga/cyberpunk_robot.mp4',
    prompt: 'A cyberpunk-style illustration depicting a lone robot navigating a neon-lit cityscape. The robot stands tall with sleek, metallic armor, adorned with blinking lights and wires. Its eyes, glowing with a deep blue hue, scan the surroundings with curiosity.'
  },
  {
    id: 'alley_tsunami',
    name: 'Alley Tsunami',
    length: '5s',
    baseline: './static/videos/comparisons/5s/baseline/alley_tsunami.mp4',
    saga: './static/videos/comparisons/5s/saga/alley_tsunami.mp4',
    prompt: 'A dramatic and dynamic scene in the style of a disaster movie, depicting a powerful tsunami rushing through a narrow alley in Bulgaria. The water is turbulent and chaotic, with waves crashing violently.'
  },
  {
    id: 'corgi_beach',
    name: 'Corgi on Maui Beach',
    length: '10s',
    baseline: './static/videos/comparisons/10s/baseline/corgi_beach.mp4',
    saga: './static/videos/comparisons/10s/saga/corgi_beach.mp4',
    prompt: 'A vibrant and lively vlog-style photo of a corgi in tropical Maui, showcasing the dog energetically filming itself on a sandy beach with turquoise waters under bright sunshine. Continuous 10-second rollout.'
  },
  {
    id: 'dandelion_macro',
    name: 'Dandelion Macro Zoom',
    length: '10s',
    baseline: './static/videos/comparisons/10s/baseline/dandelion_macro.mp4',
    saga: './static/videos/comparisons/10s/saga/dandelion_macro.mp4',
    prompt: 'A hyper-realistic macro photograph capturing the intricate details of a dandelion, continuous long-horizon autoregressive rollout over 10 seconds without drift or structural collapse.'
  },
  {
    id: 'floating_balloon',
    name: 'Floating Red Balloon',
    length: '5s',
    baseline: './static/videos/comparisons/5s/baseline/floating_balloon.mp4',
    saga: './static/videos/comparisons/5s/saga/floating_balloon.mp4',
    prompt: 'A handheld tracking shot following a red balloon floating above the ground in an abandoned street. The balloon drifts gracefully, its bright red color contrasting sharply against the decaying urban background.'
  },
  {
    id: 'underwater_city',
    name: 'Underwater City FPV',
    length: '5s',
    baseline: './static/videos/comparisons/5s/baseline/underwater_city.mp4',
    saga: './static/videos/comparisons/5s/saga/underwater_city.mp4',
    prompt: 'A dynamic FPV aerial view of a vibrant underwater suburban neighborhood, where colorful corals line the streets. The camera moves swiftly, capturing the intricate details of the coral formations.'
  }
];

function initComparisonTheater() {
  const sideVideoBase = document.getElementById('sideVideoBase');
  const sideVideoSaga = document.getElementById('sideVideoSaga');
  if (!sideVideoBase || !sideVideoSaga) return;

  // Controls
  const btnPlayPause = document.getElementById('btnPlayPause');
  const seekSlider = document.getElementById('seekSlider');
  const timeDisplay = document.getElementById('timeDisplay');
  const activePromptText = document.getElementById('activePromptText');
  const presetTabs = document.getElementById('presetTabs');

  let currentSpeed = 1.0;
  let isPlaying = false;

  // Build preset pills
  if (presetTabs) {
    presetTabs.innerHTML = '';
    PRESETS.forEach((preset, idx) => {
      const pill = document.createElement('button');
      pill.className = `preset-pill ${idx === 0 ? 'active' : ''}`;
      pill.innerHTML = `
        <span>${preset.name}</span>
        <span class="badge-length">${preset.length}</span>
      `;
      pill.addEventListener('click', () => selectPreset(idx));
      presetTabs.appendChild(pill);
    });
  }

  function selectPreset(idx) {
    const preset = PRESETS[idx];

    // Update active pill
    const pills = presetTabs.querySelectorAll('.preset-pill');
    pills.forEach((p, i) => p.classList.toggle('active', i === idx));

    // Update Prompt
    if (activePromptText) {
      activePromptText.textContent = preset.prompt;
    }

    // Set video sources
    loadVideoPair(preset.baseline, preset.saga);
  }

  function loadVideoPair(baseSrc, sagaSrc) {
    // Set video sources
    sideVideoBase.src = baseSrc;
    sideVideoSaga.src = sagaSrc;

    sideVideoBase.load();
    sideVideoSaga.load();

    // Reset seek
    if (seekSlider) seekSlider.value = 0;
    if (timeDisplay) timeDisplay.textContent = '0.00 / 0.00s';

    // Synchronize play once loaded
    let loadedCount = 0;
    const onLoaded = () => {
      loadedCount++;
      if (loadedCount >= 2) {
        setPlaybackSpeed(currentSpeed);
        playAll();
      }
    };

    sideVideoBase.addEventListener('loadeddata', onLoaded, { once: true });
    sideVideoSaga.addEventListener('loadeddata', onLoaded, { once: true });
  }

  function getActiveVideos() {
    return [sideVideoBase, sideVideoSaga];
  }

  function playAll() {
    const active = getActiveVideos();
    Promise.all(active.map(v => v.play().catch(() => {}))).then(() => {
      isPlaying = true;
      if (btnPlayPause) btnPlayPause.innerHTML = '<i class="fas fa-pause"></i>';
    });
  }

  function pauseAll() {
    const active = getActiveVideos();
    active.forEach(v => v.pause());
    isPlaying = false;
    if (btnPlayPause) btnPlayPause.innerHTML = '<i class="fas fa-play"></i>';
  }

  function togglePlayPause() {
    if (isPlaying) {
      pauseAll();
    } else {
      playAll();
    }
  }

  if (btnPlayPause) {
    btnPlayPause.addEventListener('click', togglePlayPause);
  }

  // Synchronized Loop & Timeupdate
  function syncVideos() {
    const active = getActiveVideos();
    const primary = active[0];
    const secondary = active[1];

    if (!primary || !secondary) return;

    // Drastic drift correction (> 0.08s)
    if (Math.abs(primary.currentTime - secondary.currentTime) > 0.08) {
      secondary.currentTime = primary.currentTime;
    }

    // Update seek bar
    if (primary.duration && !isDraggingSeek) {
      const progress = (primary.currentTime / primary.duration) * 100;
      if (seekSlider) seekSlider.value = progress;
      if (timeDisplay) {
        timeDisplay.textContent = `${primary.currentTime.toFixed(2)} / ${primary.duration.toFixed(2)}s`;
      }
    }
  }

  let isDraggingSeek = false;
  if (seekSlider) {
    seekSlider.addEventListener('input', (e) => {
      isDraggingSeek = true;
      const active = getActiveVideos();
      const primary = active[0];
      if (primary && primary.duration) {
        const targetTime = (e.target.value / 100) * primary.duration;
        active.forEach(v => v.currentTime = targetTime);
        if (timeDisplay) {
          timeDisplay.textContent = `${targetTime.toFixed(2)} / ${primary.duration.toFixed(2)}s`;
        }
      }
    });

    seekSlider.addEventListener('change', () => {
      isDraggingSeek = false;
    });
  }

  sideVideoBase.addEventListener('timeupdate', syncVideos);

  // Seamless synchronized loop
  sideVideoBase.addEventListener('ended', () => {
    sideVideoBase.currentTime = 0;
    sideVideoSaga.currentTime = 0;
    playAll();
  });

  // Playback speed buttons
  const speedButtons = document.querySelectorAll('.speed-btn');
  speedButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const speed = parseFloat(e.target.dataset.speed);
      setPlaybackSpeed(speed);
      speedButtons.forEach(b => b.classList.toggle('active', b === btn));
    });
  });

  function setPlaybackSpeed(speed) {
    currentSpeed = speed;
    getActiveVideos().forEach(v => v.playbackRate = speed);
  }

  // Load initial preset
  selectPreset(0);
}

// ============================================================================
// 2. Qualitative Gallery Filters & Video Playback
// ============================================================================

function initGalleryFilters() {
  const filterBtns = document.querySelectorAll('.gallery-tab-btn');
  const cards = document.querySelectorAll('.gallery-card');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const filter = btn.dataset.filter;

      filterBtns.forEach(b => b.classList.toggle('active', b === btn));

      cards.forEach(card => {
        const category = card.dataset.category || '';
        if (filter === 'all' || category.includes(filter)) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  // Auto-play videos on hover for a preview feel
  cards.forEach(card => {
    const video = card.querySelector('video');
    if (!video) return;

    card.addEventListener('mouseenter', () => {
      video.play().catch(() => {});
    });

    card.addEventListener('mouseleave', () => {
      video.pause();
    });
  });
}

// ============================================================================
// 3. High-Resolution Figure Lightbox
// ============================================================================

function initLightbox() {
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxImg = document.getElementById('lightboxImg');
  const lightboxClose = document.getElementById('lightboxClose');
  const zoomTriggers = document.querySelectorAll('[data-zoomable]');

  if (!lightboxModal || !lightboxImg) return;

  zoomTriggers.forEach(el => {
    el.addEventListener('click', () => {
      const img = el.querySelector('img') || el;
      if (img && img.src) {
        lightboxImg.src = img.src;
        lightboxModal.classList.add('active');
        document.body.style.overflow = 'hidden';
      }
    });
  });

  function closeLightbox() {
    lightboxModal.classList.remove('active');
    document.body.style.overflow = '';
  }

  if (lightboxClose) {
    lightboxClose.addEventListener('click', closeLightbox);
  }

  lightboxModal.addEventListener('click', (e) => {
    if (e.target === lightboxModal) closeLightbox();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightboxModal.classList.contains('active')) {
      closeLightbox();
    }
  });
}

// ============================================================================
// 4. BibTeX Citation Copy Button with Toast
// ============================================================================

function initBibtexCopy() {
  const btnCopy = document.getElementById('btnCopyBibtex');
  const bibtexCode = document.getElementById('bibtexCode');
  const toastMsg = document.getElementById('toastMsg');

  if (!btnCopy || !bibtexCode) return;

  btnCopy.addEventListener('click', () => {
    const textToCopy = bibtexCode.innerText.trim();
    navigator.clipboard.writeText(textToCopy).then(() => {
      const originalHTML = btnCopy.innerHTML;
      btnCopy.innerHTML = '<i class="fas fa-check"></i> Copied!';
      btnCopy.style.background = '#15803d';

      showToast('BibTeX citation copied to clipboard!');

      setTimeout(() => {
        btnCopy.innerHTML = originalHTML;
        btnCopy.style.background = '';
      }, 2500);
    }).catch(err => {
      console.error('Failed to copy text: ', err);
    });
  });
}

function showToast(msg) {
  const toast = document.getElementById('toastMsg');
  if (!toast) return;
  toast.querySelector('.toast-text').textContent = msg;
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 2800);
}

// ============================================================================
// 5. Scrollspy Navigation
// ============================================================================

function initScrollspy() {
  const nav = document.querySelector('.site-nav');
  const links = document.querySelectorAll('.nav-link');
  const sections = Array.from(links).map(link => {
    const href = link.getAttribute('href');
    return href.startsWith('#') ? document.querySelector(href) : null;
  }).filter(Boolean);

  window.addEventListener('scroll', () => {
    const scrollPos = window.scrollY + 100;

    // Sticky nav elevation
    if (nav) {
      nav.classList.toggle('scrolled', window.scrollY > 20);
    }

    // Scrollspy section active highlight
    let currentId = '';
    sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = section.getAttribute('id');
      }
    });

    links.forEach(link => {
      const href = link.getAttribute('href');
      link.classList.toggle('active', href === `#${currentId}`);
    });
  }, { passive: true });
}
