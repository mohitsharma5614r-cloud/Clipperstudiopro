// Clipper Studio • Professional Web Engine Client
document.addEventListener('DOMContentLoaded', () => {
  const clipForm = document.getElementById('clipForm');
  const videoUrlInput = document.getElementById('videoUrl');
  const pasteBtn = document.getElementById('pasteBtn');
  const aspectRatioInput = document.getElementById('aspectRatio');
  const aspectSegmented = document.getElementById('aspectSegmented');
  const numClipsInput = document.getElementById('numClips');
  const numClipsVal = document.getElementById('numClipsVal');
  const minLenInput = document.getElementById('minLen');
  const maxLenInput = document.getElementById('maxLen');
  const submitBtn = document.getElementById('submitBtn');

  const progressSection = document.getElementById('progressSection');
  const progressBar = document.getElementById('progressBar');
  const progressPercent = document.getElementById('progressPercent');
  const progressStage = document.getElementById('progressStage');
  const progressSubtitle = document.getElementById('progressSubtitle');

  const stepIngest = document.getElementById('stepIngest');
  const stepSignals = document.getElementById('stepSignals');
  const stepDetect = document.getElementById('stepDetect');
  const stepRender = document.getElementById('stepRender');

  const resultsSection = document.getElementById('resultsSection');
  const resultsVideoTitle = document.getElementById('resultsVideoTitle');
  const resultsMeta = document.getElementById('resultsMeta');
  const clipsGrid = document.getElementById('clipsGrid');
  const downloadAllBtn = document.getElementById('downloadAllBtn');

  let currentJobId = null;
  let pollInterval = null;

  // Theme Toggle Controller (Light & Dark Mode)
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const sunIcon = themeToggleBtn?.querySelector('.sun-icon');
  const moonIcon = themeToggleBtn?.querySelector('.moon-icon');
  const themeText = document.getElementById('themeToggleText');

  function updateThemeUI(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('clipper-theme', theme);
    } catch (e) {}

    if (theme === 'light') {
      if (sunIcon) sunIcon.style.display = 'none';
      if (moonIcon) moonIcon.style.display = 'inline-block';
      if (themeText) themeText.textContent = 'Dark';
      if (themeToggleBtn) themeToggleBtn.title = 'Switch to Dark Theme';
    } else {
      if (sunIcon) sunIcon.style.display = 'inline-block';
      if (moonIcon) moonIcon.style.display = 'none';
      if (themeText) themeText.textContent = 'Light';
      if (themeToggleBtn) themeToggleBtn.title = 'Switch to Light Theme';
    }
  }

  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  updateThemeUI(currentTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const active = document.documentElement.getAttribute('data-theme') || 'dark';
      const target = active === 'dark' ? 'light' : 'dark';
      updateThemeUI(target);
    });
  }

  // Clipboard Paste Helper
  if (pasteBtn) {
    pasteBtn.addEventListener('click', async () => {
      try {
        const text = await navigator.clipboard.readText();
        if (text) {
          videoUrlInput.value = text.trim();
          videoUrlInput.focus();
        }
      } catch (err) {
        videoUrlInput.focus();
      }
    });
  }

  // Segmented Aspect Ratio Control
  if (aspectSegmented) {
    const buttons = aspectSegmented.querySelectorAll('.segment-btn');
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        buttons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        aspectRatioInput.value = btn.dataset.aspect;
      });
    });
  }

  // Sync slider label
  numClipsInput.addEventListener('input', (e) => {
    numClipsVal.textContent = `${e.target.value} Clips`;
  });

  // Handle Form Submission
  clipForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = videoUrlInput.value.trim();
    if (!url) return;

    setSubmittingState(true);

    resultsSection.style.display = 'none';
    progressSection.style.display = 'block';
    updateProgress(5, 'Initializing Pipeline...', 'Connecting to media ingestion worker', 'queued');

    try {
      const response = await fetch('/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: url,
          num_clips: parseInt(numClipsInput.value, 10),
          min_length: parseFloat(minLenInput.value),
          max_length: parseFloat(maxLenInput.value),
          aspect_ratio: aspectRatioInput.value,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to create pipeline job');
      }

      const data = await response.json();
      currentJobId = data.job_id;
      if (data.status === 'completed') {
        loadClips(currentJobId);
        setSubmittingState(false);
      } else {
        startPolling(currentJobId);
      }
    } catch (err) {
      alert(`Pipeline error: ${err.message}`);
      setSubmittingState(false);
    }
  });

  // Poll Job Status
  function startPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/jobs/${jobId}`);
        if (!res.ok) throw new Error('Could not fetch pipeline status');
        const job = await res.json();

        updateProgress(job.progress, job.stage, `Pipeline Status: ${job.status.toUpperCase()}`, job.status);

        if (job.status === 'completed') {
          clearInterval(pollInterval);
          loadClips(jobId);
          setSubmittingState(false);
        } else if (job.status === 'failed') {
          clearInterval(pollInterval);
          alert(`Pipeline processing failed: ${job.error || 'Unknown error'}`);
          setSubmittingState(false);
        }
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 1100);
  }

  // Update Progress Track & Stages
  function updateProgress(percent, stageText, subText, status) {
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    progressStage.textContent = stageText || 'Processing pipeline...';
    progressSubtitle.textContent = subText || '';

    const stages = [stepIngest, stepSignals, stepDetect, stepRender];
    stages.forEach((s) => s.className = 'stage-step');

    if (percent < 35) {
      stepIngest.className = 'stage-step active';
    } else if (percent < 60) {
      stepIngest.className = 'stage-step completed';
      stepSignals.className = 'stage-step active';
    } else if (percent < 65) {
      stepIngest.className = 'stage-step completed';
      stepSignals.className = 'stage-step completed';
      stepDetect.className = 'stage-step active';
    } else {
      stepIngest.className = 'stage-step completed';
      stepSignals.className = 'stage-step completed';
      stepDetect.className = 'stage-step completed';
      stepRender.className = 'stage-step active';
    }

    if (status === 'completed') {
      stages.forEach((s) => s.className = 'stage-step completed');
    }
  }

  // Parse Reason into Structured Chips
  function renderReasonChips(reason) {
    if (!reason) return '';
    // Strip raw emojis and split by '+' or bullet
    const parts = reason
      .split('+')
      .map((p) => p.replace(/[⚡💥🔥🎬🎙️⏱️★]/g, '').trim())
      .filter(Boolean);

    return parts
      .map((tag, idx) => {
        const isHighlight = idx === 0 || tag.toLowerCase().includes('hook') || tag.toLowerCase().includes('climax');
        return `<span class="reason-chip ${isHighlight ? 'highlight-chip' : ''}">${escapeHtml(tag)}</span>`;
      })
      .join('');
  }

  // Load and Render Clips
  async function loadClips(jobId) {
    try {
      const res = await fetch(`/jobs/${jobId}/clips`);
      if (!res.ok) throw new Error('Could not retrieve rendered clips');
      const data = await res.json();

      resultsVideoTitle.textContent = data.video_title || 'Rendered Clips';
      resultsMeta.textContent = `${data.clips_count} high-retention moments reframed in 1080p HD`;
      clipsGrid.innerHTML = '';

      const aspectClass = `aspect-${data.clips[0]?.aspect_ratio.replace(':', '-') || '9-16'}`;

      data.clips.forEach((clip) => {
        const card = document.createElement('div');
        const isTop = clip.is_top_viral || clip.clip_index === 1;
        card.className = `studio-clip-card ${isTop ? 'rank-top-card' : ''}`;

        const startTime = formatTime(clip.start_sec);
        const endTime = formatTime(clip.end_sec);
        const isUltra = clip.score >= 80;

        card.innerHTML = `
          <div class="clip-frame ${aspectClass}">
            <video controls preload="metadata" playsinline>
              <source src="${clip.video_url}" type="video/mp4">
              Your browser does not support HTML5 video playback.
            </video>
            <div class="clip-overlay-top">
              <span class="rank-pill ${isTop ? 'rank-1' : ''}">
                ${isTop ? 'RANK #1 • PEAK ENGAGEMENT' : `RANK #${clip.clip_index}`}
              </span>
              <span class="format-pill">1080p • ${clip.aspect_ratio}</span>
            </div>
          </div>

          <div class="clip-body">
            <div class="clip-title-row">
              <h4 class="clip-title">${escapeHtml(clip.title)}</h4>
              <div class="score-badge">
                <span class="score-val ${isUltra ? 'ultra-score' : ''}">${clip.score.toFixed(0)}</span>
                <span class="score-label">Index</span>
              </div>
            </div>

            <div class="timecode-row">
              <span class="timecode-text">${startTime} — ${endTime}</span>
              <span class="duration-pill">${clip.duration.toFixed(1)}s</span>
            </div>

            <div class="reason-tags-row">
              ${renderReasonChips(clip.reason)}
            </div>

            <div class="telemetry-block">
              <div class="telemetry-row">
                <span class="telemetry-label">Audience Heatmap</span>
                <div class="telemetry-meter-group">
                  <div class="telemetry-track">
                    <div class="telemetry-fill fill-retention" style="width: ${clip.breakdown.heatmap}%;"></div>
                  </div>
                  <span class="telemetry-val">${clip.breakdown.heatmap}%</span>
                </div>
              </div>

              <div class="telemetry-row">
                <span class="telemetry-label">Acoustic Energy</span>
                <div class="telemetry-meter-group">
                  <div class="telemetry-track">
                    <div class="telemetry-fill fill-acoustic" style="width: ${clip.breakdown.audio}%;"></div>
                  </div>
                  <span class="telemetry-val">${clip.breakdown.audio}%</span>
                </div>
              </div>

              <div class="telemetry-row">
                <span class="telemetry-label">Visual Dynamics</span>
                <div class="telemetry-meter-group">
                  <div class="telemetry-track">
                    <div class="telemetry-fill fill-motion" style="width: ${clip.breakdown.visual}%;"></div>
                  </div>
                  <span class="telemetry-val">${clip.breakdown.visual}%</span>
                </div>
              </div>

              <div class="telemetry-row">
                <span class="telemetry-label">Speech Cadence</span>
                <div class="telemetry-meter-group">
                  <div class="telemetry-track">
                    <div class="telemetry-fill fill-cadence" style="width: ${clip.breakdown.pacing}%;"></div>
                  </div>
                  <span class="telemetry-val">${clip.breakdown.pacing}%</span>
                </div>
              </div>
            </div>

            <div class="clip-actions-row">
              <a href="${clip.download_url}" class="btn-export-clip" download>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <span>Export 1080p MP4</span>
              </a>
              <button class="btn-copy-time" title="Copy timestamp" data-time="${startTime} - ${endTime}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect>
                  <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>
                </svg>
              </button>
            </div>
          </div>
        `;

        const copyBtn = card.querySelector('.btn-copy-time');
        copyBtn.addEventListener('click', () => {
          navigator.clipboard.writeText(copyBtn.dataset.time);
          copyBtn.style.color = '#10b981';
          setTimeout(() => { copyBtn.style.color = ''; }, 1200);
        });

        clipsGrid.appendChild(card);
      });

      downloadAllBtn.onclick = () => {
        window.location.href = `/jobs/${jobId}/download-all`;
      };

      progressSection.style.display = 'none';
      resultsSection.style.display = 'block';
    } catch (e) {
      alert(`Error loading clips: ${e.message}`);
    }
  }

  function setSubmittingState(isSubmitting) {
    submitBtn.disabled = isSubmitting;
    const btnText = submitBtn.querySelector('.btn-text');
    const btnArrow = submitBtn.querySelector('.btn-arrow');
    const btnLoader = submitBtn.querySelector('.btn-loader');

    if (isSubmitting) {
      btnText.textContent = 'Analyzing...';
      if (btnArrow) btnArrow.style.display = 'none';
      if (btnLoader) btnLoader.style.display = 'inline-block';
    } else {
      btnText.textContent = 'Extract Clips';
      if (btnArrow) btnArrow.style.display = 'inline-block';
      if (btnLoader) btnLoader.style.display = 'none';
    }
  }

  function formatTime(sec) {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${r.toString().padStart(2, '0')}`;
    }
    return `${m.toString().padStart(2, '0')}:${r.toString().padStart(2, '0')}`;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }
});
