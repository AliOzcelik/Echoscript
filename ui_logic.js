
class Component extends DCLogic {
  SPCOL = {
    light: ['#d64545','#0f9488','#7c4dff','#c07800','#2563c9','#3a9142'],
    dark:  ['#ff7b7b','#34d0c0','#b39bff','#f0b64a','#6aa6ff','#6fd47a']
  };

  TOK = {
    studio: { label:'Studio', radius:'14px', fd:"'Newsreader',Georgia,serif", fb:"'Public Sans',system-ui,sans-serif", fm:"'IBM Plex Mono',ui-monospace,monospace",
      light:{ bg:'#f4f0e7',surface:'#fffdf8',surface2:'#efe8da',border:'#e5ddca',text:'#2b2620',muted:'#736a5c',faint:'#a99d88',accent:'#c2673b',accentSoft:'#f2e0d3',accentText:'#fffdf8',shadow:'0 1px 2px rgba(60,45,25,.06),0 10px 26px -14px rgba(60,45,25,.22)' },
      dark:{ bg:'#17130e',surface:'#211b14',surface2:'#2b231a',border:'#3a3025',text:'#f2ece0',muted:'#b3a897',faint:'#7d7263',accent:'#e08a5a',accentSoft:'#3a2a1e',accentText:'#17130e',shadow:'0 1px 2px rgba(0,0,0,.4),0 14px 32px -16px rgba(0,0,0,.7)' } },
    signal: { label:'Signal', radius:'10px', fd:"'Space Grotesk',system-ui,sans-serif", fb:"'Public Sans',system-ui,sans-serif", fm:"'IBM Plex Mono',ui-monospace,monospace",
      light:{ bg:'#eceff3',surface:'#ffffff',surface2:'#e7ebf1',border:'#d3dbe4',text:'#111826',muted:'#586173',faint:'#8792a3',accent:'#1f6feb',accentSoft:'#e2ecfe',accentText:'#ffffff',shadow:'0 1px 2px rgba(20,30,50,.06),0 12px 26px -14px rgba(20,30,50,.2)' },
      dark:{ bg:'#0b0e13',surface:'#12171f',surface2:'#1a212c',border:'#232c38',text:'#e7edf5',muted:'#909cad',faint:'#5e6a7b',accent:'#3b8bff',accentSoft:'#122442',accentText:'#04070c',shadow:'0 1px 2px rgba(0,0,0,.5),0 14px 32px -16px rgba(0,0,0,.75)' } },
    aurora: { label:'Aurora', radius:'18px', fd:"'Bricolage Grotesque',system-ui,sans-serif", fb:"'Instrument Sans',system-ui,sans-serif", fm:"'IBM Plex Mono',ui-monospace,monospace",
      light:{ bg:'#f3f1fb',surface:'#ffffff',surface2:'#efeaff',border:'#e5e0f4',text:'#191531',muted:'#635d7d',faint:'#9992b5',accent:'#6d4aff',accentSoft:'#eae4ff',accentText:'#ffffff',shadow:'0 1px 2px rgba(40,25,80,.06),0 14px 30px -14px rgba(70,45,150,.28)' },
      dark:{ bg:'#0e0c18',surface:'#171326',surface2:'#211c37',border:'#2d2749',text:'#efecff',muted:'#a49dc4',faint:'#6d6690',accent:'#8a6bff',accentSoft:'#241d40',accentText:'#0b0818',shadow:'0 1px 2px rgba(0,0,0,.5),0 16px 36px -18px rgba(0,0,0,.75)' } }
  };

  STR = {
    en: { tagline:'On-device meeting notes', nav_meetings:'Meetings', nav_new:'New recording', device_status:'Offline · on-device', meetings_title:'Meetings', lib_search:'Search meetings', speakers_word:'speakers', st_done:'Ready', st_processing:'Processing', st_queued:'Queued', st_error:'Error', new_title:'New recording', cap_record:'Record', cap_upload:'Upload', cap_rec_hint:'Tap to start recording — audio never leaves this device.', cap_recording:'Recording…', cap_stop:'Stop', cap_rec_again:'Record again', cap_drop:'Drop an audio file here', cap_browse:'or browse files', cap_formats:'WAV, MP3, M4A · up to 2 hours', cap_lang:'Transcription language', cap_lang_note:'Sets the language of the transcript — separate from the interface language above.', cap_process:'Process on device', cap_captured:'Recording captured', proc_note:'Transcription and diarization run side by side — nothing leaves this device.', proc_done:'Notes are ready', proc_done_sub:'Transcript generated on device.', proc_view:'View results', stage_convert:'Convert', stage_convert_sub:'Decode to 16 kHz mono', stage_vad:'Detect speech', stage_vad_sub:'Find where people talk', stage_pack:'Chunk', stage_pack_sub:'Group speech into 30 s windows', stage_stt:'Transcribe', stage_stt_sub:'Speech to text', stage_diar:'Diarize', stage_diar_sub:'Who spoke when', stage_merge:'Merge', stage_merge_sub:'Align speakers & words', stage_summary:'Summarize', stage_summary_sub:'Local LLM summary', tab_transcript:'Transcript', tab_summary:'Summary', sp_panel_title:'Speakers', sp_panel_hint:'Name each voice — labels update everywhere.', sp_name_ph:'Add a name', sp_play:'Play sample', sp_playing:'Playing…', sp_color:'Color', turns_word:'turns', sum_overview:'Overview', sum_key:'Key points', sum_decisions:'Decisions', sum_actions:'Action items', export_transcript:'transcript.md', export_summary:'summary.md', back_word:'Meetings', speaker_word:'Speaker', theme_word:'Theme', sp_close:'Close', rec_title:'Recording', uploading:'Uploading…', st_recording:'Recording', mic_error:'Could not start the Pi’s microphone', cap_auto_sum:'Summarize automatically', cap_auto_sum_note:'Starts as soon as the transcript is ready. Otherwise use Summarize on the meeting page.', sum_btn:'Summarize', sum_btn_again:'Try again', sum_running:'Summarizing…', sum_queued:'Waiting to summarize…', sum_combining:'merging parts', sum_none:'No summary yet. It is made on the device and can take several minutes for a long meeting.', sum_failed:'The summary failed', st_summarizing:'summarizing' },
    tr: { tagline:'Cihazda toplantı notları', nav_meetings:'Toplantılar', nav_new:'Yeni kayıt', device_status:'Çevrimdışı · cihazda', meetings_title:'Toplantılar', lib_search:'Toplantı ara', speakers_word:'konuşmacı', st_done:'Hazır', st_processing:'İşleniyor', st_queued:'Sırada', st_error:'Hata', new_title:'Yeni kayıt', cap_record:'Kaydet', cap_upload:'Yükle', cap_rec_hint:'Kaydı başlatmak için dokunun — ses bu cihazdan çıkmaz.', cap_recording:'Kaydediliyor…', cap_stop:'Durdur', cap_rec_again:'Tekrar kaydet', cap_drop:'Ses dosyasını buraya bırakın', cap_browse:'veya dosya seçin', cap_formats:'WAV, MP3, M4A · en fazla 2 saat', cap_lang:'Yazıya dökme dili', cap_lang_note:'Dökümün dilini belirler — yukarıdaki arayüz dilinden bağımsızdır.', cap_process:'Cihazda işle', cap_captured:'Kayıt alındı', proc_note:'Yazıya dökme ve ayrıştırma aynı anda çalışır — hiçbir veri cihazdan çıkmaz.', proc_done:'Notlar hazır', proc_done_sub:'Döküm cihazda oluşturuldu.', proc_view:'Sonuçları gör', stage_convert:'Dönüştürme', stage_convert_sub:'16 kHz mono’ya çevir', stage_vad:'Konuşma tespiti', stage_vad_sub:'Konuşulan yerleri bul', stage_pack:'Parçalama', stage_pack_sub:'Konuşmayı 30 sn’lik parçalara grupla', stage_stt:'Yazıya dökme', stage_stt_sub:'Konuşmadan metne', stage_diar:'Ayrıştırma', stage_diar_sub:'Kim ne zaman konuştu', stage_merge:'Birleştirme', stage_merge_sub:'Konuşmacı ve kelimeleri hizala', stage_summary:'Özetleme', stage_summary_sub:'Yerel LLM özeti', tab_transcript:'Döküm', tab_summary:'Özet', sp_panel_title:'Konuşmacılar', sp_panel_hint:'Her sesi adlandırın — etiketler her yerde güncellenir.', sp_name_ph:'İsim ekle', sp_play:'Örnek çal', sp_playing:'Çalıyor…', sp_color:'Renk', turns_word:'konuşma', sum_overview:'Genel bakış', sum_key:'Önemli noktalar', sum_decisions:'Kararlar', sum_actions:'Aksiyon maddeleri', export_transcript:'transcript.md', export_summary:'summary.md', back_word:'Toplantılar', speaker_word:'Konuşmacı', theme_word:'Tema', sp_close:'Kapat', rec_title:'Kayıt', uploading:'Yükleniyor…', st_recording:'Kaydediliyor', mic_error:'Pi’nin mikrofonu başlatılamadı', cap_auto_sum:'Otomatik özetle', cap_auto_sum_note:'Döküm hazır olur olmaz başlar. Kapalıysa toplantı sayfasındaki Özetle düğmesini kullanın.', sum_btn:'Özetle', sum_btn_again:'Tekrar dene', sum_running:'Özetleniyor…', sum_queued:'Özet sırada…', sum_combining:'parçalar birleştiriliyor', sum_none:'Henüz özet yok. Özet cihazda oluşturulur; uzun bir toplantıda birkaç dakika sürebilir.', sum_failed:'Özet oluşturulamadı', st_summarizing:'özetleniyor' }
  };

  // the pipeline stages up to the transcript, in the order pipeline/run.py runs them (the summary is its own job)
  STAGES = ['convert','vad','pack','stt','diar','merge'];

  rootRef = React.createRef();
  state = {
    direction: this.props.direction || 'studio',
    mode: this.props.mode || 'light',
    lang: this.props.lang || 'en',
    screen: 'library', tab: 'transcript', query: '',
    recording: false, recTime: 0, recBars: new Array(44).fill(7), recId: null, recError: '',
    captureMode: 'record', meetingLang: 'auto', autoSummary: this.savedAutoSummary(),
    file: null, fileTitle: '', uploadError: '',
    meetings: [], openId: null, detail: null,
    playing: null, playProg: 0,
    speakersOpen: true,
    narrow: false
  };

  componentDidMount() {
    this.applyTheme();
    this.onResize();
    window.addEventListener('resize', this.onResize);
    window.addEventListener('dragover', this.onDragOver);
    window.addEventListener('drop', this.onDrop);
    this.refresh();
    this.pollTimer = setInterval(this.refresh, 1500);
  }
  componentDidUpdate(pp, ps) {
    if (ps.direction !== this.state.direction || ps.mode !== this.state.mode) this.applyTheme();
  }
  componentWillUnmount() {
    window.removeEventListener('resize', this.onResize);
    window.removeEventListener('dragover', this.onDragOver);
    window.removeEventListener('drop', this.onDrop);
    clearInterval(this.pollTimer);
    clearInterval(this.recTimer);
    if (this.audio) this.audio.pause();
  }

  onResize = () => {
    const n = window.innerWidth < 900;
    if (n !== this.state.narrow) this.setState({ narrow: n });
  };

  tokens() {
    const d = this.TOK[this.state.direction];
    const c = d[this.state.mode];
    return { ...c, radius: d.radius, fd: d.fd, fb: d.fb, fm: d.fm };
  }
  applyTheme() {
    const tk = this.tokens();
    const map = { '--bg':tk.bg,'--surface':tk.surface,'--surface2':tk.surface2,'--border':tk.border,'--text':tk.text,'--muted':tk.muted,'--faint':tk.faint,'--accent':tk.accent,'--accent-soft':tk.accentSoft,'--accent-text':tk.accentText,'--radius':tk.radius,'--shadow':tk.shadow,'--font-display':tk.fd,'--font-body':tk.fb,'--font-mono':tk.fm };
    const targets = [document.documentElement, this.rootRef && this.rootRef.current].filter(Boolean);
    for (const el of targets) for (const k in map) el.style.setProperty(k, map[k]);
    try { document.body.style.background = tk.bg; document.body.style.color = tk.text; } catch (e) {}
  }

  mix = (c, p) => `color-mix(in srgb, ${c} ${p}%, transparent)`;
  initials(name) {
    const parts = name.trim().split(/\s+/).filter(Boolean);
    if (!parts.length) return '?';
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  clock(sec) {
    sec = Math.floor(sec);
    const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = String(sec % 60).padStart(2, '0');
    return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
  }
  day(iso) { return new Date(iso).toLocaleDateString(this.state.lang, { month: 'short', day: 'numeric', year: 'numeric' }); }
  barsFor(id) {
    if (!this._bars) this._bars = {};
    if (this._bars[id]) return this._bars[id];
    let seed = 7; for (const ch of id) seed = (seed * 31 + ch.charCodeAt(0)) % 9973;
    const out = [];
    for (let i = 0; i < 26; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; const env = Math.sin((i / 25) * Math.PI); out.push(Math.round(22 + (seed % 60) * (0.45 + 0.55 * env))); }
    this._bars[id] = out; return out;
  }

  // ---- backend ----
  refresh = async () => {
    try {
      const meetings = await (await fetch('/api/meetings')).json();
      // a recording started from another browser, or before this page was reloaded
      const rec = await (await fetch('/api/record')).json();
      if (rec.id && !this.state.recording && !this.stopping) this.followRec();

      const S = this.state;
      const open = meetings.find(m => m.id === S.openId);
      // the open meeting's transcript is loaded when it is done, and again when its summary
      // changes state (e.g. running -> done brings the summary)
      let detail = null;
      if (open && open.status === 'done' && !(S.detail && S.detail.id === open.id && S.detail.summary_status === open.summary_status)) {
        detail = await (await fetch(`/api/meetings/${open.id}`)).json();
      }
      this.setState(s => {
        if (!(detail && detail.id === s.openId)) return { meetings };
        // a reload keeps the speaker names typed here, which may not be saved yet
        const keep = s.detail && s.detail.id === detail.id ? { speakers: s.detail.speakers } : {};
        return { meetings, detail: { ...detail, ...keep } };
      });
    } catch (e) {}   // backend unreachable: keep the last known state
  };

  // ---- navigation ----
  goMeetings = () => { this.stopAudio(); this.setState({ screen: 'library' }); };
  goNew = () => this.setState(s => s.recording ? { screen: 'capture' } : { screen: 'capture', recId: null, file: null, recTime: 0, recError: '', uploadError: '', recBars: new Array(44).fill(7) });
  setTabT = () => this.setState({ tab: 'transcript' });
  setTabS = () => this.setState({ tab: 'summary' });
  toggleSpeakers = () => this.setState(s => ({ speakersOpen: !s.speakersOpen }));
  openMeeting = (id) => {
    const m = this.state.meetings.find(x => x.id === id);
    this.stopAudio();
    this.setState({ openId: id, detail: null, tab: 'transcript', screen: m && m.status === 'done' ? 'review' : 'processing' });
    setTimeout(this.refresh);
  };
  goReviewResults = () => this.openMeeting(this.state.openId);

  // ---- appearance ----
  setDirection = (k) => this.setState({ direction: k });
  toggleTheme = () => this.setState(s => ({ mode: s.mode === 'light' ? 'dark' : 'light' }));
  setEn = () => this.setState({ lang: 'en' });
  setTr = () => this.setState({ lang: 'tr' });

  // ---- capture ----
  setRecordMode = () => this.setState({ captureMode: 'record' });
  setUploadMode = () => this.setState({ captureMode: 'upload' });
  onSearch = (e) => this.setState({ query: e.target.value });
  onMeetingLang = (e) => this.setState({ meetingLang: e.target.value });
  // "Summarize automatically" is remembered in this browser
  savedAutoSummary() { try { return localStorage.getItem('echoscript.autoSummary') === '1'; } catch (e) { return false; } }
  toggleAutoSummary = () => this.setState(s => {
    try { localStorage.setItem('echoscript.autoSummary', s.autoSummary ? '0' : '1'); } catch (e) {}
    return { autoSummary: !s.autoSummary };
  });

  browseUpload = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'audio/*,video/*';
    input.onchange = () => input.files[0] && this.pickFile(input.files[0]);
    input.click();
  };
  onDragOver = (e) => e.preventDefault();   // without it the browser opens a dropped file itself
  onDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && this.state.screen === 'capture' && !this.state.recording) this.pickFile(file);
  };
  pickFile = (file) => this.setState({ captureMode: 'upload', file, fileTitle: file.name.replace(/\.[^.]+$/, '') });

  // recording happens on the Pi (arecord on its mic); the page only starts, follows and stops it
  toggleRec = () => { this.state.recording ? this.stopRec() : this.startRec(); };
  startRec = async () => {
    if (this.starting) return;
    this.starting = true;
    const S = this.state, t = this.STR[S.lang];
    this.setState({ recError: '', recId: null, recTime: 0 });
    try {
      const res = await fetch('/api/record/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: `${t.rec_title} · ${new Date().toLocaleString(S.lang)}`, language: S.meetingLang, auto_summary: S.autoSummary })
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.status);
      this.followRec();
    } catch (e) {
      this.setState({ recError: String(e.message || e) });   // e.g. arecord: no such device
    } finally {
      this.starting = false;
    }
  };
  // poll the Pi's recorder for the timer and the level bars
  followRec() {
    this.setState({ recording: true });
    clearInterval(this.recTimer);
    this.recTimer = setInterval(async () => {
      try {
        const r = await (await fetch('/api/record')).json();
        if (!r.id) return this.endRec();   // stopped from another browser
        const h = Math.min(100, Math.round(9 + r.level * 400));
        this.setState(s => ({ recTime: Math.floor(r.seconds), recBars: [...s.recBars.slice(1), h] }));
      } catch (e) {}
    }, 150);
  }
  endRec() {
    clearInterval(this.recTimer);
    this.setState({ recording: false, recBars: new Array(44).fill(7) });
  }
  stopRec = async () => {
    this.stopping = true;
    this.endRec();
    try {
      // the Pi queues the meeting for processing at once; id null = under a second, discarded.
      // auto_summary again: it may have been switched while recording
      const { id } = await (await fetch('/api/record/stop', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ auto_summary: this.state.autoSummary })
      })).json();
      this.setState({ recId: id, captureMode: 'record' });
    } catch (e) {}
    this.stopping = false;
    setTimeout(this.refresh);
  };

  startProcessing = async () => {
    const S = this.state;
    if (S.captureMode === 'record') return this.openMeeting(S.recId);   // already queued by the stop
    if (!S.file || this.uploading) return;
    this.uploading = true;
    const form = new FormData();
    form.append('file', S.file);
    form.append('title', S.fileTitle);
    form.append('language', S.meetingLang);
    form.append('auto_summary', S.autoSummary ? 'true' : 'false');
    this.setState({ screen: 'processing', openId: null, detail: null, uploadError: '' });
    try {
      const res = await fetch('/api/meetings', { method: 'POST', body: form });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      const { id } = await res.json();
      this.setState({ openId: id, file: null });
      setTimeout(this.refresh);
    } catch (e) {
      this.setState({ uploadError: String(e.message || e) });
    } finally {
      this.uploading = false;
    }
  };

  // ---- summary ----
  startSummary = async () => {
    const d = this.state.detail;
    if (!d || this.summaryStarting) return;
    this.summaryStarting = true;
    try {
      const res = await fetch(`/api/meetings/${d.id}/summary`, { method: 'POST' });
      if (res.ok) {
        const m = await res.json();     // now "queued": the button shows it at once
        this.setState(s => ({
          meetings: s.meetings.map(x => x.id === m.id ? m : x),
          detail: s.detail && s.detail.id === m.id ? { ...s.detail, summary_status: m.summary_status, summary_progress: m.summary_progress, summary_error: m.summary_error } : s.detail
        }));
      }
    } catch (e) {}
    this.summaryStarting = false;
  };

  // ---- speakers ----
  editSpeaker(id, change) {
    const mid = this.state.detail.id;
    this.setState(s => ({ detail: { ...s.detail, speakers: { ...s.detail.speakers, [id]: { ...s.detail.speakers[id], ...change } } } }));
    clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => {   // save once typing pauses
      const d = this.state.detail;
      if (!d || d.id !== mid) return;
      fetch(`/api/meetings/${mid}/speakers`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(d.speakers) }).catch(() => {});
    }, 400);
  }
  rename = (id, name) => this.editSpeaker(id, { name });
  setColor = (id, color) => this.editSpeaker(id, { color });

  togglePlay = (id) => {
    const again = this.state.playing === id;
    this.stopAudio();
    if (again) return;
    const d = this.state.detail;
    // the speaker's longest utterance, at most 6 s: enough to recognise a voice
    const u = d.utterances.filter(x => x.speaker === id).reduce((a, b) => (b.end_ms - b.start_ms > a.end_ms - a.start_ms ? b : a));
    const audio = new Audio(`/api/meetings/${d.id}/audio?start_ms=${u.start_ms}&end_ms=${Math.min(u.end_ms, u.start_ms + 6000)}`);
    const mine = () => this.audio === audio;
    audio.ontimeupdate = () => mine() && this.setState({ playProg: audio.duration ? audio.currentTime / audio.duration : 0 });
    audio.onended = () => mine() && this.stopAudio();
    audio.play().catch(() => mine() && this.stopAudio());
    this.audio = audio;
    this.setState({ playing: id, playProg: 0 });
  };
  stopAudio() {
    if (this.audio) { this.audio.pause(); this.audio = null; }
    this.setState({ playing: null, playProg: 0 });
  }

  // the open meeting, its speaker ids in order, their display names, and text with SPK_n replaced by names
  current() {
    const S = this.state, t = this.STR[S.lang];
    const d = S.detail || { title: '', created: null, duration_s: null, speakers: {}, utterances: [], summary: { overview: '', key: [], decisions: [], actions: [] } };
    const ids = Object.keys(d.speakers);
    const name = (id) => (d.speakers[id] && d.speakers[id].name) || `${t.speaker_word} ${ids.indexOf(id) + 1}`;
    const named = (text) => text.replace(/SPK_\d+/g, (id) => ids.includes(id) ? name(id) : id);
    return { d, ids, name, named };
  }

  // ---- export ----
  download(name, text) {
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  exportTranscript = () => {
    const { d, name } = this.current();
    const lines = d.utterances.map(u => `[${this.clock(u.start_ms / 1000)}] ${name(u.speaker)}: ${u.text}`);
    this.download('transcript.md', [`# ${d.title} — Transcript`, '', ...lines].join('\n'));
  };
  exportSummary = () => {
    const { d, named } = this.current();
    const t = this.STR[this.state.lang], S = d.summary;
    const list = (items) => items.map(x => `- ${named(x)}`);
    const out = [`# ${d.title} — Summary`, '', `## ${t.sum_overview}`, named(S.overview), '', `## ${t.sum_key}`, ...list(S.key), '', `## ${t.sum_decisions}`, ...list(S.decisions), '', `## ${t.sum_actions}`, ...list(S.actions)];
    this.download('summary.md', out.join('\n'));
  };

  statusInfo(m, t, mode) {
    const map = {
      recording:  { label: t.st_recording,  color: mode === 'dark' ? '#ff7b7b' : '#d64545' },
      done:       { label: ['queued', 'running'].includes(m.summary_status) ? `${t.st_done} · ${t.st_summarizing}` : t.st_done,
                    color: mode === 'dark' ? '#6fd47a' : '#3a9142' },
      processing: { label: t['stage_' + m.stage] || t.st_processing, color: 'var(--accent)' },
      queued:     { label: t.st_queued,     color: 'var(--faint)' },
      error:      { label: t.st_error,      color: mode === 'dark' ? '#ff7b7b' : '#d64545' }
    };
    return map[m.status] || map.queued;
  }

  renderVals() {
    this.applyTheme();
    const S = this.state;
    // the record hint doubles as the place that says why recording didn't start
    const t = S.recError ? { ...this.STR[S.lang], cap_rec_hint: `${this.STR[S.lang].mic_error}: ${S.recError}` } : this.STR[S.lang];
    const pal = this.SPCOL[S.mode];
    const seg = (a) => a ? { bg:'var(--accent)', fg:'var(--accent-text)', fg2:'var(--accent)' } : { bg:'transparent', fg:'var(--muted)', fg2:'var(--muted)' };
    const metaLine = (m, n) => `${this.day(m.created)} · ${m.duration_s == null ? '—' : this.clock(m.duration_s)} · ${n} ${t.speakers_word}`;
    const colorIdx = (sp, i) => (sp.color ?? i) % pal.length;

    // ---- review ----
    const { d, ids, name, named } = this.current();
    const turns = {};
    for (const u of d.utterances) turns[u.speaker] = (turns[u.speaker] || 0) + 1;
    const speakers = ids.map((id, i) => {
      const ci = colorIdx(d.speakers[id], i);
      const dot = pal[ci];
      const displayName = name(id);
      const isPlaying = S.playing === id;
      const raw = this.barsFor(id);
      const bars = raw.map((h, bi) => ({ h, color: (isPlaying && bi / raw.length <= S.playProg) ? dot : 'var(--border)' }));
      const swatches = pal.map((c, idx) => ({ dot: c, ring: 'var(--surface)', outline: idx === ci ? 'var(--text)' : 'transparent', onPick: () => this.setColor(id, idx) }));
      return { id, dot, soft: this.mix(dot, 16), name: d.speakers[id].name, displayName, placeholder: `${t.speaker_word} ${i + 1}`, initials: this.initials(displayName), turnsLabel: `${turns[id] || 0} ${t.turns_word}`, isPlaying, notPlaying: !isPlaying, bars, swatches, playLabel: isPlaying ? t.sp_playing : t.sp_play, onName: (e) => this.rename(id, e.target.value), onPlay: () => this.togglePlay(id) };
    });
    const byId = {}; speakers.forEach(s => byId[s.id] = s);
    const nobody = { displayName: '', dot: 'var(--faint)', soft: this.mix('var(--faint)', 16), initials: '·' };

    const utterances = d.utterances.map((u, i) => { const s = byId[u.speaker] || nobody; return { id: 'u' + i, ts: this.clock(u.start_ms / 1000), text: u.text, name: s.displayName, dot: s.dot, soft: s.soft, initials: s.initials }; });
    const sumActions = d.summary.actions.map(text => {
      const s = byId[(text.match(/SPK_\d+/) || [])[0]] || nobody;   // the first speaker the action mentions
      return { text: named(text), name: s.displayName, dot: s.dot, soft: s.soft, initials: s.initials };
    });

    // ---- summary: its own job after the transcript ----
    // state and progress from the polled list (live); the summary text itself comes with the detail
    const live = S.meetings.find(m => m.id === d.id) || d;
    const sumState = live.summary_status || null;           // null (not asked for) | queued | running | done | error
    const sumBusy = sumState === 'queued' || sumState === 'running';
    const hasSummary = sumState === 'done' && d.summary_status === 'done';
    const part = live.summary_progress === 'combining' ? t.sum_combining : (live.summary_progress || '');
    const sumBtnLabel = sumState === 'running' ? `${t.sum_running}${part ? ' ' + part : ''}`
      : sumState === 'queued' ? t.sum_queued : sumState === 'error' ? t.sum_btn_again : t.sum_btn;
    const sumMsg = sumState === 'error' ? `${t.sum_failed}: ${live.summary_error || ''}` : sumBusy ? sumBtnLabel
      : sumState === 'done' ? '' : t.sum_none;      // done, but the text arrives with the next poll
    const sumSw = S.autoSummary ? { bg: 'var(--accent)', knob: '21px' } : { bg: 'var(--border)', knob: '3px' };

    // ---- library ----
    const q = S.query.trim().toLocaleLowerCase(S.lang);
    const meetings = S.meetings.filter(m => m.title.toLocaleLowerCase(S.lang).includes(q)).map(m => {
      const st = this.statusInfo(m, t, S.mode);
      const sp = Object.values(m.speakers);
      return { id: m.id, title: m.title, metaLine: metaLine(m, sp.length), stLabel: st.label, stColor: st.color, stSoft: this.mix(st.color, 15), cursor: 'pointer', dots: sp.map((x, i) => pal[colorIdx(x, i)]), onOpen: () => this.openMeeting(m.id) };
    });

    const directions = ['studio','signal','aurora'].map(k => ({ key: k, label: this.TOK[k].label, dot: this.TOK[k][S.mode].accent, ring: k === S.direction ? 'var(--text)' : 'transparent', onPick: () => this.setDirection(k) }));

    // ---- processing ----
    const job = S.meetings.find(m => m.id === S.openId);
    // no stage yet (queued, recording, between live STT and diarization): the stages before
    // first_stage are already done, e.g. convert..stt after a live recording
    const at = !job ? -1 : job.status === 'done' ? this.STAGES.length
      : this.STAGES.indexOf(job.stage || job.first_stage);
    const side = job && job.status === 'processing' && job.side;   // a stage running next to job.stage: diar next to stt
    const stages = this.STAGES.map((k, idx) => {
      let isDone = idx < at, isActive = !!job && !!job.stage && idx === at;
      if (side && k === side.stage) { isDone = side.done; isActive = !side.done; }
      const isIdle = !isDone && !isActive;
      const progress = k === job?.stage ? job.progress : null;   // only the main stage reports progress
      const fill = isDone ? '100%' : isActive ? `${Math.round((progress ?? 0.5) * 100)}%` : '0%';
      return { key: k, label: t['stage_' + k], sub: t['stage_' + k + '_sub'], isDone, isActive, isIdle, fill,
        rowBg: isActive ? 'var(--surface)' : 'transparent', rowBorder: isActive ? 'var(--border)' : 'transparent',
        iconBg: isDone ? this.mix('#3a9142', 16) : (isActive ? 'var(--accent)' : 'var(--surface2)'),
        iconFg: isDone ? (S.mode === 'dark' ? '#6fd47a' : '#3a9142') : (isActive ? 'var(--accent-text)' : 'var(--faint)'),
        labelColor: isIdle ? 'var(--muted)' : 'var(--text)' };
    });
    const procLogs = S.uploadError ? [`${t.st_error}: ${S.uploadError}`]
      : !job ? [t.uploading]
      : [...(job.status === 'recording' ? [t.cap_recording] : job.status === 'queued' ? [t.st_queued] : []), ...job.log, ...(job.error ? [`${t.st_error}: ${job.error}`] : [])];

    return {
      t, isDark: S.mode === 'dark', isLight: S.mode === 'light',
      showSidebar: !S.narrow, showBottomNav: S.narrow,
      isLibrary: S.screen === 'library', isCapture: S.screen === 'capture', isProcessing: S.screen === 'processing', isReview: S.screen === 'review',
      isTabT: S.tab === 'transcript', isTabS: S.tab === 'summary',
      speakersOpen: S.speakersOpen, spToggle: S.speakersOpen ? { bg:'var(--accent)', fg:'var(--accent-text)', bd:'var(--accent)' } : { bg:'var(--surface)', fg:'var(--text)', bd:'var(--border)' },
      contentPadRight: (S.speakersOpen && !S.narrow) ? '312px' : '0px',
      isRecord: S.captureMode === 'record', isUpload: S.captureMode === 'upload',
      recording: S.recording, notRecording: !S.recording, captured: S.captureMode === 'record' ? !!S.recId && !S.recording : !!S.file, procDone: !!job && job.status === 'done',
      query: S.query, meetingLang: S.meetingLang,
      recTimeStr: this.clock(S.recTime), recBars: S.recBars.map(h => ({ h, color: S.recording ? 'var(--accent)' : 'var(--border)' })),
      capturedName: S.captureMode === 'record' ? `${t.cap_record} · ${this.clock(S.recTime)}` : (S.file ? S.file.name : ''),
      meetingTitle: d.title, meetingMeta: S.detail ? metaLine(d, ids.length) : '',
      langEn: seg(S.lang === 'en'), langTr: seg(S.lang === 'tr'),
      tabT: seg(S.tab === 'transcript'), tabS: seg(S.tab === 'summary'),
      capRec: seg(S.captureMode === 'record'), capUp: seg(S.captureMode === 'upload'),
      navMeet: seg(S.screen === 'library'), navNew: seg(S.screen === 'capture'),
      rootRef: this.rootRef,
      directions, meetings, speakers, utterances, stages, procLogs,
      sumOverview: named(d.summary.overview), sumKey: d.summary.key.map(named), sumDecisions: d.summary.decisions.map(named), sumActions,
      hasSummary, noSummary: !hasSummary, sumBusy, sumIdle: !sumBusy, sumBtnLabel, sumMsg, startSummary: this.startSummary,
      showSumBtn: sumState !== 'done', sumCanStart: sumState === null || sumState === 'error',
      sumBtn: sumBusy ? { bg: 'var(--surface2)', fg: 'var(--muted)', bd: 'var(--border)', cursor: 'default' } : { bg: 'var(--accent)', fg: 'var(--accent-text)', bd: 'var(--accent)', cursor: 'pointer' },
      autoSummary: S.autoSummary, autoSummaryAria: S.autoSummary ? 'true' : 'false', sumSw, toggleAutoSummary: this.toggleAutoSummary,
      goMeetings: this.goMeetings, goNew: this.goNew, setEn: this.setEn, setTr: this.setTr, toggleTheme: this.toggleTheme,
      setTabT: this.setTabT, setTabS: this.setTabS, toggleSpeakers: this.toggleSpeakers, setRecordMode: this.setRecordMode, setUploadMode: this.setUploadMode,
      onSearch: this.onSearch, onMeetingLang: this.onMeetingLang, browseUpload: this.browseUpload, toggleRec: this.toggleRec,
      startProcessing: this.startProcessing, goReviewResults: this.goReviewResults, exportTranscript: this.exportTranscript, exportSummary: this.exportSummary
    };
  }
}
