# Echoscript — Chunking, diarization, memory and a full run

What was measured on a Raspberry Pi 5 with the recordings in `recordings/` (October 2026), and why
the pipeline works the way it does.

- **Part 1 (§1–9):** how audio is cut into chunks for whisper.
- **Part 2 (§10–19):** diarization, who spoke when.
- **Part 3 (§20–21):** how much RAM it needs while it works, and one whole meeting through the web app.

# Part 1 — Chunking for transcription

## 1. Decision

| | Choice |
|---|---|
| Live recording | **E2**: chunks are built and transcribed *during* the recording (`LiveChunker`, `main.py`) |
| Uploads, and fallback if live STT fails | **A2**: `pack()` after the recording, with the same cut rule |
| Chunk rule | ≤ 30 s, cut at silences; a silence > 2 s always ends a chunk |
| Speech > 30 s without a pause ("forced" cut) | Cut at the quietest 100 ms of the last 5 s |
| Prompt | On: the last 200 characters of text go to the next chunk |

| Result (oscar, hand-corrected reference) | Old pipeline (A) | Chosen (E2 / A2) |
|---|---|---|
| CER / WER | 6.2% / 10.5% | **3.6% / 7.1%** |
| Words lost at forced cuts | 41 | **8** |
| Transcript ready after stop (sim, 22–31 min recordings) | 22–35 min (A) | **0.8–5.2 min** (E2) |

## 2. Constraint: every request costs ~25 s

whisper.cpp `whisper-server` runs large-v3-turbo q8_0 on cores 1–3. Its encoder always processes
a full 30 s window. Fitted on 188 chunks (`eval/bench_cost.py`):

**Time per request = 21.9 s + 0.19 × chunk seconds** (sd 2.8 s).

| Chunk | Time | RTF |
|---|---|---|
| 30 s | ~27.6 s | 0.93 |
| 5 s | ~22.9 s | 4.6 |

**Consequences:**
- Chunks must be as full as possible.
- Live transcription keeps up only because speech piles up into full chunks while whisper is busy.
- Encoding less than 30 s (`audio_ctx`) would be faster, but it broke the text (CER 23–97%).

## 3. Test setup

| Recording | Length | Section scored | Forced cuts in section |
|---|---|---|---|
| `htalks_gs_barcelona` | 22.4 min | 01:00–11:00 | none |
| `oscar_da_bizi_gorecek_mi` | 22.0 min | 08:30–18:30 | many |
| `balkan_asidik` | 30.5 min | 01:30–11:30 | many |

**References:**
- **R:** `whisper-cli` with the **f16** model, **beam 5**, using whisper's own long-form decoding
  (`eval/reference.py`). It shares none of our cuts, so it shows cut losses well. It is not ground
  truth: against the hand-corrected oscar reference, R itself has CER 5.0%.
- **Hand-corrected `ref.txt`:** only for oscar so far.

**Metrics** (`eval/wer.py`):
- WER and CER after Turkish normalization.
- **Cut CER:** errors on reference words within ±2 s of a chunk boundary.
- **Forced CER:** the same, at forced cuts only.
- **Deletions:** words a strategy lost at its own forced cuts.

## 4. Approaches and verdicts

| Id | How it works | Tested | Verdict and why |
|---|---|---|---|
| **A** | Offline `pack()`: ≤ 30 s, cut at silences. Speech > 30 s without a pause is split into equal parts | yes | **Replaced by A2.** Equal parts split words and whisper drops the chunk's end: forced CER 23–30% |
| **A2** | A, but speech > 30 s without a pause is cut at the quietest point of the last 5 s | yes | **Chosen for offline.** Forced CER 4.5–5.9%; as accurate as E2 |
| **B** | Fixed 30 s windows, cut anywhere | yes | **Rejected.** Cuts through words: cut CER 48–65% |
| **C** | 30 s windows every 25 s (5 s overlap); each window keeps its words up to the middle of the overlap | yes | **Rejected.** Still cuts through words (cut CER 39–47%), and the overlap costs extra requests |
| **D** | One request per VAD region (one utterance per request) | cost only | **Rejected.** Every request costs ~22 s even for a 3 s region: htalks would take 128 min for 22 min of audio (RTF 5.7; A2: 1.0). Each request also gets the least context |
| **E** | Live `LiveChunker`. When whisper is free, it sends the speech gathered so far. Forced cut at exactly 30 s | yes | **Replaced by E2.** The hard cut splits words: forced CER 48–50%, 60–68 words lost |
| **E2** | E, forced cut at the quietest point of the last 5 s | yes | **Chosen for live.** Forced CER 4.4–5.9%, 1–8 words lost |
| **E3** | E, cut at 30 s, but the next chunk starts again 3–6 s earlier (overlap); words up to the next chunk's start come from the first chunk | yes | **Rejected.** Word times are 0.2–0.5 s off, so a word at the seam is lost ("Her çektiği film" → "Her") or doubled ("O O"). Forced CER 19–31%, and more delay |
| **E′** | Re-decode a growing buffer; commit words once two passes agree (LocalAgreement, as in whisper-streaming) | no | **Rejected.** Needs several ~25 s passes per chunk; the Pi cannot keep up |
| `_noprompt` | Any of the above without the previous text as prompt | A, E2, A2 | **Rejected.** Never better. On oscar it adds 22 words that were never spoken (fillers like "yani"), checked by ear |
| `min_send` | E: wait until a chunk has N s before sending at a pause | sim | **Set to 0.** 0 / 15 / 25 s give 5.2 / 5.3 / 5.1 min after stop on balkan |

### 4.1 A vs E: offline vs live

A and E use the same rules. What differs is **when** the cut is decided.

| | A / A2 (offline) | E / E2 (live) |
|---|---|---|
| Decides chunks | After stop, with all speech regions known | During the recording, whenever whisper is free, from the audio so far |
| Speech > 30 s without a pause | Knows its full length (A: equal parts) | Cut while it is still going (E: at 30 s) |
| Chunk fill | Always as full as possible | Full while whisper is busy. If whisper is free at a pause, a small piece goes at once (`early`) |
| Transcript ready | 22–35 min after stop | 0.8–5.2 min after stop |

### 4.2 Why E2, and A2 as well

The two are **equally accurate**:
- On oscar they cut at the same places and give identical text (3.6%).
- On balkan, CER is 7.1% for both; WER is 12.3% for A2 and 12.8% for E2.

**E2 is used for recordings because of time:** the text is ready minutes after the meeting instead
of half an hour. Its cost is small:
- a few extra small chunks (balkan: 81 vs 79);
- the text lags 1–5 min behind the speech during the meeting.

**A2 stays for:**
- uploads, where the file is already complete;
- the fallback, when live STT fails or the server restarts during transcription.

## 5. Results

### 5.1 Latency (simulation, `eval/live_sim.py`)

The simulation replays the VAD regions on a virtual clock with the cost model of §2.

| Recording | Offline A, after stop | E, after stop | **E2, after stop** | E3, after stop | E2 lag p50 / p90 during the meeting |
|---|---|---|---|---|---|
| htalks | 22.3 min | 0.8 min | **0.8 min** | 0.8 min | 0.8 / 1.0 min |
| oscar | 25.7 min | 2.1 min | **2.8 min** | 3.2 min | 2.0 / 3.0 min |
| balkan | 34.6 min | 3.7 min | **5.2 min** | 5.9 min | 3.1 / 4.8 min |

### 5.2 Accuracy

**oscar vs the hand-corrected reference** (1506 words)

| Strategy | CER | WER | Cut CER | Forced CER | Forced deletions |
|---|---|---|---|---|---|
| **A2 / E2** | **3.6%** | **7.1%** | **6.5%** | **5.9%** | 8 |
| A2 / E2 without prompt | 4.6% | 8.6% | 9.3% | 6.8% | 5 |
| R (f16, beam 5) | 5.0% | 9.2% | – | – | – |
| A | 6.2% | 10.5% | 20.0% | 30.0% | 41 |
| E3 | 7.1% | 11.0% | 24.7% | 31.0% | 44 |
| E | 8.2% | 10.6% | 28.3% | 47.9% | 68 |
| C | 13.5% | 17.5% | 46.6% | – | – |
| B | 14.5% | 19.6% | 48.2% | – | – |

**balkan vs R**

| Strategy | CER | WER | Cut CER | Forced CER | Forced deletions |
|---|---|---|---|---|---|
| **A2** | **7.1%** | **12.3%** | **10.7%** | 4.5% | 1 |
| **E2** | **7.1%** | 12.8% | 12.8% | **4.4%** | 1 |
| E2 without prompt | 7.4% | 12.6% | 16.2% | 10.8% | 9 |
| A | 7.5% | 13.2% | 20.3% | 23.6% | 16 |
| E3 | 8.4% | 14.0% | 18.7% | 19.4% | 13 |
| B | 10.4% | 15.7% | 52.3% | – | – |
| E | 11.8% | 17.6% | 28.8% | 48.7% | 60 |
| C | 14.5% | 17.9% | 39.0% | – | – |

**htalks vs R** (no forced cuts: here E = E2)

| Strategy | CER | WER | Cut CER |
|---|---|---|---|
| A | 8.6% | 18.1% | 14.2% |
| E / E2 | 9.0% | 18.0% | 9.8% |
| E2 without prompt | 9.0% | 18.6% | 12.7% |
| B / C | 21.6–21.8% | 28.2–29.2% | 40.2–65.0% |

### 5.3 End-to-end live test (`main.py`)

arecord was replaced by a stream of oscar 08:30–12:30, played in real time.

| Chunks | Lag while recording | Ready after stop | CER / WER | Words lost at forced cuts |
|---|---|---|---|---|
| 10 (7 forced) | ~50 s | **58 s** | **3.5% / 6.4%** | 2 |

## 6. Observations

**1. Whisper drops the end of a chunk that is cut mid-speech.** This was the main finding. E's lost
words, by distance before the cut: 0–1 s: 23, 1–2 s: 15, 2–3 s: 9. A had the same problem at its own
forced cuts. Cutting at a breath removes almost all of it.

**2. Pitfalls when making a reference by hand:**

| Mistake | Effect | Rule |
|---|---|---|
| Editing a tested system's output (oscar's first `ref.txt` came from E) | E scored 3.5% instead of 11% | Start from R's draft. `check.txt` lists only the spots where systems disagree |
| Leaving out fillers and repeats | Penalized the runs that write them; nearly reversed the prompt decision | Write everything that is said |
| Numbers as digits | "4" vs "dört" counted as an error | Spell numbers out |
| Covering more than the section | Extra words counted as deletions | Match the section, or clip when scoring |

**3. A tiny `early` chunk costs a full request.** In the live test a 1.5 s chunk took 23 s and raised
the lag from 29 s to 50 s. Over whole recordings this made no difference (`min_send`, §4).

**4. The whisper-server is shared.** If an upload is processed during a recording, the live text
falls further behind, but stays correct.

## 7. Implementation

| File | What |
|---|---|
| `pipeline/stt.py` | `quiet_point()`. `pack(regions, config, wav)` cuts long speech at quiet points. `transcribe_chunk()` is shared by offline and live |
| `pipeline/live.py` | `LiveChunker(config, min_send_ms, cut, wav)` with cuts `"hard"` (E), `"quiet"` (E2), `"overlap"` (E3) |
| `pipeline/vad.py` | `StreamingVAD.open_start_ms`: start of the speech going on now |
| `pipeline/run.py` | `stage_pack` passes the wav (A2) |
| `main.py` | `LIVE_CUT = "quiet"`. `capture()` feeds VAD regions to the chunker. `live_stt()` transcribes while recording. `finish_live()` sends the rest after stop, writes `words.json` / `chunks.json`, and queues the meeting from `diar` |

```
start ─► arecord ─► capture() ─► audio.wav
                         └─► StreamingVAD ─► LiveChunker
         live_stt(): whenever whisper is free ─► next_chunk ─► transcribe_chunk(prompt) ─► words
stop  ─► vad.json ─► finish_live(): last chunks ─► words.json ─► pipeline from "diar"
         (live STT error or server restart ─► pipeline from "pack" = A2)
```

## 8. Reproduce

```bash
python -m recordings.record --name my_meeting                     # record (Ctrl+C stops)
python -m eval.live_sim recordings/*/ --min-send 0 --cut hard quiet overlap     # latency, instant
python -m eval.reference recordings/oscar_da_bizi_gorecek_mi --from 510 --to 1110          # R + check.txt
python -m eval.bench_chunking recordings/oscar_da_bizi_gorecek_mi --from 510 --to 1110 --only E2 A2
```

Logs: `recordings/*.log`.

## 9. Open items

| Item | Why |
|---|---|
| Hand-correct `ref.txt` for balkan and htalks | The decision rests on one hand-corrected recording |
| Test with the real mic in the app | Diar, merge and summary after a live recording have not run end to end |

# Part 2 — Diarization

How Echoscript decides who spoke when, what was measured, and why. **Provisional:** there are no
hand-labelled speaker references yet, so nothing here is a DER. The settings were judged by
plausibility, by agreement between independent settings, and by cost (§13).

## 10. Decision

| | Choice |
|---|---|
| Voiceprint model | **CAM++** (`3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`, 512-d) |
| Windows | 1.5 s every 0.75 s, mean removed per window (CMN) |
| Clustering | **Spectral (NME-SC)**, `p_ratio` 0.25: finds the speaker count itself, no threshold |
| Speaker count | Never given (`num_speakers = 0`) |
| Merge with the words | Word times capped at 1.5 s (`MAX_WORD_MS`) to decide speakers: Whisper folds pauses into the last word of a sentence (§15.6) |
| Unchanged, not tested yet | `min_cluster_ms` 5000, `MIN_TURN_MS` 500, `MERGE_GAP_MS` 500, merge's `SNAP_MS` 2000 |

| Result (htalks / oscar / balkan, 3 speakers each) | Old (ResNet34, AHC 0.65) | Chosen (CAM++, spectral) |
|---|---|---|
| Speakers found | 23 / 9 / 69 | **3 / 3 / 3** |
| Speech given to tiny "speakers" (< 5%) | 23% / 5% / 46% | **0% / 0% / 0%** |
| Disagreement with ResNet34 + spectral | 24% / 6% / 57% | **2.8% / 1.8% / 5.8%** |
| Diarization after stop, 30 min meeting (core 0) | ~8.3 min | **~2.6 min** |

## 11. How it works

```
VAD regions → 1.5 s windows → CAM++ voiceprints → spectral clustering → small clusters absorbed
            → turns → merge with Whisper's words → utterances ("[0:02:32] SPK_2: …")
```

| Step | What happens | Constants |
|---|---|---|
| 1. Windows | Every VAD speech region is cut into 1.5 s windows every 0.75 s (50% overlap); the last one ends at the region end. A region up to 1.5 s is one window. A region under 0.5 s gets none: its words take the nearest speaker in step 6 | `WINDOW_MS` 1500, `STEP_MS` 750, `MIN_WINDOW_MS` 500 |
| 2. Voiceprints | 80 mel filterbank features per region (10 ms frames), the mean of each window removed (CMN: takes out the mic and the room), then CAM++ gives a 512-d vector per window, L2-normalised. Two windows of the same voice point the same way: their cosine similarity is high | ONNX, 1 thread, core 0 |
| 3. Spectral clustering | A graph of who sounds like whom: each window keeps its *p* most similar windows as neighbours. With *k* speakers the graph has *k* groups that barely connect, which shows as *k* eigenvalues of its Laplacian near 0 and then a jump (the eigengap). *p* is tried from 2 up to 25% of the windows; the clearest jump for the sparsest graph wins (NME-SC, Park et al. 2019), and its position is the speaker count. k-means++ on the *k* smallest eigenvectors then gives every window a speaker | `p_ratio` 0.25, ≤ 8 speakers, 10 k-means starts |
| … large meetings | eigh is O(n³): above 1000 windows (~13 min of speech) an even subsample over time is clustered, and every window takes the speaker of the nearest centroid | `SC_MAX_WINDOWS` 1000 |
| … short meetings | Below 20 windows (~15 s of speech) the graph is too small: AHC with a cosine threshold instead | `SC_MIN_WINDOWS` 20, `threshold` 0.65 |
| 4. Small clusters | A cluster with under 5 s of windows is usually one person split off (a laugh, a cough): its windows go to the most similar big cluster. Speakers are numbered in order of first appearance: SPK_0 speaks first | `min_cluster_ms` 5000 |
| 5. Turns | Each window owns the time nearest its centre, so a change falls midway between two window centres, on a 0.75 s grid. The same speaker across a gap ≤ 0.5 s is one turn; a turn under 0.5 s glued to a neighbour takes its speaker; a short turn alone between silences ("Evet.") stays | `MERGE_GAP_MS` 500, `MIN_TURN_MS` 500 |
| 6. Merge (`pipeline/merge.py`) | Every word gets the speaker of the turn it overlaps most, with its time capped at 1.5 s (§15.6), so a change never falls inside a word. Each change then moves to the nearest break within 2 s, a pause ≥ 150 ms or after `. ? ! …`, so it falls between sentences. Words become utterances at every change and every pause > 2 s | `MAX_WORD_MS` 1500, `SNAP_MS` 2000, `MIN_PAUSE_MS` 150, `MAX_PAUSE_MS` 2000 |

**Why this way** (the measurements are in §14–16):
- **Spectral, not a threshold.** The speaker count comes from the shape of the graph, not from a
  distance limit. AHC's threshold had no value that worked on all three 20–30 min recordings:
  a little lower and one voice broke into many, a little higher and two people merged (§15.1).
  Spectral gave 3/3/3 across models, windows and `p_ratio` ≥ 0.25 (§15.2–15.3).
- **CAM++, not ResNet34.** Both give the same speakers to 94–98% of the speech (§15.4), so the
  choice is cost: CAM++ is 3.2× faster, ~2.6 instead of ~8.3 min after stop for 30 min (§15.3).
- **1.5 s windows with CMN.** Shorter windows are noisier, longer ones cost more without a gain,
  and without CMN every voice looks alike (§14).
- **Merge by words, then snapped to a break.** Diarization alone places a change only to ±0.75 s;
  the words and their punctuation say where a sentence really ends.

## 12. Constraint: diarization runs after stop

A live recording is transcribed during the meeting (E2, §1). Diarization only starts after stop:
`finish_live()` queues the meeting from `diar`. Its time is added in full to "transcript ready".

**Time after stop ≈ RTF × recording length.** The voiceprints are almost all of it. Clustering is
under 0.1 s (AHC) or ~10 s (spectral) for 30 min.

## 13. Test setup

| Recording | Length | Speakers | Used for |
|---|---|---|---|
| `htalks_gs_barcelona` | 22.4 min | 3 | count check, agreement |
| `oscar_da_bizi_gorecek_mi` | 22.0 min | 3 | count check, agreement |
| `balkan_asidik` | 30.5 min | 3 | count check, agreement |
| `habersizler_168` | 9.9 min | not known yet | cost; more than 3 speakers? |

- The whole recording is diarized, as in the app. The speaker count of the videos is only used to
  check results. It never goes into the system.
- **No reference labels yet.** `recordings/*/ref.labels.txt` are untouched drafts (§18).

**Measures** (`eval/bench_diar.py`, `eval/diar_agree.py`):
- **Speakers found**, and **major speakers**: those with ≥ 5% of the speech.
- **Tiny-speaker share:** the speech given to speakers under 5%. It shows one voice broken into pieces.
- **Disagreement:** the DER of one setting against another (collar 0, speakers mapped one-to-one),
  over the whole recording: the share of speech the two give to different people. Two different
  networks that agree are probably both right. Agreement is not proof: they can share a mistake.
- **Cost:** embedding from scratch, alone on core 0, one BLAS thread, as in the app
  (`eval/diar_cost.py`, on habersizler).

**Grid:** 14 embedding settings × 117 clustering settings × 4 recordings ≈ 6550 diarizations. The
voiceprints were computed once per embedding setting (`runs/diar/*.npz`), then clustered many times.

## 14. Approaches and verdicts

| Area | Tested | Verdict and why |
|---|---|---|
| **AHC average, threshold 0.65** (old) | yes | **Replaced.** Tuned on 1.5 min clips. On 20–30 min one voice drifts and breaks into many clusters: 23 / 9 / 69 speakers |
| AHC average, threshold 0.5–1.0 | yes | **Rejected.** 0.6–0.8 finds the 3 main voices but always leaves small clusters (balkan 13 at 0.8). 0.05 higher, real speakers merge: 1–2 left. A cliff, no plateau |
| AHC average + `min_cluster` 30 s | yes | **Rejected.** Reaches 3/3/3, but only within 0.05–0.15 of threshold, and that range moves with the model and window (ResNet34 0.75–0.8, CAM++ 0.7–0.75, CAM++ 3 s 0.5–0.6) |
| AHC complete linkage, 0.9–1.6 | yes | **Rejected.** No threshold gives 3/3/3. Best (1.2): 4 / 3 / 5 |
| **Spectral (NME-SC)** | yes | **Chosen.** 3/3/3 on 11 of the 12 embeddings with CMN, no threshold. Agrees with itself across models (§15.4) |
| Spectral `p_ratio` 0.05–0.15 | yes | **Rejected.** A 4th small cluster on balkan. 0.25 and 0.35 are the same: **0.25 kept** |
| **CAM++** vs ResNet34 | yes | **CAM++ chosen.** Same speaker counts, 2–6% disagreement, **3.2× faster** (RTF 0.087 vs 0.278) |
| Window 1.0 s | yes | **Rejected.** Noisier: more turns per minute, drifts away from the others (6–11% disagreement), habersizler collapses to 2 speakers with CAM++ |
| Window 2.0 / 3.0 s | yes | **Rejected.** No gain; slower. ResNet34 3.0 s finds 5 speakers on balkan |
| Step 0.5 s | yes | **Rejected.** Same counts, +42% cost |
| CMN per region | yes | **Equivalent** to per window. Per window kept |
| **No CMN** | yes | **Never.** Collapses to 1–2 speakers (1 / 2 / 1 with ResNet34): without it the channel and room dominate the voiceprint |
| AHC ward, VBx resegmentation, overlap detection, live voiceprints | no | Not tested (§19) |

## 15. Results

### 15.1 AHC: speakers found vs threshold

1.5/0.75 s, `min_cluster` 5 s. Cells: speakers found (major speakers, if fewer).

| Threshold | ResNet34 htalks / oscar / balkan | CAM++ htalks / oscar / balkan |
|---|---|---|
| 0.60 | 34 (3) / 22 (3) / 84 (3) | 36 (3) / 21 (3) / 69 (3) |
| 0.65 | 23 (3) / 9 (3) / 69 (4) | 15 (3) / 8 (3) / 60 (4) |
| 0.70 | 16 (3) / 8 (3) / 45 (3) | 15 (3) / 7 (3) / 46 (3) |
| 0.75 | 9 (3) / 8 (3) / 27 (3) | 8 (3) / 7 (3) / 25 (3) |
| 0.80 | 4 (3) / 3 / 13 (3) | 6 (2) / 6 (3) / 9 (3) |
| 0.85 | 2 / 4 (3) / 7 (3) | 3 (1) / 3 (2) / 6 (2) |
| 0.90 | 2 (1) / 2 / 2 (1) | 1 / 2 / 2 (1) |

### 15.2 Methods side by side (1.5/0.75 s, CMN per window)

| Setting | htalks | oscar | balkan | habersizler | Tiny-speaker share, balkan |
|---|---|---|---|---|---|
| ResNet34 AHC 0.65, `min_cluster` 5 s (old) | 23 | 9 | 69 | 16 | 46% |
| ResNet34 AHC 0.80, `min_cluster` 5 s | 4 | 3 | 13 | 10 | 7.4% |
| ResNet34 AHC 0.80, `min_cluster` 30 s | 3 | 3 | 3 | 5 | 0% |
| CAM++ AHC 0.75, `min_cluster` 30 s | 3 | 3 | 3 | 4 | 0% |
| ResNet34 AHC complete 1.2 | 4 | 3 | 5 | 3 | – |
| ResNet34 spectral 0.25 | 3 | 3 | 3 | 6 (5) | 0% |
| **CAM++ spectral 0.25** | **3** | **3** | **3** | **6 (5)** | **0%** |

### 15.3 Embedding settings (spectral 0.25) and cost

Cost: habersizler embedded from scratch alone on core 0. "30 min" = RTF × 30.

| Embedding | htalks / oscar / balkan / habersizler | RTF | After stop, 30 min |
|---|---|---|---|
| ResNet34 1.0 / 0.75 s | 3 / 3 / 3 / 7 (6) | 0.246 | 7.4 min |
| ResNet34 1.5 / 0.75 s | 3 / 3 / 3 / 6 (5) | 0.278 | 8.3 min |
| ResNet34 2.0 / 0.75 s | 3 / 3 / 3 / 6 (5) | 0.352 | 10.6 min |
| ResNet34 3.0 / 0.75 s | 3 / 3 / 5 (4) / 6 (5) | 0.433 | 13.0 min |
| ResNet34 1.5 / 0.5 s | 3 / 3 / 3 / 6 (5) | 0.395 | 11.9 min |
| CAM++ 1.0 / 0.75 s | 3 / 3 / 3 / 2 | 0.065 | 2.0 min |
| **CAM++ 1.5 / 0.75 s** | **3 / 3 / 3 / 6 (5)** | **0.087** | **2.6 min** |
| CAM++ 2.0 / 0.75 s | 3 / 3 / 3 / 5 | 0.109 | 3.3 min |
| CAM++ 3.0 / 0.75 s | 3 / 3 / 3 / 6 (5) | 0.142 | 4.3 min |
| CAM++ 1.5 / 0.5 s | 3 / 3 / 3 / 6 (5) | 0.123 | 3.7 min |
| CMN per region (either model) | 3 / 3 / 3 / 5–6 | same | same |
| No CMN | 1 / 2 / 1 / 2 (ResNet34), 1 / 4 / 1 / 4 (CAM++) | same | – |

Spectral clustering itself: ~10 s for 30 min (1000 windows at most, §17), 2.4 s for habersizler.

### 15.4 Agreement

Share of speech two settings give to different people, whole recording (1.5/0.75 s unless noted).

| Pair | htalks | oscar | balkan | habersizler |
|---|---|---|---|---|
| **ResNet34 spectral vs CAM++ spectral** | **2.8%** | **1.8%** | **5.8%** | 10.2% |
| ResNet34 spectral vs ResNet34 AHC 0.8 | 4.3% | 1.4% | 10.9% | 12.8% |
| CAM++ spectral 1.5 s vs 1.0 s | 6.4% | 3.5% | 11.2% | 49.6% |
| ResNet34 spectral vs CAM++ AHC 0.75 | 7.7% | 4.3% | 18.6% | 25.1% |
| ResNet34 spectral vs old (AHC 0.65) | 24.3% | 5.5% | 57.2% | 26.8% |

### 15.5 End-to-end: STT and diarization in parallel (`pipeline.run --parallel`)

All of oscar (22.0 min) as an upload: convert → VAD → pack (A2) → STT ∥ diarization → merge.
whisper-server (q8_0) on cores 1–3, the pipeline on core 0 (`taskset -c 0`), as in `main.py`.

| Stage | Where | Time | RTF |
|---|---|---|---|
| convert + VAD + pack | core 0 | 13 s | 0.01 |
| STT, 55 chunks | cores 1–3 | 1415 s | 1.07 |
| Diarization, CAM++ + spectral (in parallel) | core 0 | 208 s | 0.158 |
| Merge waited for diarization | – | **0 s** | – |
| **Total** | | **23 min 49 s** | **1.08** |

| Effect of running both | Measured |
|---|---|
| STT chunks while diarization runs (7) | +15% vs the cost model of §2 (213 s vs 185 s) |
| STT chunks after it (48) | −5% (1202 s vs 1269 s) |
| Diarization next to whisper | RTF 0.158 vs 0.087 alone on core 0: 1.8× slower, still hidden behind STT |

The cores are separate; the likely cause is shared memory bandwidth. Never throttled (peak ~65 °C).

| Result | |
|---|---|
| STT, 08:30–18:30, vs the hand-corrected reference | **CER 3.3% / WER 6.6%** (A2 in Part 1, rescored the same way: 3.6% / 7.2%). The 27 chunks there are identical to A2's |
| Diarization | 3 speakers (245 / 389 / 678 s of speech, 103 utterances). Same turns as the benchmark's CAM++ + spectral (0.0% disagreement); 1.8% from ResNet34 + spectral |
| Errors seen in the transcript (by text) | The speakers are right, but a change was often one word early: the last word of a sentence went to the next speaker ("…kulis nedir \| hocam? Aslında…"). Cause and fix: §15.6 |

`ref.txt` of oscar was extended after Part 1: it now has 1625 words, ~73 before 08:30 and ~45 after
18:30. Scored as it is, every run gets CER ~11%. The numbers above are scored against it trimmed to
the section (1507 words, as in Part 1).

### 15.6 Merge: the last word of a sentence went to the next speaker

The transcript of §15.5 has 12 changes where a new speaker starts mid-sentence. In 7 of them the
moved piece is one word, the last of the previous speaker's sentence:

```
words:  … kulis  nedir  hocam?[150.57–152.62]  Aslında  kulis …
turns:  SPK_0 ………………………………… 151.51 | SPK_2 ……………………………
out:    SPK_0: … kulis nedir    SPK_2: hocam? Aslında kulis …
```

**Cause.** Whisper leaves no gap between words: the pause after a sentence is folded into its
last word, which then lasts 2–3 s. "hocam?" reaches 1.1 s into SPK_2's turn, overlaps it more than
SPK_0's (1.11 s vs 0.94 s) and goes to SPK_2. Snap should have moved it back, since the "?" is a
break right after it, but it measured from the word's start to the next word's start, that is the
stretched word's own length: 2.05, 2.07, 2.19, 2.25, 2.84, 2.87 and 3.14 s in the 7 cases, all
just over `SNAP_MS` (2 s). The repair failed exactly where it was needed. This is also why the
cut is always early, never late: only the outgoing speaker's word is stretched.

| Word length in oscar (3045 words) | Median | 99% | Max |
|---|---|---|---|
| Inside a sentence (2562) | 270 ms | 1140 ms | 4.0 s |
| Before punctuation `. , ? ! …` (483) | 750 ms | 3220 ms | 4.4 s |

**Fix** (`pipeline/merge.py`). Speakers and breaks are decided on word times capped at 1.5 s
(`MAX_WORD_MS`: longer than almost any word inside a sentence), and snap measures to the nearer
edge of a pause. The transcript keeps Whisper's times. Merge was rerun on oscar (diarization
unchanged, 0.0 s, 103 utterances as before):

| | Before | After |
|---|---|---|
| Last word of a sentence on the next speaker | 7 | **1**: fixed at 2:30 "hocam?", 5:23 "var.", 10:13 "ettiniz?", 14:28 "yani.", 16:25 "ikincisinde.", 17:37 "ekiple." |
| Left | | 15:04 "sonra.": the only word in a 1.9 s turn of another speaker; snap never empties an utterance |
| One-window false turns inside a monologue | 3 | 3: not a merge problem (§19) |

**Side effect: 5 sentences changed speaker.** A stretched word used to carry its whole sentence
to the speaker of the following turn. Now each sentence takes the speaker whose turn it lies in:

| Time | Sentence | Was → now | Why |
|---|---|---|---|
| 1:16 | "O süreci biraz anlatarak başlarsak." | SPK_2 → SPK_0 | Inside SPK_0's turn; "başlarsak." lasted 4.2 s |
| 4:24 | "Orada resepsiyonlar veriyorsunuz." / "- Resepsiyon, sahne sergimekler." | SPK_0 / SPK_2 → SPK_2 / SPK_0 | A near tie: "veriyorsunuz." overlaps the two turns 0.77 s vs 0.73 s |
| 14:56 | "Ve yönetmen olarak sizin beklentiniz neydi?" | SPK_1 → SPK_0 | Inside SPK_0's turn; "neydi?" lasted 2.9 s |
| 16:41 | "Bu oyuncu yönetmen görüşmesi mi deyip görüştüğünüzü?" | SPK_0 → SPK_1 | Inside SPK_1's turn; three speakers within 8 s here |
| 19:00 | "Rookie graduation vermiştir ve sette problem yaşayabilirsiniz." | SPK_2 → SPK_0 | Inside SPK_0's 4.5 s turn; "yaşayabilirsiniz." lasted 4.0 s |

The merge now follows diarization where it used to be overruled by a word's timing, so it is
right where diarization is. The transcript was reviewed after the fix and accepted as it is; the
changes were not checked by ear one by one.

## 16. Observations

**1. A threshold tuned on short clips fails on long recordings.** AHC 0.65 gave 3 speakers on the
1.5 min sessions and 69 on balkan. Over 30 minutes a voice changes more (laughing, louder, further
from the mic), and the clusters of one person drift apart. Speaker counts must be checked on full
meetings.

**2. AHC has a cliff, spectral a plateau.** AHC's fragments only disappear at the threshold where
real speakers start to merge. Spectral reads the count from the eigengap and gave the same answer
across models, windows, `p_ratio` ≥ 0.25 and `min_cluster`.

**3. Two networks agree.** ResNet34 and CAM++ were trained separately, and with spectral they give
the same speaker to 94–98% of the speech. So the choice between them is about cost, and CAM++ is
3.2× cheaper.

**4. CMN is the one embedding detail that matters.** Without it every voice looks alike (same
mic, same room). Per window or per region makes no difference.

**5. Parallel benchmarks need one BLAS thread per worker.** Four workers with default threads made
spectral take 220 s instead of 10 s. The app is not affected: OpenBLAS follows its `taskset -c 0`.

**6. habersizler is not settled.** The settings disagree there (2–6 speakers, 10–50%
disagreement), unlike on the videos. Either it has more speakers, which is harder, or something
in it splits a voice. Its true count decides which.

**7. In parallel, diarization costs STT almost nothing.** Diarization finishes in 3.5 min while STT
needs 23.6 min, so merge never waits. STT is 15% slower only during those 3.5 min.

**8. The remaining errors were where a change is placed, not who speaks, and most came from the
words' times, not from the voiceprints.** Whisper folds the pause after a sentence into its last
word (up to 4.4 s), so that word reached into the next speaker's turn and went to them. Capping word
times at 1.5 s for the merge fixed 6 of 7 such cuts (§15.6). Word times from STT are input to
diarization's last step and need the same care as the voiceprints.

**9. What is left is small and needs labels to tune.** One-window false turns (750 ms, longer than
`MIN_TURN_MS` 500, so smoothing keeps them) and near ties at a change. Each fix trades one error
for another, e.g. a longer `MIN_TURN_MS` would also swallow a real "Evet." (§19).

## 17. Implementation

| File | What |
|---|---|
| `pipeline/diar.py` | `Tuning`: window, step, CMN, linkage, `p_ratio`, merge gap, min turn (the pipeline uses the defaults). `diarize()` = `embed_windows()` + `cluster_turns()`. Spectral clusters at most 1000 windows (`SC_MAX_WINDOWS`) and gives the rest the nearest centroid; below 20 windows it falls back to AHC |
| `pipeline/merge.py` | Words → speakers → utterances. Speaker by overlap on word times capped at `MAX_WORD_MS`; `snap()` moves each change to the nearest pause or sentence end within `SNAP_MS` (distance to the nearer edge of the pause) and never empties an utterance |
| `config.toml [diar]` | `model` CAM++, `method = "spectral"` |
| `eval/der.py` | DER with `--from/--to`, overlap share, WDER after `merge()` (`--words`) |
| `eval/bench_diar.py` | Voiceprints cached per setting, any grid of clusterings, oracle VAD, plausibility measures, CSV |
| `eval/diar_report.py` | Tables of §15.1–15.3 from the CSVs |
| `eval/diar_agree.py` | Disagreement between settings (§15.4) |
| `eval/diar_cost.py` | Clean cost on core 0 (§15.3) |

## 18. Reproduce

```bash
R="recordings/htalks_gs_barcelona recordings/oscar_da_bizi_gorecek_mi recordings/balkan_asidik recordings/habersizler_168"
CL="--method ahc spectral --linkage average complete --threshold 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.1 1.2 1.3 1.4 1.5 1.6
    --p-ratio 0.05 0.1 0.15 0.25 0.35 --min-cluster 0 5000 30000 --jobs 4 --top 0"
python -m eval.bench_diar $R --model resnet34 campplus --window 1000 1500 2000 3000 $CL --out data/bench/diar_grid_window.csv
python -m eval.bench_diar $R --model resnet34 campplus --step 500 $CL --out data/bench/diar_grid_step.csv
python -m eval.bench_diar $R --model resnet34 campplus --cmn region none $CL --out data/bench/diar_grid_cmn.csv
python -m eval.diar_report
python -m eval.diar_agree "resnet34 1500 750 window spectral 0.25" "campplus 1500 750 window spectral 0.25"
OPENBLAS_NUM_THREADS=1 taskset -c 0 python -m eval.diar_cost recordings/habersizler_168

# end-to-end (§15.5): whisper-server on cores 1-3, then
ln -sfn ../../recordings/oscar_da_bizi_gorecek_mi/audio.wav data/test/oscar_da_bizi_gorecek_mi.wav
taskset -c 0 python -m pipeline.run data/test/oscar_da_bizi_gorecek_mi.wav --to merge --parallel
python -m eval.wer data/sessions/oscar_da_bizi_gorecek_mi --from 510 --to 1110 --ref <ref.txt trimmed to the section>

# merge only, after a change in pipeline/merge.py (§15.6): reuses words.json and turns.json
python -m pipeline.run data/test/oscar_da_bizi_gorecek_mi.wav --from merge --to merge

# references: draft from the VAD regions, label by ear in Audacity, then DER / WDER
python -m eval.bench_diar recordings/htalks_gs_barcelona --draft
python -m eval.bench_diar $R --model campplus resnet34 --method spectral ahc --vad system oracle
```

~4.5 h on 4 cores for the grid. Log: `recordings/bench_diar_grid.log`.

## 19. Open items

| Item | Why |
|---|---|
| Hand-label the sections (`ref.labels.txt`) | Everything above is plausibility and agreement. DER/WDER is needed to confirm the choice, and to compare settings that agree with each other |
| Speaker count of habersizler, and a real-mic meeting with 4–6 people | All three videos have 3 speakers: whether the count estimate works for other counts is unknown |
| `MIN_TURN_MS`, `MERGE_GAP_MS`, `SNAP_MS`, `MAX_WORD_MS`, `min_cluster_ms` | They move turn edges and short turns: needs WDER. `MAX_WORD_MS` 1500 was set from word lengths (§15.6), not tuned |
| One-window false turns (oscar 5:33, 7:37, 17:46) | One 750 ms window labelled as another speaker inside a monologue makes a one-word utterance. `MIN_TURN_MS` 500 is shorter than one step, so smoothing never removes them; 800–1000 would, but could also swallow a real interjection |
| Near ties and one-word turns at a change (oscar 4:24, 15:04) | A word split almost evenly between two turns goes either way, and snap can then move the next sentence too. A turn holding a single word ("sonra.") cannot be snapped away by design |
| Voiceprints during the recording | Would leave only ~10 s of clustering after stop instead of ~2.6 min |
| Diarization after a live recording | §15.5 is the upload path. After a live recording diarization runs after stop, mostly alone: expected ~RTF 0.087, not measured |
| oscar `ref.txt` covers more than 08:30–18:30 | Trim it, or move the section, so it scores as in Part 1 without a trimmed copy |

# Part 3 — Memory and a full run

How much RAM Echoscript needs while it works, and how long one whole meeting takes, measured on
the Pi 5 (16 GiB) on 2026-10-19.

## 20. Peak RAM

**Method.** whisper-server: its peak resident memory (`VmHWM` in `/proc/<pid>/status`) after
transcribing all of oscar (55 chunks) and a 3 min clip. Pipeline: every stage run as its own
process under `taskset -c 0`, as `main.py` runs it, and its peak taken from
`getrusage(RUSAGE_CHILDREN).ru_maxrss`. Stages ran on a scratch copy of oscar (22.0 min), STT on a
3 min clip of it (08:30–11:30). Web server: `main.py` imported and the live VAD loaded, without
starting the server.

| Process | When it runs | Peak RAM |
|---|---|---|
| whisper-server, large-v3-turbo q8_0 | always | **1078 MiB** |
| Web server (`main.py`) with the live VAD loaded | always | ~101 MiB |
| Pipeline: convert, VAD, pack, merge | seconds per job | 88–115 MiB |
| Pipeline: STT (sends chunks to whisper-server) | during transcription | 100 MiB |
| Pipeline: diarization, CAM++ + spectral, 22 min | after STT, or next to it (`--parallel`) | **484 MiB** |
| Pipeline: summary, `backend = "hailo"` (3 min clip) | last, alone | 267 MiB |

The diarization run took 147 s alone on core 0 and gave the same turns as the oscar session of §15.5.
On the 3 min clip diarization peaked at 378 MiB, so 18 more minutes added ~100 MiB. The summary was
measured in a full upload through the web API (`main.py`), as the peak of the pipeline process; Qwen3
itself runs in the AI HAT+2's own memory, so this is HailoRT and the pipeline. The summary is its own
job after the transcript (§21), so it can overlap the next meeting's transcription: then it adds ~250 MiB.

| Situation | Running at the same time | Total |
|---|---|---|
| Idle | whisper-server + web server | ~1.2 GiB |
| Live recording | whisper-server + web server with the live VAD | ~1.2 GiB |
| After stop: diarization | + diarization | ~1.7 GiB |
| **Upload: STT ∥ diarization** | + STT process + diarization process | **~1.8 GiB** (the peak; measured together in §21: **1.70 GiB**) |

**Observations.**
- **whisper-server is ~60% of the peak**, and it stays resident: it keeps the model loaded between
  jobs. Everything else is freed when a job ends, because `main.py` runs each job as a subprocess.
- **Diarization should grow little with length.** Features are computed per VAD region, and
  spectral clusters at most 1000 windows (§11), so its largest matrices stay ~8 MB. Not measured
  beyond 22 min.
- **A 4 GB Pi 5 should be enough; 2 GB would be tight** once the OS and the file cache are added.
  An estimate from these numbers, not tested. The q5_0 model is 574 MB on disk instead of 875 MB
  and would make whisper-server smaller, but `main.py` notes that q8_0 encodes a 30 s chunk 2.4×
  faster on the Pi 5, so it is only worth it if RAM becomes the limit.

| Not measured | Why |
|---|---|
| Summary, `backend = "llama"` | llama-server would hold its own model in the Pi's RAM, on top of the table above |
| Meetings longer than 22 min | See above: expected to grow little |

```bash
# whisper-server, after a job
grep -E "VmRSS|VmHWM" /proc/$(pgrep -x whisper-server)/status
# one pipeline stage, as main.py runs it
taskset -c 0 python -c "import resource, subprocess, sys; subprocess.run(sys.argv[1:]);
print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, 'MiB')" \
    python -m pipeline.run <audio.wav> --from diar --to diar
```

## 21. Full run through the web app (oscar, 22.0 min)

**Setup.** The whole system as the README starts it: whisper-server (q8_0, `-t 3`) on cores 1–3,
the app (`main.py`) on core 0. oscar's `audio.wav` was uploaded through `POST /api/meetings` as the
page does, language `tr`, with *Summarize automatically* on (`backend = "hailo"`).
`eval/full_run.py` followed the meeting through the API and sampled every second: the memory in use
on the whole system (`MemTotal − MemAvailable`), the RSS of each Echoscript process, and the CPU
temperature. 2352 samples; raw data in `data/bench/full_run_oscar_*`.

| Step | Time |
|---|---|
| Upload | 0.2 s |
| Convert, VAD, pack | 15 s |
| STT ∥ diarization | from 15 s; diarization done at 4 min 1 s (223 s, RTF 0.169) |
| STT alone | until 24 min 55 s (merge waited 0 s, took < 1 s) |
| **Transcript ready: the meeting is "Ready"** | **24 min 55 s** (1.13 × the recording) |
| Summary, started by itself, Qwen3-1.7B on the AI HAT+2 | 15 min 38 s |
| **Everything, with the summary** | **40 min 33 s** |

| Phase | Echoscript processes, peak | Whole system in use, peak | CPU temperature, peak |
|---|---|---|---|
| **STT ∥ diarization** (3.5 min) | **1701 MiB** | **3674 MiB** | 70.0 °C |
| STT alone (21 min) | 1280 MiB | 3187 MiB | 68.8 °C |
| Summary (15.5 min) | 1404 MiB | 3299 MiB | 60.6 °C |

| Process | Peak RSS |
|---|---|
| whisper-server | 1065 MiB (905 MiB idle, before the first chunk) |
| Diarization (subprocess, next to STT) | 428 MiB |
| Summary (HailoRT side) | 248 MiB |
| Pipeline: STT, then merge | 126 MiB |
| App (`main.py`) | 91 MiB |

**Observations.**
- **Echoscript's peak is 1.70 GiB**, while STT and diarization run together, a little below the
  ~1.8 GiB estimated in §20 by adding separate peaks (diarization: 428 MiB here, 484 MiB alone).
  Before whisper-server started, the OS and the desktop already used ~2.0 GB, hence 3.67 GB in use
  on the whole system.
- **The transcript was ready 1 min later than in the command-line run of §15.5** (24 min 55 s vs
  23 min 49 s). The pipeline is the same; the web app and the monitor run next to it, and whisper
  varies between runs. Not investigated further.
- **The summary no longer delays the transcript.** It ran as its own job after the meeting was
  "Ready", and the transcript could be read meanwhile. It took 15.5 min for 22 min of meeting: the
  2048-token context of the HAT splits the transcript into 9 parts, which are then merged.
- **The transcript is identical to §15.5's**, all 103 utterances, with 3 speakers. The summary is
  not: Qwen3 samples at temperature 0.2. This run's summary covers the Oscar nomination; the earlier
  one (README, *Known limitations*) missed it. Both are repetitive.
- **Never throttled** (`vcgencmd get_throttled` = `0x0`); 70 °C at most, during STT ∥ diarization.

```bash
# whisper-server on cores 1-3 and the app on core 0 (README, "Running it"), then
python eval/full_run.py http://127.0.0.1:8000 recordings/oscar_da_bizi_gorecek_mi/audio.wav runs/full
```
