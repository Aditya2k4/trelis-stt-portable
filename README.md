# Trelis STT for Windows and Intel graphics

A portable local transcription setup for a Windows 11 laptop with an Intel Core Ultra / Intel Arc GPU. The source setup was tested on a Core Ultra 7 268V, Arc 140V and 32 GB RAM. No NVIDIA GPU or cloud inference is required for this configuration.

**The entire model package is available in [Releases](https://github.com/Aditya2k4/trelis-stt-portable/releases/latest), without signing in.** It is split into six files because GitHub limits each release asset to under 2 GiB. The downloader below rejoins them automatically. Do not try to unzip an individual `.part001` file.

## Set up on another laptop

1. Install **64-bit Python 3.12** with its Windows `py` launcher, and **FFmpeg**. Use [Python's official downloads](https://www.python.org/downloads/windows/) and [FFmpeg's download page](https://ffmpeg.org/download.html), or your approved package manager. Ensure the Intel graphics driver is installed. Keep approximately **35 GB of disk space free** for download parts, the joined archive, extracted model, and dependencies.
2. Choose **Code → Download ZIP** on this page and extract it. Alternatively, clone this repository.
3. Open PowerShell in the extracted repository folder and run:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\Get-Trelis.ps1
   ```

   This downloads about **9.31 GB**, verifies every part and the full archive, then extracts and verifies `package\Trelis-STT`. Downloads can resume if interrupted. Python handles the large ZIP64 files; no separate archive utility is needed.

4. Install the transcription environment:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\package\Trelis-STT\setup-windows.ps1
   ```

5. Follow `package\Trelis-STT\START HERE.txt`. Run its included public smoke test before using your own recordings. **For ChatGPT-assisted setup, paste [CHATGPT HANDOFF.txt](handoff/CHATGPT%20HANDOFF.txt) into a local ChatGPT/Codex session and give it the actual extracted package location.** A web-only chat can guide commands but cannot directly operate your laptop.

The first dependency installation needs internet. Model inference is local and offline. No recordings are uploaded by the transcriber. Python, FFmpeg, drivers and Python wheels are installed separately; the full model weights are included in the release.

## Downloads

- **Full model package:** all six `Trelis-STT-FULL.zip.part…` assets in the [release](https://github.com/Aditya2k4/trelis-stt-portable/releases/latest). Use `Get-Trelis.ps1` to download, verify and extract them.
- **Small ChatGPT kit:** `Trelis-STT-KIT.zip`, about 45.7 MB. Contains documentation, scripts, the experimental adapter and public pilot data. **It does not contain the base-model weights and cannot transcribe by itself.**
- `release-manifest.json` lists exact sizes and SHA256 hashes. The complete archive hash is `165ae0f825575fe3b2225b07b402f2ba5c8372e1af621f93693e8a93278cf00a`.
- GitHub's automatically generated **Source code ZIP** contains this repository, not the multi-gigabyte model. Use the release assets for the full package.

## Included

- Original `Trelis/whisper-hinglish-preview` weights at revision `eab1188fd2d0e91f2584229b32b3bfe1901c896c`.
- The tested FP16 OpenVINO export for Intel GPU inference.
- A portable, resumable batch transcriber, dependency setup, checksums, and public smoke audio.
- The actual decoder-only rank-8 LoRA adapter, training/evaluation scripts, public pilot clips, settings, and aggregate results.
- A detailed setup guide and a copy-paste prompt for ChatGPT.

**The original model is the default. The LoRA adapter is separate, unmerged, and disabled:** the small local pilot produced **21.44% WER for both the original and adapter** on 12 reviewed meeting clips (122 errors / 569 words). Two clips improved, one worsened, nine were unchanged. No overall accuracy improvement was established. See [the experiment guide](handoff/fine-tuning/EXPERIMENT%20GUIDE.txt).

The portable runner passed a local Intel GPU test with two public inputs, duplicate skipping, batching, and continuous full-duration coverage. Package weight hashes matched the originals and both archives were checked. That is not a guarantee of compatibility on an untested device: run the included diagnostics and smoke test after installation. Automatic transcription can misrecognize names and noisy speech; timestamps are approximate audio-window boundaries, not word alignments. Speaker diarization is not included.

## Data and licenses

No private meeting recordings, transcripts, corrected private references, credentials, or application database are distributed here. Public pilot labels are publisher-provided and were not manually certified; source group IDs are not independently verified speaker identities.

The Trelis model card declares Apache-2.0; the code and license notices are included. Public pilot data comes from [agarwalayushi/hinglish](https://huggingface.co/datasets/agarwalayushi/hinglish), whose compilation declares CC BY 4.0 and references [AI4Bharat Spoken-Tutorial](https://huggingface.co/datasets/ai4bharat/Spoken-Tutorial). Per-row original provenance was not independently certified. Preserve the [full attribution](handoff/licenses/ATTRIBUTION.txt) when redistributing.

This is an independent deployment handoff, not an official Trelis product or an endorsement by its publishers. It contains the model engine and CLI, not the previous desktop GUI.

Release asset size reference: [GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
