---
language:
- hi
- en
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- hinglish
- code-switching
- hindi
- asr
- speech-recognition
base_model:
- ARTPARK-IISc/whisper-large-v3-vaani-hindi
- openai/whisper-large-v3
---

# Whisper Hinglish (Preview)

A Whisper-large-v3 model specialised for **Hinglish (Hindi–English code-switched) speech**, with strong
pure-Hindi and English transcription. It is the best-performing open weights model in our internal evaluation
across code-switch, Hindi, and English benchmarks.

> **Research preview.** Numbers and weights may change. Evaluated on internal benchmarks; see disclaimers below.

---

## Try it live

Available on **Trelis Router** (<https://router.trelis.com/models>):

- **UI** — log in and upload an audio clip to transcribe right in the browser.
- **API** — `POST https://router.trelis.com/api/v1/transcribe` for programmatic access (requires an API key).



## Evaluation

Corpus WER (%, lower is better) under a **script-safe `indic-hindi` normaliser** (NFC + Indic
normalisation, keeps Devanagari matras/nuktas, strips punctuation; not the Whisper default, which strips
matras and inflates Devanagari WER). Compared against two leading commercial APIs: **Sarvam** (Saaras-v3)
and **ElevenLabs Scribe-v2**.

### 🟠 Hinglish — code-switched (Hindi + English in one utterance, each in their native script)

| Benchmark | **whisper-hinglish-preview** | Sarvam | Scribe-v2 | whisper-large-v3 | Vaani |
|---|--:|--:|--:|--:|--:|
| [CoSHE-500 (conversational CS)](https://huggingface.co/datasets/Trelis/CoSHE-500) | 13.67 | **11.47** ᶜᵐ | 12.43 | 29.74 | 73.96 |
| cs-fleurs (read CS) | 10.19 | 16.47 ᶜᵐ | **7.57** | 33.92 | 34.12 |
| hiacc-adult (accented CS) | **12.73** | 14.44 ᶜᵐ | 16.98 | 28.53 | 60.09 |
| hiacc-child (accented CS) | **10.69** | 14.11 ᶜᵐ | 18.36 | 27.91 | 32.17 |

### 🔵 Hindi (pure Devanagari)

| Benchmark | **whisper-hinglish-preview** | Sarvam | Scribe-v2 | whisper-large-v3 | Vaani |
|---|--:|--:|--:|--:|--:|
| Common Voice Hindi (cv-hi) | 12.86 | **12.40** | 13.44 | 30.82 | 14.48 |
| FLEURS-hi | 12.57 | **10.07** | 11.33 | 27.50 | 11.58 |

### ⚪ English

| Benchmark | **whisper-hinglish-preview** | Sarvam | Scribe-v2 | whisper-large-v3 | Vaani |
|---|--:|--:|--:|--:|--:|
| FLEURS-en | 6.93 | 5.14 | **4.01** | 4.81 | 101.66 |

**Bold** = best on that row.

---

## How to use

Like any Whisper model, **specify the language** when you transcribe.

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf, torch

repo = "Trelis/whisper-hinglish-preview"
proc = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda").eval()

audio, sr = sf.read("clip.wav")  # 16 kHz mono
feat = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.bfloat16)

# Hindi audio → force <|hi|> ; English audio → force <|en|>
ids = proc.tokenizer.convert_tokens_to_ids
prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), ids("<|transcribe|>"), ids("<|notimestamps|>")]
out = model.generate(input_features=feat,
                     decoder_input_ids=torch.tensor([prompt]).to("cuda"),
                     max_new_tokens=440)
print(proc.tokenizer.decode(out[0], skip_special_tokens=True))
```

**Code-switched audio.** The model uses a dedicated `<|mixedcode|>` marker/token for utterances that mix
Devanagari and Latin script. Insert it right after the language token, choosing the language token by the
dominant script of the utterance:

```python
mc = proc.tokenizer("<|mixedcode|>", add_special_tokens=False).input_ids
prompt = [ids("<|startoftranscript|>"), ids("<|hi|>"), *mc, ids("<|transcribe|>"), ids("<|notimestamps|>")]
```

---


## Disclaimers

- **Commercial-API WERs on pure Hindi benchmarks here are pessimistic.** Sarvam and Scribe keep English loanwords in Latin script
  and numbers as digits, whereas our references render everything in Devanagari. A translit-blind WER then
  charges a substitution per loanword/number against them. The comparison is apples-to-apples on *our*
  Devanagari-reference protocol, not a claim about their raw quality.
- ᶜᵐ Sarvam evaluated in its code-mixed mode.
- **Specify the language** (`<|hi|>` / `<|en|>`) as shown above — standard Whisper usage — for the
  reported quality.

---

## Attributions

- **Architecture base:** [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3).
- **Starting checkpoint — Whisper-Vaani.** Our Hindi/Hinglish training started from
  [`ARTPARK-IISc/whisper-large-v3-vaani-hindi`](https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi),
  a Vaani-fine-tuned Whisper-large-v3 from the **Vaani** project (ARTPARK @ IISc). We gratefully credit the
  Whisper-Vaani model and the Vaani team.
- **Evaluation benchmark:** CoSHE-500 is derived from [`soketlabs/CoSHE-Eval`](https://huggingface.co/datasets/soketlabs/CoSHE-Eval) (Soket Labs, CC-BY-NC-4.0).