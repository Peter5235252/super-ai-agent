//! Speech-to-text: microphone capture (cpal) + local Whisper (whisper-rs).
//!
//! Fully offline after a one-time ~75 MB model download into the app data
//! dir: no accounts, no keys, nothing leaves the PC. The UI records while
//! the mic toggle is on, then transcribes in the background and appends
//! the text to the composer draft.
//!
//! Only [`to_mono_16k`] is unit-tested: CI runners have no microphone and
//! must not download model weights. All fallible device/model paths report
//! plain-language errors for the UI notice line.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Whisper model: English-only `tiny` — fast on CPU, good enough for
/// dictating chat messages. One-time download, then fully offline.
pub const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";
pub const MODEL_FILE: &str = "ggml-tiny.en.bin";
/// Whisper wants 16 kHz mono f32.
pub const TARGET_RATE: u32 = 16_000;
/// Auto-stop recordings here so a forgotten mic can't eat RAM/CPU.
pub const MAX_RECORD_SECS: u64 = 120;
/// Peak amplitude below this counts as silence.
pub const SILENCE_PEAK: f32 = 0.01;

pub fn model_path(data_dir: &Path) -> PathBuf {
    data_dir.join("models").join(MODEL_FILE)
}

/// Download the model on first use. Skipped when already present.
pub async fn ensure_model(data_dir: &Path) -> Result<PathBuf, String> {
    let dir = data_dir.join("models");
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("can't create models folder: {e}"))?;
    let path = dir.join(MODEL_FILE);
    if tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok(path);
    }
    let bytes = reqwest::get(MODEL_URL)
        .await
        .map_err(|e| format!("model download failed: {e}"))?
        .bytes()
        .await
        .map_err(|e| format!("model download failed: {e}"))?;
    if bytes.is_empty() {
        return Err("model download came back empty; check your connection".into());
    }
    tokio::fs::write(&path, &bytes)
        .await
        .map_err(|e| format!("can't save speech model: {e}"))?;
    Ok(path)
}

/// Average channels down to mono, then linear-resample to 16 kHz.
/// Pure and unit-tested (no microphone needed).
pub fn to_mono_16k(interleaved: &[f32], channels: u16, sample_rate: u32) -> Vec<f32> {
    if interleaved.is_empty() || channels == 0 || sample_rate == 0 {
        return Vec::new();
    }
    let ch = channels as usize;
    let frames = interleaved.len() / ch;
    if frames == 0 {
        return Vec::new();
    }
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut sum = 0.0f32;
        for c in 0..ch {
            sum += interleaved[f * ch + c];
        }
        mono.push(sum / ch as f32);
    }
    if sample_rate == TARGET_RATE {
        return mono;
    }
    // Linear resample to 16 kHz.
    let ratio = f64::from(sample_rate) / f64::from(TARGET_RATE);
    let out_len = (frames as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let i0 = pos as usize;
            let i1 = (i0 + 1).min(frames - 1);
            let frac = (pos - i0 as f64) as f32;
            mono[i0].mul_add(1.0 - frac, mono[i1] * frac)
        })
        .collect()
}

/// Loudest absolute sample; below [`SILENCE_PEAK`] the mic heard nothing.
pub fn peak(pcm: &[f32]) -> f32 {
    pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}

pub struct Recording {
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
    pub secs: u64,
}

/// Live capture handle. Dropping the stream stops the microphone, so
/// [`Recorder::finish`] consumes `self` to end the recording.
pub struct Recorder {
    _stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    channels: u16,
    sample_rate: u32,
    started: Instant,
    pub device_name: String,
}

pub fn start_recording() -> Result<Recorder, String> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| {
        "No microphone found. Plug one in, then check Windows Settings → Sound → Input.".to_string()
    })?;
    let desc = device
        .description()
        .map_err(|e| format!("Can't inspect the microphone: {e}"))?;
    let device_name = desc.name().to_string();
    let config = device
        .default_input_config()
        .map_err(|e| format!("Can't open {device_name}: {e}"))?;
    let channels = config.channels();
    let sample_rate = config.sample_rate();
    let sample_format = config.sample_format();
    let samples = Arc::new(Mutex::new(Vec::with_capacity(
        sample_rate as usize * channels as usize * 10,
    )));
    let writer = samples.clone();
    let push = move |data: &[f32]| {
        writer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(data);
    };
    let err_fn = |err| tracing::warn!("microphone stream error: {err}");
    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config.into(),
            move |data: &[f32], _| push(data),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config.into(),
            move |data: &[i16], _| {
                push(
                    &data
                        .iter()
                        .map(|s| *s as f32 / f32::from(i16::MAX))
                        .collect::<Vec<_>>(),
                );
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            config.into(),
            move |data: &[u16], _| {
                push(
                    &data
                        .iter()
                        .map(|s| (*s as f32 - 32768.0) / 32768.0)
                        .collect::<Vec<_>>(),
                );
            },
            err_fn,
            None,
        ),
        fmt => {
            return Err(format!(
                "Microphone {device_name} uses an unsupported format ({fmt:?})."
            ));
        }
    }
    .map_err(|e| format!("Can't open {device_name}: {e}"))?;
    stream
        .play()
        .map_err(|e| format!("Can't start {device_name}: {e}"))?;
    Ok(Recorder {
        _stream: stream,
        samples,
        channels,
        sample_rate,
        started: Instant::now(),
        device_name,
    })
}

impl Recorder {
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn buffered_secs(&self) -> u64 {
        let frames = self.samples.lock().unwrap_or_else(|e| e.into_inner()).len()
            / self.channels.max(1) as usize;
        let rate = self.sample_rate.max(1) as u64;
        frames as u64 / rate
    }

    pub fn finish(self) -> Recording {
        let samples = std::mem::take(&mut *self.samples.lock().unwrap_or_else(|e| e.into_inner()));
        Recording {
            samples,
            channels: self.channels,
            sample_rate: self.sample_rate,
            secs: self.started.elapsed().as_secs(),
        }
    }
}

/// Run Whisper over 16 kHz mono samples. CPU-heavy: call from a blocking
/// thread. Returns the transcribed text (may be empty on silence).
pub fn transcribe(model_path: &Path, pcm16k: &[f32]) -> Result<String, String> {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
    if pcm16k.is_empty() {
        return Ok(String::new());
    }
    let mut pcm = pcm16k.to_vec();
    pcm.truncate((TARGET_RATE as usize) * 180);
    let path = model_path
        .to_str()
        .ok_or_else(|| "speech model path is not valid UTF-8".to_string())?;
    let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
        .map_err(|e| format!("Can't load speech model (try deleting the models folder): {e}"))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    let mut state = ctx
        .create_state()
        .map_err(|e| format!("Speech engine failed to start: {e}"))?;
    state
        .full(params, &pcm)
        .map_err(|e| format!("Transcription failed: {e}"))?;
    let mut out = String::new();
    for seg in state.as_iter() {
        let text = seg
            .to_str_lossy()
            .map_err(|e| format!("Transcription failed: {e}"))?;
        if !out.is_empty() && !text.starts_with(char::is_whitespace) {
            out.push(' ');
        }
        out.push_str(text.trim());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_48k_to_mono_16k() {
        // 1 s of stereo 48 kHz sine-ish ramp.
        let samples: Vec<f32> = (0..48_000)
            .flat_map(|i| {
                let v = i as f32 / 48_000.0;
                [v, v * 0.5]
            })
            .collect();
        let mono = to_mono_16k(&samples, 2, 48_000);
        assert_eq!(mono.len(), 16_000);
        // first frame: (0 + 0) / 2
        assert!((mono[0] - 0.0).abs() < 1e-6);
        // mono average of the pair lands between the channels
        let last_src = 47_999.0 / 48_000.0;
        assert!((mono[15_999] - last_src * 0.75).abs() < 0.01);
    }

    #[test]
    fn passthrough_16k_mono() {
        let samples = vec![0.1, -0.2, 0.3];
        assert_eq!(to_mono_16k(&samples, 1, 16_000), samples);
    }

    #[test]
    fn empty_and_degenerate_input() {
        assert!(to_mono_16k(&[], 1, 16_000).is_empty());
        assert!(to_mono_16k(&[0.5], 0, 16_000).is_empty());
        assert!(to_mono_16k(&[0.5], 1, 0).is_empty());
        // fewer samples than channels: no full frame
        assert!(to_mono_16k(&[0.5], 2, 16_000).is_empty());
    }

    #[test]
    fn peak_detects_silence() {
        assert!(peak(&[0.0, 0.001, -0.002]) < SILENCE_PEAK);
        assert!(peak(&[0.0, 0.5]) >= SILENCE_PEAK);
        assert_eq!(peak(&[]), 0.0);
    }
}
