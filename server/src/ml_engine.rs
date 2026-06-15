/// ml_engine.rs — Inférence ONNX via RMBG-1.4 (ort 2.x stable)
/// Input  : [1, 3, 1024, 1024] float32 normalisé (pixel/255 - 0.5)
/// Output : [1, 1, 1024, 1024] float32 sigmoid (0..1 = masque alpha)

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage, GrayImage};
use std::{path::Path, sync::{Mutex, OnceLock}};
use ort::{inputs, session::Session, value::Tensor as OrtTensor};

const INPUT_SIZE: usize = 1024;

static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();

/// Retourne true si le modèle est déjà chargé en mémoire (check non-bloquant).
pub fn is_model_loaded() -> bool {
    SESSION.get().is_some()
}

/// Charge le modèle ONNX une seule fois (singleton). Idempotent.
pub fn init_model(model_path: &Path) -> Result<()> {
    if SESSION.get().is_some() {
        return Ok(());
    }

    if !model_path.exists() {
        return Err(anyhow!(
            "model.onnx introuvable à : {}. Téléchargez RMBG-1.4 depuis HuggingFace.",
            model_path.display()
        ));
    }

    let session = Session::builder()?.commit_from_file(model_path)?;

    SESSION
        .set(Mutex::new(session))
        .map_err(|_| anyhow!("Modèle déjà initialisé (race condition)"))?;

    Ok(())
}

/// Lance l'inférence et retourne le masque alpha (GrayImage taille originale).
pub fn run_inference(img: &DynamicImage) -> Result<GrayImage> {
    let session_mutex = SESSION
        .get()
        .ok_or_else(|| anyhow!("Modèle non initialisé — appelez init_model() d'abord"))?;

    let mut session = session_mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("Session ONNX corrompue — redémarrage requis"))?;

    let (orig_w, orig_h) = (img.width(), img.height());
    if orig_w == 0 || orig_h == 0 {
        return Err(anyhow!("Image invalide : dimensions 0×0"));
    }

    // ── Prétraitement ─────────────────────────────────────────────────────────
    let resized = img.resize_exact(INPUT_SIZE as u32, INPUT_SIZE as u32, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Tenseur CHW [1, 3, H, W] : (pixel/255) - 0.5
    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut data = vec![0.0f32; 3 * plane];

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let idx = y as usize * INPUT_SIZE + x as usize;
        data[idx]             = pixel[0] as f32 / 255.0 - 0.5; // R
        data[plane + idx]     = pixel[1] as f32 / 255.0 - 0.5; // G
        data[2 * plane + idx] = pixel[2] as f32 / 255.0 - 0.5; // B
    }

    // ── Création du tenseur ort ────────────────────────────────────────────────
    let shape = [1usize, 3, INPUT_SIZE, INPUT_SIZE];
    let tensor = OrtTensor::from_array((shape, data))
        .map_err(|e| anyhow!("Création tenseur : {e}"))?;

    // ── Inférence ─────────────────────────────────────────────────────────────
    let outputs = session.run(inputs!["input" => tensor]).map_err(|e| anyhow!("{e}"))?;

    // ── Post-traitement ───────────────────────────────────────────────────────
    let (_, mask_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("Extraction tenseur de sortie : {e}"))?;

    let raw_mask: Vec<u8> = mask_data
        .iter()
        .map(|&v: &f32| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();

    let mask_1024 = GrayImage::from_raw(INPUT_SIZE as u32, INPUT_SIZE as u32, raw_mask)
        .ok_or_else(|| anyhow!("Impossible de créer GrayImage depuis le masque"))?;

    let mask_orig = image::imageops::resize(&mask_1024, orig_w, orig_h, FilterType::Lanczos3);

    Ok(mask_orig)
}
