/// main.rs — PureRemove Web : API axum + frontend statique.
/// POST /api/process : multipart (file + options JSON) → PNG détouré.

mod image_processor;
mod ml_engine;

use anyhow::anyhow;
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::{header, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use image_processor::BackgroundColor;
use serde::Deserialize;
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use tokio::sync::Semaphore;
use tower_governor::{
    governor::GovernorConfigBuilder, key_extractor::KeyExtractor, GovernorError, GovernorLayer,
};
use tower_http::{
    services::{ServeDir, ServeFile},
    timeout::TimeoutLayer,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024; // 25 MB
// Le Mutex<Session> dans ml_engine sérialise déjà l'inférence → 1 seul permit suffit
const MAX_INFLIGHT: usize = 1; // Mutex<Session> sérialise — pas de concurrence réelle

// ─── État partagé ─────────────────────────────────────────────────────────────

struct AppState {
    model_path: PathBuf,
    inference_permits: Semaphore,
}

// ─── Extraction IP cliente (derrière Cloudflare + nginx) ──────────────────────

#[derive(Clone)]
struct ClientIpExtractor;

impl KeyExtractor for ClientIpExtractor {
    type Key = IpAddr;

    fn extract<B>(&self, req: &Request<B>) -> Result<Self::Key, GovernorError> {
        let header_ip = |name: &str| -> Option<IpAddr> {
            req.headers()
                .get(name)?
                .to_str()
                .ok()?
                .split(',')
                .next()?
                .trim()
                .parse()
                .ok()
        };

        header_ip("cf-connecting-ip")
            .or_else(|| header_ip("x-real-ip"))
            .or_else(|| {
                req.extensions()
                    .get::<axum::extract::ConnectInfo<SocketAddr>>()
                    .map(|ci| ci.0.ip())
            })
            .ok_or(GovernorError::UnableToExtractKey)
    }
}

// ─── Options de traitement ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ProcessOptions {
    background: BackgroundColor,
}

impl Default for ProcessOptions {
    fn default() -> Self {
        Self { background: BackgroundColor::Transparent }
    }
}

// ─── Erreur API → réponse HTTP ────────────────────────────────────────────────

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

fn bad_request(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

fn internal(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, msg.into())
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

async fn health(_state: State<Arc<AppState>>) -> impl IntoResponse {
    // is_model_loaded() est non-bloquant : ne charge pas le modèle, ne bloque pas tokio
    let model_ok = ml_engine::is_model_loaded();
    Json(serde_json::json!({
        "status": "ok",
        "model": model_ok,
        "version": VERSION,
    }))
}

async fn process_image(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    // ── Lecture multipart : file + options ──
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut options = ProcessOptions::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| bad_request(format!("Multipart invalide : {e}")))?
    {
        match field.name().unwrap_or("") {
            "file" => {
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| bad_request(format!("Lecture fichier : {e}")))?;
                file_bytes = Some(data.to_vec());
            }
            "options" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| bad_request(format!("Lecture options : {e}")))?;
                options = serde_json::from_str(&text)
                    .map_err(|e| bad_request(format!("Options JSON invalides : {e}")))?;
            }
            _ => {} // champ inconnu ignoré
        }
    }

    let bytes = file_bytes.ok_or_else(|| bad_request("Champ 'file' manquant"))?;
    if bytes.is_empty() {
        return Err(bad_request("Fichier vide"));
    }

    // ── Modèle prêt ? (idempotent) ──
    ml_engine::init_model(&state.model_path)
        .map_err(|e| ApiError(StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    // ── Sémaphore : limite les inférences simultanées ──
    let _permit = state
        .inference_permits
        .acquire()
        .await
        .map_err(|_| internal("Serveur en cours d'arrêt"))?;

    // ── Pipeline bloquant hors du runtime tokio ──
    let png = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<u8>> {
        let img = image_processor::load_image_from_bytes(&bytes)?;
        let mask = ml_engine::run_inference(&img)?;
        let result = image_processor::apply_mask(&img, &mask, &options.background)?;
        image_processor::encode_png(&result)
    })
    .await
    .map_err(|e| internal(format!("Tâche interrompue : {e}")))?
    .map_err(|e| {
        tracing::error!("Traitement échoué : {e}");
        bad_request(format!("Traitement impossible : {e}"))
    })?;

    Ok((
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        png,
    )
        .into_response())
}

// ─── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(3002);
    let model_path = PathBuf::from(
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "./model.onnx".into()),
    );
    let static_dir = std::env::var("STATIC_DIR").unwrap_or_else(|_| "../web/dist".into());

    // Init modèle AVANT le bind TCP : bloque le démarrage si le modèle est absent
    ml_engine::init_model(&model_path)?;
    tracing::info!("Modèle RMBG-1.4 chargé : {}", model_path.display());

    let state = Arc::new(AppState {
        model_path,
        inference_permits: Semaphore::new(MAX_INFLIGHT),
    });

    // ── Rate-limit : burst 10, recharge 1 toutes les 6 s (≈ 10 req/min/IP) ──
    let governor_conf = Arc::new(
        GovernorConfigBuilder::default()
            .per_second(6)
            .burst_size(10)
            .key_extractor(ClientIpExtractor)
            .finish()
            .ok_or_else(|| anyhow!("Config rate-limit invalide"))?,
    );

    let api = Router::new()
        .route("/api/process", post(process_image))
        .route("/api/health", get(health))
        .layer(GovernorLayer { config: governor_conf })
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(TimeoutLayer::new(Duration::from_secs(60)))
        .with_state(state);

    // ── Frontend statique (SPA fallback sur index.html) ──
    let index = format!("{static_dir}/index.html");
    let static_service = ServeDir::new(&static_dir).fallback(ServeFile::new(&index));

    let app = api.fallback_service(static_service);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    tracing::info!("PureRemove Web v{VERSION} — écoute sur http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Arrêt demandé");
    })
    .await?;

    Ok(())
}
