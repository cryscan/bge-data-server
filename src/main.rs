use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv::tokenizer::Tokenizer;

const MAX_LEN: usize = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    pub query: String,
    pub pos: Vec<String>,
    pub neg: Vec<String>,
}

impl DataItem {
    pub fn format(&self) -> Vec<String> {
        let mut prompts = Vec::with_capacity(self.pos.len() + self.neg.len());
        for prompt in &self.pos {
            prompts.push(format!("{}\x16{}\x17+", prompt, self.query));
        }
        for prompt in &self.neg {
            prompts.push(format!("{}\x16{}\x17-", prompt, self.query));
        }
        prompts
    }
}

#[derive(Debug, Clone)]
struct AppState {
    dataset: Arc<Vec<Vec<u16>>>,
}

fn main() -> Result<()> {
    let tokenizer = Tokenizer::new(include_str!("rwkv_vocab_v20230424.json"))?;

    let mut dataset = vec![];
    let pattern = "../synthia/bge-m3-data/*/*.jsonl";
    let total = glob::glob(pattern)?.count();
    for (id, path) in glob::glob(pattern)?.enumerate() {
        let Ok(path) = path else {
            continue;
        };

        let mut text = vec![];
        let data: Vec<DataItem> = serde_jsonlines::json_lines(&path)?.try_collect()?;
        for item in data {
            let mut prompts = item.format();
            text.append(&mut prompts);
        }

        let mut tokens = Vec::with_capacity(text.len());
        for prompt in text {
            let prompt = tokenizer.encode(prompt.as_bytes())?;
            if prompt.len() < MAX_LEN {
                tokens.push(prompt);
            }
        }

        println!("{:?}\tdata: {}\t{}/{}", path, tokens.len(), id, total);

        dataset.append(&mut tokens);
    }

    axum_main(dataset);

    Ok(())
}

#[tokio::main]
async fn axum_main(dataset: Vec<Vec<u16>>) {
    let state = AppState {
        dataset: Arc::new(dataset),
    };
    let app = Router::new()
        .route("/len", get(dataset_len))
        .route("/item", get(dataset_item))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9961").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn dataset_len(State(state): State<AppState>) -> impl IntoResponse {
    state.dataset.len().to_string()
}

async fn dataset_item(
    Query(index): Query<usize>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    match state.dataset.iter().nth(index) {
        Some(data) => Ok(Json(data.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}
