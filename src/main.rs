use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use clap::{command, Parser};
use itertools::Itertools;
use rayon::prelude::*;
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
            prompts.push(format!("{}\x16{}\x17+\0", prompt, self.query));
        }
        for prompt in &self.neg {
            prompts.push(format!("{}\x16{}\x17-\0", prompt, self.query));
        }
        prompts
    }
}

#[derive(Debug, Clone)]
struct AppState {
    dataset: Arc<Vec<Vec<u16>>>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct DataItemQuery {
    idx: usize,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, short, value_name = "PATH")]
    path: String,
}

fn main() -> Result<()> {
    let tokenizer = Tokenizer::new(include_str!("rwkv_vocab_v20230424.json"))?;

    let args = Args::parse();

    let mut dataset = vec![];
    // let pattern = "../synthia/bge-m3-data/*/*.jsonl";
    let pattern = &args.path;

    let paths = glob::glob(pattern)?.collect_vec();
    let paths = paths.into_iter().filter_map(|x| x.ok()).collect_vec();
    println!("{:#?}", paths);

    let total = paths.len();
    for (id, path) in paths.into_iter().enumerate() {
        let data: Vec<DataItem> = serde_jsonlines::json_lines(&path)?.try_collect()?;
        println!("{:?}\tdata: {}\t{}/{}", path, data.len(), id, total);

        let text: Vec<_> = data
            .into_par_iter()
            .map(|item| item.format())
            .reduce(Vec::new, |x, y| [x, y].concat());

        let tokens: Vec<_> = text
            .into_par_iter()
            .filter_map(|prompt| tokenizer.encode(prompt.as_bytes()).ok())
            .filter(|prompt| prompt.len() < MAX_LEN)
            .collect();

        let mut padded = vec![];
        let mut start = 0usize;
        for data in tokens {
            let end = start + data.len();
            let buffer = match (padded.last_mut(), end <= MAX_LEN) {
                (Some(buffer), true) => buffer,
                (None, _) | (_, false) => {
                    start = 0;
                    padded.push(vec![0u16; MAX_LEN]);
                    padded.last_mut().unwrap()
                }
            };

            let end = start + data.len();
            buffer[start..end].copy_from_slice(&data);
            start = end;

            assert_eq!(buffer.len(), MAX_LEN);
        }

        dataset.append(&mut padded);
    }

    println!("dataset size: {}", dataset.len());

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
    println!("server started");

    axum::serve(listener, app).await.unwrap();
}

async fn dataset_len(State(state): State<AppState>) -> impl IntoResponse {
    state.dataset.len().to_string()
}

async fn dataset_item(
    Query(query): Query<DataItemQuery>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    match state.dataset.iter().nth(query.idx) {
        Some(data) => Ok(Json(data.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}
