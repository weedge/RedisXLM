use num_cpus;
use serde::{Deserialize, Serialize};
use std::{env, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigParams {
    /// The number of threads each individual endpoint session can use.
    pub llama_cpp_inference_threads: u32,

    // just for admin to
    // TODO temporary, until the model parameter in incoming requests can be parsed into local paths
    pub inferences_models_dir: String,
    /// The inference model that Edgen will use when the user does not provide a model
    pub inferences_model_name: String,
    /// The inference model repo eg: hugface model repo
    pub inferences_model_repo: String,

    // TODO temporary, until the model parameter in incoming requests can be parsed into local paths
    pub embeddings_models_dir: String,
    /// The embeddings model that Edgen will use when the user does not provide a model
    pub embeddings_model_name: String,
    /// The embeddings repo that Edgen will use for downloads
    pub embeddings_model_repo: String,

    /// The maximum size, in bytes, any request can have. This is most relevant in requests with files, such as audio
    /// transcriptions.
    pub max_request_size: usize,
}

impl ConfigParams {
    pub fn auto_threads(&self, physical: bool) -> u32 {
        let max_threads = if physical {
            num_cpus::get_physical()
        } else {
            num_cpus::get()
        };
        let max_threads = max_threads as u32;

        if self.llama_cpp_inference_threads == 0 || self.llama_cpp_inference_threads > max_threads {
            max_threads
        } else {
            self.llama_cpp_inference_threads
        }
    }
}

impl Default for ConfigParams {
    fn default() -> Self {
        let data_dir = env::current_dir().unwrap();
        let inferences_dir = data_dir.join(Path::new("models/inferences"));
        let embeddings_dir = data_dir.join(Path::new("models/embeddings"));

        let inferences_str = inferences_dir.into_os_string().into_string().unwrap();
        let embeddings_str = embeddings_dir.into_os_string().into_string().unwrap();

        let cpus = num_cpus::get_physical();
        let threads = if cpus > 1 { cpus - 1 } else { 1 };

        Self {
            llama_cpp_inference_threads: threads as u32,
            inferences_model_name: "neural-chat-7b-v3-3.Q4_K_M.gguf".to_string(),
            inferences_model_repo: "TheBloke/neural-chat-7B-v3-3-GGUF".to_string(),
            inferences_models_dir: inferences_str,
            embeddings_model_name: "nomic-embed-text-v1.5.f16.gguf".to_string(),
            embeddings_model_repo: "nomic-ai/nomic-embed-text-v1.5-GGUF".to_string(),
            embeddings_models_dir: embeddings_str,
            max_request_size: 1024 * 1014 * 100, // 100 MB
        }
    }
}
