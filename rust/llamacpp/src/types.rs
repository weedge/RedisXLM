/*
 * Copyright (c) 2024, weedge <weege007 at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

use llama_cpp::{LlamaModel, LlamaParams, SplitMode};
use llama_cpp_sys::llama_model_default_params;
use redis_module::RedisString;
use redis_module::{native_types::RedisType, raw, RedisValue};
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::os::raw::{c_int, c_void};
use std::{env, fmt, path::Path, sync::Arc};
use std::{mem, ptr};

static APP_VERSION: i32 = 0;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum MType {
    LocalInferenceLm,
    LocalEmbeddingLm,
}
impl Default for MType {
    fn default() -> Self {
        Self::LocalInferenceLm
    }
}
impl From<String> for MType {
    fn from(opts: String) -> Self {
        match opts.as_str() {
            "local_inference_lm" => Self::LocalInferenceLm,
            "local_embedding_lm" => Self::LocalEmbeddingLm,
            _ => unimplemented!(),
        }
    }
}

// for llama.cpp model params
// defualt ModelParams -> default LlamaParams -> c llama_model_default_params()
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ModelParams {
    /// Number of layers to store in VRAM.
    ///
    /// If this number is bigger than the amount of model layers, all layers are loaded to VRAM.
    #[serde(default)]
    pub n_gpu_layers: u32,

    /// How to split the model across multiple GPUs
    #[serde(default = "default_split_mode")]
    pub split_mode: String, //"","layer","row"

    /// The GPU that is used for scratch and small tensors
    #[serde(default)]
    pub main_gpu: u32,

    /// Only load the vocabulary, no weights
    #[serde(default)]
    pub vocab_only: bool,

    /// Use mmap if possible
    #[serde(default = "default_mmap")]
    pub use_mmap: bool,

    /// Force system to keep model in RAM
    #[serde(default)]
    pub use_mlock: bool,
}
fn default_split_mode() -> String {
    "".to_string()
}
fn default_mmap() -> bool {
    let c_params = unsafe { llama_model_default_params() };
    c_params.use_mmap
}
impl Default for ModelParams {
    fn default() -> Self {
        // SAFETY: Stack constructor, always safe
        let c_params = unsafe { llama_model_default_params() };

        Self {
            n_gpu_layers: c_params.n_gpu_layers as u32,
            split_mode: "".to_string(),
            main_gpu: c_params.main_gpu as u32,
            vocab_only: c_params.vocab_only,
            use_mmap: c_params.use_mmap,
            use_mlock: c_params.use_mlock,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelOpts {
    #[serde(default)]
    pub model_path: String,

    #[serde(default)]
    pub model_type: MType,

    #[serde(default)]
    pub model_params: ModelParams,
}
impl Default for ModelOpts {
    fn default() -> Self {
        let data_dir = env::current_dir().unwrap();
        let inferences_dir = data_dir.join(Path::new("models/inferences"));
        let inference_model_path = inferences_dir.join("qwen1_5-1_8b-chat-q8_0.gguf");
        Self {
            model_path: inference_model_path.into_os_string().into_string().unwrap(),
            model_type: MType::LocalInferenceLm,
            model_params: ModelParams::default(),
        }
    }
}

#[derive(Default, Clone)]
pub struct ModelRedis {
    pub name: String,                   // model name
    pub model_opts: ModelOpts,          // model options
    pub model: Option<Arc<LlamaModel>>, // llamacpp model instance
}
impl fmt::Debug for ModelRedis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}, \
           model_opts: {}, \
            ",
            self.name,
            serde_json::to_string(&self.model_opts).unwrap(),
        )
    }
}
impl From<ModelRedis> for RedisValue {
    fn from(model: ModelRedis) -> Self {
        let mut reply: Vec<RedisValue> = Vec::new();
        reply.push("name".into());
        reply.push(model.name.into());
        reply.push("model_opts".into());
        reply.push(serde_json::to_string(&model.model_opts).unwrap().into());
        reply.into()
    }
}

pub struct EmbeddingResult {
    pub vecs: Vec<Vec<f32>>,
}

impl From<EmbeddingResult> for RedisValue {
    fn from(value: EmbeddingResult) -> Self {
        let mut reply: Vec<RedisValue> = Vec::new();
        let mut vecs_f64: Vec<f64> = Vec::new();
        for vec in value.vecs {
            for val in vec {
                vecs_f64.push(val as f64);
            }
            reply.push(vecs_f64.clone().into());
        }
        reply.into()
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "snake_case", default)]
pub struct SampleParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_top_k")]
    pub top_k: i32,

    #[serde(default = "default_top_p")]
    pub top_p: f32,

    #[serde(default = "default_min_p")]
    pub min_p: f32,

    #[serde(default = "default_min_keep")]
    pub min_keep: usize,

    // token_selector: softmax, gready
    // TODO:
    /// [Mirostat](https://arxiv.org/pdf/2007.14966.pdf).
    /// [Mirostat V2](https://arxiv.org/pdf/2007.14966.pdf).
    #[serde(default = "default_token_selector")]
    pub token_selector: String,

    // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    #[serde(default)]
    pub repetition_penalty: RepetitionPenalty,
}
fn default_temperature() -> f32 {
    0.8
}
fn default_top_k() -> i32 {
    40
}
fn default_top_p() -> f32 {
    0.95
}
fn default_min_p() -> f32 {
    0.05
}
fn default_min_keep() -> usize {
    1
}
fn default_token_selector() -> String {
    "softmax".to_string()
}
fn default_max_tokens() -> usize {
    2048
}
impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_k: default_top_k(),
            top_p: default_top_p(),
            min_p: default_min_p(),
            min_keep: default_min_keep(),
            token_selector: default_token_selector(),
            max_tokens: default_max_tokens(),
            repetition_penalty: RepetitionPenalty::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
/// Penalizes generating a token that is within the `last_n` tokens of context in various ways.
pub struct RepetitionPenalty {
    /// Divide the token's logit by this value if they appear one or more time in the `last_n`
    /// tokens. 1.0 disables this, and values from 1.0-1.2 work well.
    ///
    /// See page 5 of <https://arxiv.org/pdf/1909.05858.pdf>
    pub repetition_penalty: f32,

    /// Subtract this value from the token's logit for each time the token appears in the
    /// `last_n` tokens. 0.0 disables this, and 0.0-1.0 are reasonable values.
    ///
    /// See: <https://platform.openai.com/docs/guides/text-generation/parameter-details>
    pub frequency_penalty: f32,

    /// Subtract this value from the token's logit if the token appears in the `last_n` tokens.
    /// 0.0 disables this, and 0.0-1.0 are reasonable values.
    ///
    /// See: <https://platform.openai.com/docs/guides/text-generation/parameter-details>
    pub presence_penalty: f32,

    /// How many tokens back to look when determining penalties. -1 means context size, and 0
    /// disables this stage.
    pub last_n: i32,
}
impl Default for RepetitionPenalty {
    fn default() -> Self {
        Self {
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            last_n: 64,
        }
    }
}

// note: Redis requires the length of native type names to be exactly 9 characters
pub static LLAMACPP_MODEL_REDIS_TYPE: RedisType = RedisType::new(
    "lm_modelx",
    APP_VERSION,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(load_model),
        rdb_save: Some(save_model),
        aof_rewrite: None,
        free: Some(free_model),

        // Currently unused by Redis
        mem_usage: Some(mem_usage_model),
        digest: None,

        // Aux data
        aux_load: None,
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 0,

        copy: Some(copy_model),
        free_effort: None,
        unlink: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        unlink2: None,
    },
);

unsafe extern "C" fn save_model(_rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    let _m = unsafe { &*value.cast::<ModelRedis>() };
    let name_cstring = CString::new(_m.name.as_str()).unwrap();
    raw::save_string(_rdb, name_cstring.to_str().unwrap());

    let opts_serialized_json = serde_json::to_string(&_m.model_opts).unwrap();
    let opts_cjson = CString::new(opts_serialized_json).unwrap();
    raw::save_string(_rdb, opts_cjson.to_str().unwrap());
}

unsafe extern "C" fn load_model(_rdb: *mut raw::RedisModuleIO, encver: c_int) -> *mut c_void {
    match encver {
        0 => {
            let mut model = Box::new(ModelRedis::default());

            model.name = RedisString::from_ptr(raw::RedisModule_LoadString.unwrap()(_rdb))
                .unwrap()
                .to_owned();

            let model_opts_json = RedisString::from_ptr(raw::RedisModule_LoadString.unwrap()(_rdb))
                .unwrap()
                .to_owned();
            model.model_opts = serde_json::from_str(&model_opts_json).unwrap();

            let mut params = LlamaParams::default();
            if model.model_opts.model_params.n_gpu_layers > 0 {
                params.n_gpu_layers = model.model_opts.model_params.n_gpu_layers;
            }
            if model.model_opts.model_params.split_mode == "layer" {
                params.split_mode = SplitMode::Layer;
            } else if model.model_opts.model_params.split_mode == "row" {
                params.split_mode = SplitMode::Row;
            }
            params.main_gpu = model.model_opts.model_params.main_gpu;
            params.use_mlock = model.model_opts.model_params.use_mlock;
            params.use_mmap = model.model_opts.model_params.use_mmap;
            params.vocab_only = model.model_opts.model_params.vocab_only;
            let res = LlamaModel::load_from_file(&model.model_opts.model_path, params);
            if res.is_err() {
                println!(
                    "model {} load_from_file {} error {}",
                    model.name,
                    model.model_opts.model_path,
                    res.as_ref().err().unwrap().to_string()
                );
            }
            let _model = res.unwrap();
            model.model = Some(Arc::new(_model));

            println!("load llamacpp model {:?}", model);
            let model: *mut c_void = Box::into_raw(model) as *mut c_void;
            model
        }
        _ => ptr::null_mut() as *mut c_void,
    }
}

unsafe extern "C" fn free_model(value: *mut c_void) {
    if value.is_null() {
        // on Redis 6.0 we might get a NULL value here, so we need to handle it.
        return;
    }
    drop(Box::from_raw(value as *mut ModelRedis));
}

unsafe extern "C" fn mem_usage_model(_value: *const c_void) -> usize {
    let m = Box::from_raw(_value as *mut ModelRedis);
    mem::size_of::<ModelRedis>() + mem::size_of::<LlamaModel>() + m.name.len() as usize
}

unsafe extern "C" fn copy_model(
    _: *mut raw::RedisModuleString,
    _: *mut raw::RedisModuleString,
    value: *const c_void,
) -> *mut c_void {
    let m = unsafe { &*value.cast::<ModelRedis>() };
    let value = m.clone();
    Box::into_raw(Box::new(value)).cast::<c_void>()
}

#[derive(Default, Clone)]
pub struct PromptRedis {
    pub name: String,         // prompt name
    pub prompts: Vec<String>, // prompts
}
impl fmt::Debug for PromptRedis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}, \
            prompts: {:?}, \
            ",
            self.name, self.prompts
        )
    }
}
impl From<PromptRedis> for RedisValue {
    fn from(prompt: PromptRedis) -> Self {
        let mut reply: Vec<RedisValue> = Vec::new();
        reply.push("name".into());
        reply.push(prompt.name.into());
        reply.push("prompts".into());
        reply.push(prompt.prompts.into());
        reply.into()
    }
}

pub static LLAMACPP_PROMPT_REDIS_TYPE: RedisType = RedisType::new(
    "lm_prompt",
    APP_VERSION,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(load_prompt),
        rdb_save: Some(save_prompt),
        aof_rewrite: None,
        free: Some(free_prompt),

        // Currently unused by Redis
        mem_usage: Some(mem_usage_prompt),
        digest: None,

        // Aux data
        aux_load: None,
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 0,

        copy: Some(copy_prompt),
        free_effort: None,
        unlink: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        unlink2: None,
    },
);

unsafe extern "C" fn save_prompt(_rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    let _m = unsafe { &*value.cast::<PromptRedis>() };
    let name_cstring = CString::new(_m.name.as_str()).unwrap();
    raw::save_string(_rdb, name_cstring.to_str().unwrap());

    let prompts_serialized_json = serde_json::to_string(&_m.prompts).unwrap();
    let prompts_cjson = CString::new(prompts_serialized_json).unwrap();
    raw::save_string(_rdb, prompts_cjson.to_str().unwrap());
}

unsafe extern "C" fn load_prompt(_rdb: *mut raw::RedisModuleIO, encver: c_int) -> *mut c_void {
    match encver {
        0 => {
            let mut prompt = Box::new(PromptRedis::default());

            prompt.name = RedisString::from_ptr(raw::RedisModule_LoadString.unwrap()(_rdb))
                .unwrap()
                .to_owned();

            let prompt_opts_json =
                RedisString::from_ptr(raw::RedisModule_LoadString.unwrap()(_rdb))
                    .unwrap()
                    .to_owned();
            prompt.prompts = serde_json::from_str(&prompt_opts_json).unwrap();

            println!("load prompt {:?}", prompt);
            let prompt: *mut c_void = Box::into_raw(prompt) as *mut c_void;
            prompt
        }
        _ => ptr::null_mut() as *mut c_void,
    }
}

unsafe extern "C" fn free_prompt(value: *mut c_void) {
    if value.is_null() {
        // on Redis 6.0 we might get a NULL value here, so we need to handle it.
        return;
    }
    drop(Box::from_raw(value as *mut PromptRedis));
}

unsafe extern "C" fn mem_usage_prompt(_value: *const c_void) -> usize {
    let m = Box::from_raw(_value as *mut PromptRedis);
    let mut ps = 0;
    for p in m.prompts {
        ps += p.len() as usize;
    }
    m.name.len() as usize + ps
}

unsafe extern "C" fn copy_prompt(
    _: *mut raw::RedisModuleString,
    _: *mut raw::RedisModuleString,
    value: *const c_void,
) -> *mut c_void {
    let m = unsafe { &*value.cast::<PromptRedis>() };
    let value = m.clone();
    Box::into_raw(Box::new(value)).cast::<c_void>()
}

#[derive(Default, Clone)]
pub struct InferenceRedis {
    pub name: String,        // inference name
    pub model_name: String,  // model name
    pub prompt_name: String, // prompt name
}
impl fmt::Debug for InferenceRedis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}, \
            model_name: {}, \
            prompt_name: {}, \
            ",
            self.name, self.model_name, self.prompt_name,
        )
    }
}
impl From<InferenceRedis> for RedisValue {
    fn from(inference: InferenceRedis) -> Self {
        let mut reply: Vec<RedisValue> = Vec::new();
        reply.push("name".into());
        reply.push(inference.name.into());
        reply.push("model_name".into());
        reply.push(inference.model_name.into());
        reply.push("prompt_name".into());
        reply.push(inference.prompt_name.into());
        reply.into()
    }
}

pub static LLAMACPP_INFERENCE_REDIS_TYPE: RedisType = RedisType::new(
    "lm_inferx",
    APP_VERSION,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(load_inference),
        rdb_save: Some(save_inference),
        aof_rewrite: None,
        free: Some(free_inference),

        // Currently unused by Redis
        mem_usage: Some(mem_usage_inference),
        digest: None,

        // Aux data
        aux_load: None,
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 0,

        copy: Some(copy_inference),
        free_effort: None,
        unlink: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        unlink2: None,
    },
);

unsafe extern "C" fn save_inference(_rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    let _m = unsafe { &*value.cast::<InferenceRedis>() };
    let name_cstring = CString::new(_m.name.as_str()).unwrap();
    raw::save_string(_rdb, name_cstring.to_str().unwrap());

    let name_cstring = CString::new(_m.model_name.as_str()).unwrap();
    raw::save_string(_rdb, name_cstring.to_str().unwrap());

    let name_cstring = CString::new(_m.prompt_name.as_str()).unwrap();
    raw::save_string(_rdb, name_cstring.to_str().unwrap());
}

unsafe extern "C" fn load_inference(_rdb: *mut raw::RedisModuleIO, encver: c_int) -> *mut c_void {
    match encver {
        0 => {
            let mut inference = Box::new(InferenceRedis::default());

            inference.name = RedisString::from_ptr(raw::RedisModule_LoadString.unwrap()(_rdb))
                .unwrap()
                .to_owned();

            inference.model_name =
                RedisString::from_ptr(raw::RedisModule_LoadString.unwrap()(_rdb))
                    .unwrap()
                    .to_owned();

            inference.prompt_name =
                RedisString::from_ptr(raw::RedisModule_LoadString.unwrap()(_rdb))
                    .unwrap()
                    .to_owned();

            println!("load inference {:?}", inference);
            let inference: *mut c_void = Box::into_raw(inference) as *mut c_void;
            inference
        }
        _ => ptr::null_mut() as *mut c_void,
    }
}

unsafe extern "C" fn free_inference(value: *mut c_void) {
    if value.is_null() {
        // on Redis 6.0 we might get a NULL value here, so we need to handle it.
        return;
    }
    drop(Box::from_raw(value as *mut InferenceRedis));
}

unsafe extern "C" fn mem_usage_inference(_value: *const c_void) -> usize {
    let m = Box::from_raw(_value as *mut InferenceRedis);
    m.name.len() as usize + m.model_name.len() as usize + m.prompt_name.len() as usize
}

unsafe extern "C" fn copy_inference(
    _: *mut raw::RedisModuleString,
    _: *mut raw::RedisModuleString,
    value: *const c_void,
) -> *mut c_void {
    let m = unsafe { &*value.cast::<InferenceRedis>() };
    let value = m.clone();
    Box::into_raw(Box::new(value)).cast::<c_void>()
}
