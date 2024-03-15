use llama_cpp::LlamaModel;
use llama_cpp_sys::llama_model_default_params;
use redis_module::{native_types::RedisType, raw, RedisValue};
use serde::{Deserialize, Serialize};
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
#[serde(rename_all = "snake_case")]
#[serde(default)]
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
}

unsafe extern "C" fn load_model(_rdb: *mut raw::RedisModuleIO, encver: c_int) -> *mut c_void {
    match encver {
        //0 => {}
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
    "lm_promptx",
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
}

unsafe extern "C" fn load_prompt(_rdb: *mut raw::RedisModuleIO, encver: c_int) -> *mut c_void {
    match encver {
        //0 => {}
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
    "lm_promptx",
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
}

unsafe extern "C" fn load_inference(_rdb: *mut raw::RedisModuleIO, encver: c_int) -> *mut c_void {
    match encver {
        //0 => {}
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
