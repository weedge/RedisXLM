use lazy_static::lazy_static;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams, SplitMode};
use num_cpus;
use rayon;
use redis_module::logging::log_debug;
use redis_module::{
    redis_module, Context, NextArg, RedisError, RedisResult, RedisString, RedisValue, Status,
    ThreadSafeContext,
};
use serde_json;
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;

//#[allow(dead_code, unused_variables, unused_mut)]
mod types;
use types::*;

static PREFIX: &str = "llamacpp";

// https://www.sitepoint.com/rust-global-variables/
static mut LLM_INFERENCE_POOL: Option<rayon::ThreadPool> = None;
lazy_static! {
    // just use init load args, then read it's args for cmd,,
    static ref MODULE_ARGS_MAP: RwLock<HashMap<String, String>> = {
        let m = HashMap::new();
        RwLock::new(m)
    };
}

// like google https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models defined models
// if tuned,pretrained model, just use https://cloud.google.com/vertex-ai/docs/start/explore-models
// LLAMACPP.CREATE_MODEL qwen1_5-0_5b-chat-q8_0.gguf --OPTS '{"model_path" : "$MODEL_PATH","model_type":"local_inference_llm","model_params":{}}'
// LLAMACPP.CREATE_MODEL qwen1_5-0_5b-chat-q8_0.gguf --OPTS '{"model_path" : "$MODEL_PATH","model_type":"local_inference_llm"}'
// LLAMACPP.CREATE_MODEL qwen1_5-1_8b-chat-q8_0.gguf --OPTS '{"model_path" : "$MODEL_PATH","model_type":"local_inference_llm"}'
// LLAMACPP.CREATE_MODEL qwen1_5-7b-chat-q8_0.gguf --OPTS '{"model_path" : "$MODEL_PATH","model_type":"local_inference_llm"}'
fn create_model(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();
    if args.len() < 4 {
        return Err(RedisError::WrongArity);
    }

    let mut args = args.into_iter().skip(1);
    let name = format!("{}.{}", PREFIX, args.next_str()?);
    let model_name = ctx.create_string(name.clone());

    if args.next_string()?.to_lowercase() != "--opts" {
        return Err(RedisError::WrongArity);
    }
    let model_opts_json_str = args.next_str()?;
    let model_opts: ModelOpts = serde_json::from_str(&model_opts_json_str)?;

    // get model redisType value
    let key = ctx.open_key_writable(&model_name);
    match key.get_value::<ModelRedis>(&LLAMACPP_MODEL_REDIS_TYPE)? {
        Some(_) => {
            return Err(RedisError::String(format!(
                "Model: {} already exists",
                &model_name
            )));
        }
        None => {
            // create llama model
            let mut params = LlamaParams::default();
            if model_opts.model_params.n_gpu_layers > 0 {
                params.n_gpu_layers = model_opts.model_params.n_gpu_layers;
            }
            if model_opts.model_params.split_mode == "layer" {
                params.split_mode = SplitMode::Layer;
            } else if model_opts.model_params.split_mode == "row" {
                params.split_mode = SplitMode::Row;
            }
            params.main_gpu = model_opts.model_params.main_gpu;
            params.use_mlock = model_opts.model_params.use_mlock;
            params.use_mmap = model_opts.model_params.use_mmap;
            params.vocab_only = model_opts.model_params.vocab_only;
            let res = LlamaModel::load_from_file(model_opts.clone().model_path, params);
            if res.is_err() {
                return Err(RedisError::String(format!(
                    "model {} load error",
                    model_name
                )));
            }
            let _model = res.unwrap();

            // create model redis type
            let mut redis_model = ModelRedis::default();
            redis_model.name = name;
            redis_model.model_opts = model_opts;
            redis_model.model = Some(Arc::new(_model));

            // set index redisType value
            ctx.log_debug(format!("create LlamaCPP Model {:?}", redis_model).as_str());
            key.set_value::<ModelRedis>(&LLAMACPP_MODEL_REDIS_TYPE, redis_model.into())?;
        }
    }

    Ok("OK".into())
}

// LLAMACPP.CREATE_PROMPT hello_promt "<|SYSTEM|>You are a helpful assistant." "<|USER|>Hello!" "<|ASSISTANT|>"
// LLAMACPP.CREATE_PROMPT assistant_promt_tpl "<|SYSTEM|>{{system}}." "<|USER|>{{user}}" "<|ASSISTANT|>"
// see openai prompt example: https://platform.openai.com/examples
fn create_promt(_ctx: &Context, _args: Vec<RedisString>) -> RedisResult {
    Ok(RedisValue::NoReply)
}

// LLAMACPP.CREATE_INFERENCE hello_world --model qwen --promt hello_promt
// LLAMACPP.CREATE_INFERENCE assistant_tpl --model qwen --promt assistant_promt_tpl
fn create_inference(_ctx: &Context, _args: Vec<RedisString>) -> RedisResult {
    Ok(RedisValue::NoReply)
}

fn start_completing_with(model: &str, out_put: &mut String) -> Option<RedisError> {
    let res = LlamaModel::load_from_file(model, LlamaParams::default());
    if res.is_err() {
        return Some(RedisError::String(format!("model {} load error", model)));
    }
    let _model = res.unwrap();
    let mut params = SessionParams::default();
    params.n_ctx = 2048;
    let res = _model.create_session(params);
    if res.is_err() {
        return Some(RedisError::String(format!(
            "model {} Failed to create session",
            model
        )));
    }
    let mut session = res.unwrap();

    session
        .advance_context("<|SYSTEM|>You are a helpful assistant.")
        .unwrap();
    session.advance_context("<|USER|>Hello!").unwrap();
    session.advance_context("<|ASSISTANT|>").unwrap();

    // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
    let max_tokens = 1024;
    let mut decoded_tokens = 0;

    // `session.start_completing_with` creates a worker thread that generates tokens. When the completion
    // handle is dropped, tokens stop generating!
    let completions = session
        .start_completing_with(StandardSampler::default(), max_tokens)
        .into_strings();

    for completion in completions {
        log_debug(format!("{completion}"));
        out_put.push_str(completion.as_str());
        decoded_tokens += 1;
        if decoded_tokens > max_tokens {
            break;
        }
    }

    return None;
}

// LLAMACPP.INFERENCE_CHAT hello_world
// LLAMACPP.INFERENCE_CHAT assistant_tpl --vars '{"system": "hello", "user": "world"}'
fn inference_chat(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.log_notice(format!("llm_inference_chat args:{:?}", args).as_str());
    if args.len() < 1 {
        return Err(RedisError::WrongArity);
    }
    let mut args = args.into_iter().skip(1);
    let model = args.next_str()?;
    log_debug(format!("Loading model: {}", model));

    let blocked_client = ctx.block_client();
    unsafe {
        LLM_INFERENCE_POOL
            .borrow_mut()
            .as_ref()
            .unwrap()
            .spawn(move || {
                log_debug(format!(
                    "Task executes on thread: {:?}",
                    thread::current().id()
                ));
                let thread_ctx = ThreadSafeContext::with_blocked_client(blocked_client);
                let mut out_put = String::new();
                let res = start_completing_with(model, &mut out_put);
                if res.is_some() {
                    thread_ctx.reply(Err(res.unwrap()));
                    return;
                }
                // todo: use stream chat, need redis stream (init chat pipline by client with limit_num)
                thread_ctx.reply(Ok(out_put.into()));
            });
    }

    Ok(RedisValue::NoReply)
}

//////////////////////////////////////////////////////

redis_module! {
    name: "redisxlm-llamacpp",
    version: 1,
    allocator: (redis_module::alloc::RedisAlloc, redis_module::alloc::RedisAlloc),
    data_types: [],
    init: init,
    commands: [
        [format!("{}.create_model", PREFIX), create_model, "", 0, 0, 0],
        [format!("{}.create_prompt", PREFIX), create_promt, "", 0, 0, 0],
        [format!("{}.create_inference", PREFIX), create_inference, "", 0, 0, 0],
        [format!("{}.inference_chat", PREFIX), inference_chat, "", 0, 0, 0],
    ],
}

fn init(ctx: &Context, args: &[RedisString]) -> Status {
    if args.len() % 2 != 0 {
        ctx.log_warning(
            format!(
                "module arguments len {}, must be key:value pairs",
                args.len()
            )
            .as_str(),
        );
        return Status::Err;
    }

    for i in (0..args.len()).step_by(2) {
        MODULE_ARGS_MAP.write().unwrap().insert(
            args[i].to_string_lossy().to_string(),
            args[i + 1].to_string_lossy().to_string(),
        );
    }
    ctx.log_debug(&format!("args_map: {:?}", MODULE_ARGS_MAP.read().unwrap()).as_str());

    let mut thread_num = num_cpus::get();
    let args_map = MODULE_ARGS_MAP.read().unwrap();
    if args_map.contains_key(&"llm_inference_threads".to_string()) {
        thread_num = args_map
            .get(&"llm_inference_threads".to_string())
            .unwrap()
            .parse()
            .unwrap();
    }
    ctx.log_debug(&format!("thread_num {}", thread_num));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_num)
        .build()
        .unwrap();
    unsafe {
        *LLM_INFERENCE_POOL.borrow_mut() = Some(pool);
    }

    Status::Ok
}
