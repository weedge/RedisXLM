use lazy_static::lazy_static;
use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};
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
use std::ops::Deref;
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
    let name = format!("{}.m.{}", PREFIX, args.next_str()?);
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
                name
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
                return Err(RedisError::String(format!("model {} load error", name)));
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

// LLAMACPP.CREATE_PROMPT hello_prompt "<|SYSTEM|>You are a helpful assistant." "<|USER|>Hello!" "<|ASSISTANT|>"
// LLAMACPP.CREATE_PROMPT assistant_prompt_tpl "<|SYSTEM|>{{system}}." "<|USER|>{{user}}" "<|ASSISTANT|>"
// see openai prompt example: https://platform.openai.com/examples
fn create_prompt(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }
    let mut args = args.into_iter().skip(1);
    let name = format!("{}.p.{}", PREFIX, args.next_str()?);
    let prompt_name = ctx.create_string(name.clone());

    let key = ctx.open_key_writable(&prompt_name);
    match key.get_value::<ModelRedis>(&LLAMACPP_PROMPT_REDIS_TYPE)? {
        Some(_) => {
            return Err(RedisError::String(format!(
                "prompt: {} already exists",
                name
            )));
        }
        None => {
            let mut prompts = Vec::new();
            while let Ok(p) = args.next_string() {
                prompts.push(p);
            }
            let mut redis_prompt = PromptRedis::default();
            redis_prompt.name = name;
            redis_prompt.prompts = prompts;
            // set index redisType value
            ctx.log_debug(format!("create LlamaCPP Prompt {:?}", redis_prompt).as_str());
            key.set_value::<PromptRedis>(&LLAMACPP_PROMPT_REDIS_TYPE, redis_prompt.into())?;
        }
    }

    Ok("OK".into())
}

// LLAMACPP.CREATE_INFERENCE hello_world_infer --model qwen1_5-0_5b-chat-q8_0.gguf --prompt hello_prompt
// LLAMACPP.CREATE_INFERENCE assistant_tpl_infer --model qwen1_5-0_5b-chat-q8_0.gguf --prompt assistant_prompt_tpl
fn create_inference(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();
    if args.len() < 6 {
        return Err(RedisError::WrongArity);
    }
    let mut args = args.into_iter().skip(1);
    let name = format!("{}.p.{}", PREFIX, args.next_str()?);
    let model_name = format!("{}.p.{}", PREFIX, args.next_str()?);
    let prompt_name = format!("{}.p.{}", PREFIX, args.next_str()?);
    let inference_name = ctx.create_string(name.clone());

    let key = ctx.open_key_writable(&inference_name);
    match key.get_value::<InferenceRedis>(&LLAMACPP_INFERENCE_REDIS_TYPE)? {
        Some(_) => {
            return Err(RedisError::String(format!(
                "prompt: {} already exists",
                name
            )));
        }
        None => {
            let mut redis_inference = InferenceRedis::default();
            redis_inference.name = name;
            redis_inference.model_name = model_name;
            redis_inference.prompt_name = prompt_name;
            // set index redisType value
            ctx.log_debug(format!("create LlamaCPP Inference {:?}", redis_inference).as_str());
            key.set_value::<InferenceRedis>(
                &LLAMACPP_INFERENCE_REDIS_TYPE,
                redis_inference.into(),
            )?;
        }
    }

    Ok("OK".into())
}

fn start_completing_with(
    model_name: &str,
    _model: &LlamaModel,
    _prompts: &Vec<String>,
    _sample_params: &SampleParams,
    out_put: &mut String,
) -> Option<RedisError> {
    // use default session params
    // todo: create session redis type
    let mut params = SessionParams::default();
    params.n_ctx = 2048;
    let res = _model.create_session(params);
    if res.is_err() {
        return Some(RedisError::String(format!(
            "model {:#} Failed to create session",
            model_name
        )));
    }
    let mut session = res.unwrap();

    // prompt
    for p_ctx in _prompts {
        session.advance_context(p_ctx).unwrap();
    }

    let mut stages = Vec::<SamplerStage>::new();
    stages.push(SamplerStage::RepetitionPenalty {
        repetition_penalty: _sample_params.repetition_penalty.repetition_penalty,
        frequency_penalty: _sample_params.repetition_penalty.frequency_penalty,
        presence_penalty: _sample_params.repetition_penalty.presence_penalty,
        last_n: _sample_params.repetition_penalty.last_n,
    });
    stages.push(SamplerStage::Temperature(_sample_params.temperature));
    stages.push(SamplerStage::TopK(_sample_params.top_k));
    stages.push(SamplerStage::TopP(_sample_params.top_p));
    stages.push(SamplerStage::MinP(_sample_params.min_p));
    let mut sampler = StandardSampler::default();
    if _sample_params.token_selector == "softmax" {
        sampler = StandardSampler::new_softmax(stages, _sample_params.min_keep);
    } else if _sample_params.token_selector == "greedy" {
        sampler = StandardSampler::new_greedy();
    }
    // `session.start_completing_with` creates a worker thread that generates tokens. When the completion
    // handle is dropped, tokens stop generating!
    let completions = session
        .start_completing_with(sampler, _sample_params.max_tokens)
        .into_strings();

    let mut decoded_tokens = 0;
    for completion in completions {
        log_debug(format!("{completion}"));
        out_put.push_str(completion.as_str());
        decoded_tokens += 1;
        if decoded_tokens > _sample_params.max_tokens {
            break;
        }
    }

    return None;
}

// LLAMACPP.INFERENCE_CHAT hello_world_infer
// LLAMACPP.INFERENCE_CHAT hello_world_infer --sample_params '{"top_k":40,"top_p": 9.5,"temperature":0.8","min-p": 0.1}'
// LLAMACPP.INFERENCE_CHAT assistant_tpl_infer --vars '{"system": "hello", "user": "world"}'
// LLAMACPP.INFERENCE_CHAT assistant_tpl_infer --vars '{"system": "hello", "user": "world"}' --sample_params '{"top_k":40,"top_p": 9.5,"temperature":0.8","min_p": 0.1,"max_tokens":1024}'
/*
 --top-k N             top-k sampling (default: 40, 0 = disabled)
 --top-p N             top-p sampling (default: 0.9, 1.0 = disabled)
 --min-p N             min-p sampling (default: 0.1, 0.0 = disabled)
 --temp N              temperature (default: 0.8)
 --max_tokens(--n-predict) N  number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
*/
fn inference_chat(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    let mut args = args.into_iter().skip(1);
    let inference_name = format!("{}.p.{}", PREFIX, args.next_str()?);
    let mut tpl_vars = HashMap::<String, String>::default();
    let mut is_tpl_vars = false;
    let mut sample_params = SampleParams::default();
    while let Ok(p) = args.next_string() {
        if p.to_lowercase() == "--vars" {
            let tpl_vars_json_str = args.next_str()?;
            tpl_vars = serde_json::from_str(&tpl_vars_json_str)?;
            is_tpl_vars = true;
        }
        if p.to_lowercase() == "--sample_params" {
            let sample_json_str = args.next_str()?;
            sample_params = serde_json::from_str(&sample_json_str)?;
        }
    }

    // check params
    if !vec!["greed", "softmax"].contains(&sample_params.token_selector.as_str()) {
        return Err(RedisError::String(format!(
            "sampler token_selector {:#} don't support ",
            sample_params.token_selector
        )));
    }

    let redis_inference_name = ctx.create_string(inference_name.clone());
    let key = ctx.open_key_writable(&redis_inference_name);
    match key.get_value::<InferenceRedis>(&LLAMACPP_INFERENCE_REDIS_TYPE)? {
        None => {
            return Err(RedisError::String(format!(
                "inference: {} does not exists",
                inference_name
            )));
        }
        Some(val) => {
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
                        let ctx = thread_ctx.lock();
                        let prompt_redis = ctx.create_string(val.prompt_name.clone());
                        let prompt_key = ctx.open_key(&prompt_redis);
                        let prompt_res =
                            prompt_key.get_value::<PromptRedis>(&LLAMACPP_PROMPT_REDIS_TYPE);
                        let model_redis = ctx.create_string(val.model_name.clone());
                        let model_key = ctx.open_key(&model_redis);
                        let model_res =
                            model_key.get_value::<ModelRedis>(&LLAMACPP_MODEL_REDIS_TYPE);
                        drop(ctx);

                        if prompt_res.is_err() {
                            thread_ctx.reply(Err(prompt_res.err().unwrap()));
                            return;
                        }
                        let prompt_val = prompt_res.unwrap().unwrap();
                        let mut promts: Vec<String> = vec![];
                        for prompt in &prompt_val.prompts {
                            let mut p_str = prompt.clone();
                            if is_tpl_vars {
                                // replace var
                                for (k, v) in &tpl_vars {
                                    let key_str = format!("{{{k}}}");
                                    p_str = p_str.replace(key_str.as_str(), v);
                                }
                            }
                            promts.push(p_str);
                        }

                        if model_res.is_err() {
                            thread_ctx.reply(Err(model_res.err().unwrap()));
                            return;
                        }
                        let model_val = model_res.unwrap().unwrap();
                        let llama_model = model_val.clone().model.unwrap();

                        let mut out_put = String::new();
                        let res = start_completing_with(
                            model_val.name.as_str(),
                            &llama_model.deref(),
                            &promts,
                            &sample_params,
                            &mut out_put,
                        );
                        if res.is_some() {
                            thread_ctx.reply(Err(res.unwrap()));
                            return;
                        }
                        // todo: use stream chat, need redis stream (init chat pipline by client with limit_num)
                        thread_ctx.reply(Ok(out_put.into()));
                    });
            }
        }
    }

    Ok(RedisValue::NoReply)
}

fn start_completing(model: &str, out_put: &mut String) -> Option<RedisError> {
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

fn inference_chat_test(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.log_debug(format!("llm_inference_chat_test args:{:?}", args).as_str());
    if args.len() < 2 {
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
                let res = start_completing(model, &mut out_put);
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
    data_types: [LLAMACPP_MODEL_REDIS_TYPE,LLAMACPP_PROMPT_REDIS_TYPE,LLAMACPP_INFERENCE_REDIS_TYPE],
    init: init,
    commands: [
        [format!("{}.create_model", PREFIX), create_model, "", 0, 0, 0],
        [format!("{}.create_prompt", PREFIX), create_prompt, "", 0, 0, 0],
        [format!("{}.create_inference", PREFIX), create_inference, "", 0, 0, 0],
        [format!("{}.inference_chat", PREFIX), inference_chat, "", 0, 0, 0],
        [format!("{}.inference_chat_test", PREFIX), inference_chat_test, "", 0, 0, 0],
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
