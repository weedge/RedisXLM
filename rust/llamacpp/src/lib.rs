use lazy_static::lazy_static;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use num_cpus;
use rayon;
use redis_module::logging::log_debug;
use redis_module::{
    redis_module, Context, NextArg, RedisError, RedisResult, RedisString, RedisValue, Status,
    ThreadSafeContext,
};
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::sync::RwLock;
use std::thread;

// https://www.sitepoint.com/rust-global-variables/
static mut LLM_INFERENCE_POOL: Option<rayon::ThreadPool> = None;
lazy_static! {
    // just use init load args, then read it's args for cmd,,
    static ref MODULE_ARGS_MAP: RwLock<HashMap<String, String>> = {
        let m = HashMap::new();
        RwLock::new(m)
    };
}

// LLAMACPP.CREATE_MODEL qwen1_5-0_5b-chat-q8_0.gguf --TYPE local_inference_llm --PARAMS '{"model_path" : "$MODEL_PATH"}'
// LLAMACPP.CREATE_MODEL qwen1_5-0_5b-chat-q8_0.gguf --TYPE local_inference_llm --PARAMS '{"model_path" : "$MODEL_PATH"}'
// LLAMACPP.CREATE_MODEL qwen1_5-1_8b-chat-q8_0.gguf --TYPE local_inference_llm --PARAMS '{"model_path" : "$MODEL_PATH"}'
// LLAMACPP.CREATE_MODEL qwen1_5-7b-chat-q8_0.gguf --TYPE local_inference_llm --PARAMS '{"model_path" : "$MODEL_PATH"}'
fn create_model(_ctx: &Context, _args: Vec<RedisString>) -> RedisResult {
    Ok(RedisValue::NoReply)
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
        ["llamacpp.create_model", create_model, "", 0, 0, 0],
        ["llamacpp.create_prompt", create_promt, "", 0, 0, 0],
        ["llamacpp.create_inference", create_inference, "", 0, 0, 0],
        ["llamacpp.inference_chat", inference_chat, "", 0, 0, 0],
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
