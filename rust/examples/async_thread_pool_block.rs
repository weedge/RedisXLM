use lazy_static::lazy_static;
use num_cpus;
use rayon;
use redis_module::{
    redis_module, Context, RedisResult, RedisString, RedisValue, Status, ThreadSafeContext,
};
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::sync::RwLock;
use std::thread;
use std::time::Duration;
//use tokio;

// https://www.sitepoint.com/rust-global-variables/
static mut LLM_INFERENCE_POOL: Option<rayon::ThreadPool> = None;
lazy_static! {
    // just use init load args, then read it's args for cmd,,
    static ref MODULE_ARGS_MAP: RwLock<HashMap<String, String>> = {
        let m = HashMap::new();
        RwLock::new(m)
    };
}

fn block(ctx: &Context, _args: Vec<RedisString>) -> RedisResult {
    let blocked_client = ctx.block_client();

    unsafe {
        LLM_INFERENCE_POOL
            .borrow_mut()
            .as_ref()
            .unwrap()
            .spawn(move || {
                println!("Task executes on thread: {:?}", thread::current().id());
                let thread_ctx = ThreadSafeContext::with_blocked_client(blocked_client);
                thread::sleep(Duration::from_millis(3000));
                thread_ctx.reply(Ok("rayon thread pool async block".into()));
            });
    }

    /*
    thread::spawn(move || {
        let thread_ctx = ThreadSafeContext::with_blocked_client(blocked_client);
        thread::sleep(Duration::from_millis(10000));
        thread_ctx.reply(Ok("thread async block".into()));
    });
    */

    /*
    tokio::spawn(async {
        let thread_ctx = ThreadSafeContext::with_blocked_client(blocked_client);
        tokio::time::sleep(Duration::from_millis(10000)).await;
        thread_ctx.reply(Ok("tokio async block with runtime scheduler".into()));
    });
     */
    // We will reply later, from the thread
    Ok(RedisValue::NoReply)
}

//////////////////////////////////////////////////////

redis_module! {
    name: "block",
    version: 1,
    allocator: (redis_module::alloc::RedisAlloc, redis_module::alloc::RedisAlloc),
    data_types: [],
    init: init,
    commands: [
        ["block", block, "", 0, 0, 0],
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
