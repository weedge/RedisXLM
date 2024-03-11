use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use redis_module::raw::Version;
use redis_module::{redis_module, Context, NextArg, RedisError, RedisResult, RedisString};
use std::io;
use std::io::Write;
use std::os::raw::c_int;

fn version_from_info_string(info_str: String) -> Result<Version, RedisError> {
    let regex = regex::Regex::new(
        r"(?m)\bredis_version:(?<major>[0-9]+)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)\b",
    );

    if regex.is_ok() {
        let regex = regex.unwrap();
        let mut it = regex.captures_iter(info_str.as_str());
        let res = it.next();
        if res.is_none() {
            return Err(RedisError::Str("Error getting redis_version"));
        }
        let caps = res.unwrap();
        return Ok(Version {
            major: caps["major"].parse::<c_int>().unwrap(),
            minor: caps["minor"].parse::<c_int>().unwrap(),
            patch: caps["patch"].parse::<c_int>().unwrap(),
        });
    }
    Err(RedisError::Str("Error getting redis_version"))
}

fn test_redis_info_ver(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.log_notice(format!("{:?}", args).as_str());
    let s = "# Server redis_version:7.2.1 redis_git_sha1:00000000 redis_git_dirty:0 redis_build_id:7b8617dd94058f85 redis_mode:standalone os:Darwin 22.6.0 x86_64 arch_bits:64 monotonic_clock:POSIX clock_gettime multiplexing_api:kqueue atomicvar_api:c11-builtin gcc_version:4.2.1 process_id:72033 process_super".to_string();
    let ver_str: Vec<&str> = s.split(|c| c == ' ').collect();
    let mut s = "";
    for item in ver_str.iter() {
        if item.contains("redis_version:") {
            s = item;
            break;
        }
    }
    //println!("{s:?}");
    //let s = ver_str.as_slice()[2];
    //let s = "# Server redis_version:7.2.1".to_string();
    let ver = version_from_info_string(s.to_string())?;
    let response: Vec<i64> = vec![ver.major.into(), ver.minor.into(), ver.patch.into()];

    return Ok(response.into());
}

fn test_completions(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.log_notice(format!("{:?}", args).as_str());
    if args.len() < 1 {
        return Err(RedisError::WrongArity);
    }
    let mut args = args.into_iter().skip(1);
    let model = args.next_str()?;
    println!("Loading model: {}", model);
    let res = LlamaModel::load_from_file(model, LlamaParams::default());
    if res.is_err() {
        return Err(RedisError::String(format!("model {} load error", model)));
    }
    let _model = res.unwrap();
    let mut params = SessionParams::default();
    params.n_ctx = 2048;
    let res = _model.create_session(params);
    if res.is_err() {
        return Err(RedisError::String(format!(
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
        print!("{completion}");
        let _ = io::stdout().flush();
        decoded_tokens += 1;
        if decoded_tokens > max_tokens {
            break;
        }
    }
    /*
    // todo: async block inference use
    // rust redis module don't support async fn to tokio select!
    let timeout_by = Instant::now() + Duration::from_secs(500);
    loop {
        select! {
            _ = tokio::time::sleep_until(timeout_by) => {
                break;
            }
            completion = <TokensToStrings<CompletionHandle> as StreamExt>::next(&mut completions) => {
                if let Some(completion) = completion {
                    print!("{completion}");
                    let _ = io::stdout().flush();
                } else {
                    break;
                }
                continue;
            }
        }
    } */

    return Ok(format!("preload model {} ok", model).into());
}

#[cfg(not(test))]
macro_rules! get_allocator {
    () => {
        redis_module::alloc::RedisAlloc
    };
}

#[cfg(test)]
macro_rules! get_allocator {
    () => {
        std::alloc::System
    };
}

redis_module! {
    name: "demo_llamacpp",
    version: 1,
    allocator: (get_allocator!(), get_allocator!()),
    data_types: [],
    commands: [
        ["version_test", test_redis_info_ver, "", 0, 0, 0],
        ["completions_test", test_completions, "", 0, 0, 0],
    ],
}
