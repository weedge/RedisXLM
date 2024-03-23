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

/* Inference for Llama-2 Transformer model in pure C */
// changed from: https://github.com/karpathy/llama2.c/blob/master/run.c
// model from my baby_llm: https://github.com/weedge/baby-llm to
// inference(generate/chat(instruction-tuned model))

#ifndef LLAMA2_H
#define LLAMA2_H

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

// -------------------------------------------------
// defined vars

// boolean
#define true 1
#define false 0

// errno
#define ERRNO_NUM_PROMPT_TOKENS 1
#define ERR_STR_NUM_PROMPT_TOKENS "something is wrong, expected at least 1 prompt token"

// ----------------------------------------------------------------------------
// Transformer model

// https://docs.python.org/3/library/struct.html#format-characters
// trained model.bin use struct.pack save, u need c-type to aligment read
// if config is change, u need increase version to upgrade,
// maybe define new config struct
typedef struct {
    unsigned int magic_number;  // magic number
    int version;                // version
    int dim;                    // transformer dimension
    int hidden_dim;             // for ffn layers
    int n_layers;               // number of layers
    int n_heads;                // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
    unsigned char shared_classifier;  // shared classifier
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;  // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight;  // (layer, dim) rmsnorm weights
    float* rms_ffn_weight;  // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq;  // (layer, dim, n_heads * head_size)
    float* wk;  // (layer, dim, n_kv_heads * head_size)
    float* wv;  // (layer, dim, n_kv_heads * head_size)
    float* wo;  // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1;  // (layer, hidden_dim, dim)
    float* w2;  // (layer, dim, hidden_dim)
    float* w3;  // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight;  // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float* x;       // activation at current time stamp (dim,)
    float* xb;      // same, but inside a residual branch (dim,)
    float* xb2;     // an additional buffer just for convenience (dim,)
    float* hb;      // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q;       // query (dim,)
    float* k;       // key (dim,)
    float* v;       // value (dim,)
    float* att;     // buffer for scores/attention values (n_heads, seq_len)
    float* logits;  // output logits
    // kv cache
    float* key_cache;    // (layer, seq_len, dim)
    float* value_cache;  // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config;               // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights;  // the weights of the model
    RunState state;              // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;             // file descriptor for memory mapping
    float* data;        // memory mapped data pointer
    ssize_t file_size;  // size of the checkpoint file in bytes
} Transformer;

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
// https://leimao.github.io/blog/Byte-Pair-Encoding/

typedef struct {
    char* str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];  // stores all single-byte strings
} Tokenizer;

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex;  // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex;  // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// -----------------------------------------------------------------------------
// declare api function

/**
 * @brief
 * construct transformer inference model
 *
 * @param t
 * @param checkpoint_path
 */
void build_transformer(Transformer* t, const char* checkpoint_path);

/**
 * @brief
 * construct tokenizer
 *
 * @param t
 * @param tokenizer_path
 * @param vocab_size
 */
void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);

/**
 * @brief
 * construct sampler
 *
 * @param sampler
 * @param vocab_size
 * @param temperature
 * @param topp
 * @param rng_seed
 */
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp,
                   unsigned long long rng_seed);

/**
 * @brief
 * generation loop
 * https://huggingface.co/blog/how-to-generate
 * https://colab.research.google.com/drive/14kMyF1nDvjP1mA86Yd1xkGCaqA-c4xVy?usp=sharing
 *
 * @param transformer
 * @param tokenizer
 * @param sampler
 * @param prompt
 * @param steps
 * @param output
 * @return int 0 is ok, else err_no
 */
int generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, const char* prompt,
             int steps, char* output);

/**
 * @brief
 * free sampler
 *
 * @param sampler
 */
void free_sampler(Sampler* sampler);

/**
 * @brief
 * free tokenizer
 *
 * @param t
 */
void free_tokenizer(Tokenizer* t);

/**
 * @brief
 * free transformer
 *
 * @param t
 */
void free_transformer(Transformer* t);

#endif
