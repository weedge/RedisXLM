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

#include "llama2.h"

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache
        || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights* w, Config* p, void* ptr,
                        unsigned char shared_classifier) {
    float* fptr = (float*)ptr;  // cast our pointer to float*
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+
    // models
    unsigned long long n_layers = p->n_layers;
    // tok_embeddings
    w->token_embedding_table = fptr;
    // attention_norm
    fptr += p->vocab_size * p->dim;
    w->rms_att_weight = fptr;
    // attention.wq
    fptr += n_layers * p->dim;
    w->wq = fptr;
    // attention.wk
    fptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = fptr;
    // attention.wv
    fptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = fptr;
    // attention.wo
    fptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = fptr;
    // ffn_norm
    fptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = fptr;
    // feed_forward.w1
    fptr += n_layers * p->dim;
    w->w1 = fptr;
    // feed_forward.w2
    fptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = fptr;
    // feed_forward.w3
    fptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = fptr;
    // final rmsnorm
    fptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = fptr;

    // freqs_cos (skip)
    fptr += p->dim;
    fptr += p->seq_len * head_size / 2;  // skip what used to be freq_cis_real (for RoPE)
    fptr += p->seq_len * head_size / 2;  // skip what used to be freq_cis_imag (for RoPE)

    // final classifier weights if shared classifier
    w->wcls = shared_classifier ? w->token_embedding_table : fptr;
}

void read_checkpoint_v1(const char* checkpoint, Config* config, TransformerWeights* weights,
                        int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    // read in magic number (uint32), has to be 0x62616279, i.e. "baby" in ASCII
    // https://en.wikipedia.org/wiki/Magic_number_(programming)#In_files
    if (config->magic_number != 0x62616279) {
        fprintf(stderr, "Bad magic number\n");
        exit(EXIT_FAILURE);
    }
    // read in the version number (uint32), has to be 1
    if (config->version != 1) {
        fprintf(stderr, "Bad version %d, need version 2\n", config->version);
        exit(EXIT_FAILURE);
    }

    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END);  // move file pointer to end of file
    *file_size = ftell(file);  // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY);  // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    int header_size = 256;                             // the header size 256, left pad is \0
    void* weights_ptr = ((char*)*data) + header_size;  // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, config->shared_classifier);
}

void build_transformer(Transformer* t, const char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint_v1(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1) {
        close(t->fd);
    }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

// https://github.com/karpathy/llama2.c/issues/20 讨论了编译器选项，以及cpu向量化对推理的加速
// 可以结合gcc/clang 编译器的文档
// https://lemire.me/blog/2018/07/25/it-is-more-complicated-than-i-thought-mtune-march-in-gcc/

// https://github.com/bzhangGo/rmsnorm
// RMSNorm 根据均方根 (RMS)
// 对一层神经元的输入求和进行正则化，从而赋予模型重新缩放不变性和隐式学习率自适应能力。 RMSNorm
// 计算更简单，因此比 LayerNorm 更高效。
void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// https://en.wikipedia.org/wiki/Softmax_function
// Softmax的含义就在于不再唯一的确定某一个最大值，而是为每个输出分类的结果都赋予一个概率值，表示属于每个类别的可能性。
// :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`
// softmax函数的输出是0.0到1.0之间的实数
// 并且，softmax 函数的输出值的总和是1。输出总和为1是softmax函数的一个重要性质
// 正因为有了这个性质，我们才可以把softmax函数的输出解释为“概率”
// 使用了softmax函数，各个元素之间的大小关系也不会改变
void softmax(float* x, int size) {
    // find max value (for numerical stability)
    // 找到最大值，防止溢出
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// 相似度函数 点积 ,
// 输入 x 与 已训练的模型 linear层各 feature权重向量 的点积
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// transformer forward pass
float* forward(Transformer* transformer, int token, int pos) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;  // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    // forward all the layers
    // TransformerBlock * n_layers
    // attention all u need: https://arxiv.org/pdf/1706.03762.pdf
    // see detail
    for (unsigned long long l = 0; (int)l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // key and value point to the kv cache
        // https://medium.com/@joaolages/kv-caching-explained-276520203249
        // https://zhuanlan.zhihu.com/p/630832593
        int loff = l * p->seq_len * kv_dim;  // kv cache layer offset for convenience
        // concat
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        // use kv cache gemm(General Matrix Multiplication)变为gemv操作
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        // ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING:
        // https://arxiv.org/pdf/2104.09864v5.pdf
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;  // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // MHA: multihead attention. iterate over all heads -> GQA: Grouped-Query Attention(llama2)
        // MHA简单来说就是通过多次线性投影linear
        // projection得到原始输入的多个子空间，然后再每个子空间分别进行SDPA(Scaled Dot-Product
        // Attention), 再把SDPA的结果进行聚合Concatenation,最后再做一个linear projection。
        // SDPA的全称为Scaled Dot-Product Attention, 属于点积注意力机制，
        // 简单一句话来说就是，根据Query (Q)与Key之间的匹配度来对Value进行加权，而事实上不管是Query,
        // Key还是Value都来自于输入，因此所谓的SDPA本质上是对输入信息信息进行重组。
        // 通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。
        // self-attention 机制用于计算序列中当前token关注与其他token的联系，
        // 就是一个序列内的token，互相看其他token对自己的影响力有多大
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention with o_proj weights
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        // https://zh-v2.d2l.ai/chapter_convolutional-modern/resnet.html
        // https://en.wikipedia.org/wiki/Residual_neural_network
        // 在传统的神经网络中，每个层的输出都是通过对前一层输出的非线性变换得到的。但是，当网络的深度增加时，前一层的输出可能会被过度压缩或拉伸，导致信息丢失或重复。这种情况下，网络的性能可能会受到影响，同时也会出现梯度消失或梯度爆炸的问题。
        // 残差连接（residual
        // connection）是深度神经网络中的一种常见技术，它的作用是解决梯度消失和梯度爆炸问题，同时也可以帮助模型更快地收敛。
        // 残差连接通过在每个层的输出与输入之间添加一个跨层连接来解决这个问题。更具体地说，残差连接将前一层的输出直接添加到当前层的输出中，从而提供了一种绕过非线性变换的路径。这样，网络就可以学习到在信息压缩或拉伸后保留重要信息的方法，同时也减轻了梯度消失或梯度爆炸的问题。
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        // x为attention层的输出 与 训练好的权重 进行rmsnorm归一化缩放，
        // 隐藏状态写入s->xb中, s->xb作为 FFN 的输入
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // w1 is gate_projection
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        // w3 is up_projection
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        // GLU（Gated Linear
        // Units）其实不算是一种激活函数，而是一种神经网络层。它是一个线性变换后面接门控机制的结构。其中门控机制是一个sigmoid激活函数用来控制信息能够通过多少
        // https://en.wikipedia.org/wiki/Sigmoid_function
        // SwiGLU其实就是采用Swish作为激活函数的GLU变体, 加入了一个参数更加丝滑
        // https://en.wikipedia.org/wiki/Swish_function like sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // Sigmoid Linear Unit (SiLU)
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        // 输入 s->hb 与 权重 w2 down_projection 进行gemv,
        // 隐藏状态写入s->xb中, s->xb作为 残差连接 的输入
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        // 残差连接输出的x 作为下一层的输入
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    // 最终 n*TransformerBlock 处理的结果与训练好的权重进行rmsnorm归一化缩放，
    // 计算结果写入x中
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    // 输入 x 与 权重 wcls 进行 gemv,
    // 得到模型最终输出 logits (vocab_size) vector
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    // 神经网络只把输出值最大的神经元所对应的类别作为识别结果。并且，即便使用softmax函数，输出值最大的神经元的位置也不会变。因此，神经网络在进行推理分类时，输出层的softmax函数可以省略;
    // 节省运算量。（如果是学习训练阶段，需要softmax）
    return s->logits;
}

int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;  // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';  // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

// 使用分词器词表vocab对tokenid进行解码,返回对应tokenid在表中的字符串piece
char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void filter_piece(char* piece, int is_print) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) {
        piece = "";
        return;
    }
    if (piece[0] == '\0') {
        piece = "";
        return;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            piece = "";
            return;  // bad byte, don't print it
        }
    }
    if (is_print) {
        printf("%s", piece);
        fflush(stdout);
    }
}

int str_lookup(char* str, TokenIndex* sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str};  // acts as the key to search for
    TokenIndex* res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// 对文本进行编码 返回:
// tokens: token 序列 BOS token(=1) 对应文本的 tokenid EOS token(=2)
// n_tokens: token 序列长度
void encode(Tokenizer* t, const char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos)
        tokens[(*n_tokens)++] = 1;

    // 从已经加载的训练好的词表里获取单个词对应token id
    // 更多实现见 https://github.com/google/sentencepiece
    // https://arxiv.org/abs/1808.06226

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char* c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;  // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;  // protect against a sequence of stray UTF8 continuation bytes
    }

    // bpe 算法
    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;  // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;  // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos)
        tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// https://en.wikipedia.org/wiki/Cumulative_distribution_function
int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;  // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a;
    ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;  // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;  // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;  // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp,
                   unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) {  // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

// https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/
// https://colab.research.google.com/drive/14kMyF1nDvjP1mA86Yd1xkGCaqA-c4xVy?usp=sharing
int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next
                = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, const char* prompt,
             int steps, char* output) {
    char* empty_prompt = "";
    if (prompt == NULL) {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens
        = (int*)malloc((strlen(prompt) + 3) * sizeof(int));  // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        // fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        return ERRNO_NUM_PROMPT_TOKENS;
    }

    // start the main loop
    long start = 0;                // used to time our code, only initialized after first iteration
    int next;                      // will store the next token in the sequence
    int token = prompt_tokens[0];  // kick off with the first token in the prompt
    int pos = 0;                   // position in the sequence
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        filter_piece(piece, false);  // same as printf("%s", piece), but skips "unsafe" bytes
        strcat(output, piece);

        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = time_in_ms();
        }
    }
    // printf("\n");
    strcat(output, "\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stdout, "prompt: %s , generate achieved tok/s: %f\n", prompt,
                (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);

    return 0;
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';  // strip newline
        }
    }
}

void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
          const char* cli_user_prompt, const char* cli_system_prompt, int steps) {
    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1;  // user starts
    int next;              // will store the next token in the sequence
    int token;             // stores the current token to feed into the transformer
    // int prev_token;
    int pos = 0;  // position in the sequence
    while (pos < steps) {
        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt,
                               sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;  // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) {
            user_turn = 1;
        }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            filter_piece(piece, true);  // same as printf("%s", piece), but skips "unsafe" bytes
        }
        if (next == 2) {
            printf("\n");
        }
    }
    printf("\n");
    free(prompt_tokens);
}