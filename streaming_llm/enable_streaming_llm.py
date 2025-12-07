from streaming_llm.kv_cache_choice import StartRecentKVCacheChoice

def enable_streaming_llm(model, recent_use, compress, cache_size, start_size, recent_size):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama_shift import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
    

    kv_cache = StartRecentKVCacheChoice(
                    cache_size=cache_size, 
                    start_size=start_size, 
                    recent_size=recent_size, 
                    k_seq_dim=k_seq_dim, 
                    v_seq_dim=v_seq_dim, 
                    )
    return kv_cache
