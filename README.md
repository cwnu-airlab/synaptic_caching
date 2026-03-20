# Synaptic Caching — KV Cache Eviction

LLM 추론 시 KV Cache의 메모리 사용량을 제어하기 위한 eviction 전략입니다.

긴 시퀀스를 처리할 때 KV Cache가 GPU 메모리를 초과하지 않도록, 중요도가 낮은 토큰의 캐시를 선택적으로 eviction합니다. Sink 토큰(문맥 초반)과 Recent 토큰(최근 생성)은 항상 보존하고, 중간 영역은 전략에 따라 uniform sampling하거나 전부 제거합니다.

## 아키텍처

```
kv_eviction/
├── __init__.py          # 패키지 진입점, 전체 public API re-export
├── eviction.py          # 핵심 eviction 로직 (torch only, transformers 의존 없음)
├── streaming.py         # 청크 기반 streaming prefill + perplexity 평가 엔진
├── rope_patch.py        # RoPE monkey-patch (eviction 후 position 보정)
├── model_utils.py       # 모델/토크나이저 로딩 유틸리티
└── visualization.py     # eviction 패턴 heatmap 생성
```

### 모듈별 역할

**eviction.py** — 핵심 모듈. `SimpleEvictConfig`로 eviction 전략을 정의하고, `build_simple_keep_token_idx()`가 보존할 토큰 인덱스를 계산합니다. `evict_dynamic_cache_inplace()`는 HuggingFace `DynamicCache`의 K/V 텐서를 직접 슬라이싱하여 메모리를 해제합니다. transformers 의존 없이 torch만으로 동작합니다.

**streaming.py** — 긴 시퀀스를 chunk 단위로 나눠 모델에 입력하고, cache가 target을 초과하면 eviction을 수행합니다. 생성 태스크용 `streaming_prefill()`과 perplexity 평가용 `streaming_ppl()`, greedy 디코딩용 `greedy_decode()`를 제공합니다.

**rope_patch.py** — Eviction 후 살아남은 토큰들의 position이 불연속이 되는 문제를 해결합니다. Attention 레이어를 monkey-patch하여 raw(미회전) K를 캐시에 저장하고, attention 시점에 연속적인 slot position `[0..kv_len-1]`으로 RoPE를 재적용합니다. Llama, Qwen2의 FlashAttention2 레이어를 지원합니다.

**model_utils.py** — HuggingFace 모델과 토크나이저를 로드합니다. Flash Attention 2를 우선 시도하고, 불가능하면 standard attention으로 fallback합니다.

**visualization.py** — eviction 패턴을 heatmap으로 시각화합니다. 어떤 토큰이 보존/제거되었는지를 레이어×포지션 격자로 보여줍니다.

## Eviction 전략

캐시를 세 영역으로 분할하여 eviction을 수행합니다:

```
[0 ............. sink_end)       → SINK   (항상 보존)
[sink_end ... recent_start)      → MIDDLE (전략에 따라 처리)
[recent_start ...... total_len)  → RECENT (항상 보존)
```

### sink_recent

중간 영역을 전부 제거합니다. 가장 공격적인 전략으로 메모리 절약이 극대화되지만, 중간 문맥 정보를 모두 잃습니다.

```python
build_eviction_config(strategy="sink_recent", sink_tokens=256, recent_tokens=512)
```

### sink_recent_uniform

중간 영역에서 균일 간격으로 블록을 샘플링하여 보존합니다. 블록 크기는 Flash Attention의 블록 크기(128)에 맞춰져 있습니다. 두 가지 방식으로 중간 영역의 보존량을 제어할 수 있습니다:

- **budget 방식** — 보존할 중간 토큰 수를 직접 지정합니다. 지정된 budget을 블록 단위로 환산하여 `torch.linspace`로 균일하게 블록을 선택합니다.

```python
build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256, recent_tokens=512,
    middle_budget=256,   # 중간 영역에서 256토큰 분량의 블록을 균일 보존
)
```

- **stride 방식** — 매 N번째 블록을 보존합니다. budget보다 우선 적용됩니다.

```python
build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256, recent_tokens=512,
    uniform_stride=4,   # 중간 영역에서 4블록마다 1블록 보존
)
```

### RoPE 모드

eviction 후 position encoding 처리 방식을 선택할 수 있습니다:

- **abs** (기본값) — 원래 absolute position을 유지합니다. eviction 후 position에 gap이 생기지만 추가 패치 없이 동작합니다.
- **raw_rel** — `patch_model_raw_kv()`로 모델을 패치하여, 캐시에 미회전 K를 저장하고 attention 시 연속 position으로 RoPE를 재적용합니다. Position gap 문제를 해결하여 품질 저하를 줄입니다.
