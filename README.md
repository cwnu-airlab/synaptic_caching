## Synaptic Caching

Synaptic Caching은 대규모 언어 모델(LLM)의 효율성을 극대화하기 위한 기술로, 모델이 한 번 생각한 내용을 뇌(메모리)에 저장해두고, 똑같은 질문이나 문맥이 나오면 다시 생각하지 않고 바로 꺼내 쓰는 기술을 말합니다.   
AI 모델도 입력을 처리할 때 내부의 수많은 파라미터(가중치)를 거치며 Activation States를 만들어냅니다.    
Synaptic Caching은 이 중간 연산 결과를 저장해 두는 것입니다. 마치 뇌가 이미 처리한 정보를 시냅스에 잠시 유지하는 것과 같습니다.   


기존 방식과 비교하면 다음과 같습니다.
- 캐싱이 없을 때 (비효율적)
   - 사용자가 긴 문서를 주고 질문을 합니다.
   - 모델은 문서를 처음부터 끝까지 토큰화하고 연산합니다.
   - 사용자가 추가 질문을 합니다.
   - 모델은 아까 읽은 문서를 처음부터 다시 연산해야 합니다. (중복 연산 발생)
- Synaptic Caching 적용 시
   - 사용자가 긴 문서를 줍니다.
   - 모델이 연산을 수행하면서, 각 토큰에 대한 Key-Value(KV) 상태값을 캐시(Cache) 메모리에 저장합니다.
   - 사용자가 추가 질문을 합니다.
   - 모델은 문서를 다시 읽지 않고, 저장된 캐시 데이터(KV Cache)를 그대로 가져와서 답변만 생성합니다.

KV Cache 관리(Attention Sink)를 위하여 다음 연구를 참조합니다.
- https://github.com/mit-han-lab/streaming-llm
- 해당 연구를 참조하여 '무한한 대화'를 위해 기억을 효율적으로 관리하는 기술을 접목합니다.
- 핵심 개념
   - KV Cache Slicing
   ```python
   import torch
   
   def perform_streaming_llm_eviction(past_key_values, sink_size=4, window_size=1020):
       """
       StreamingLLM의 핵심 로직: KV Cache가 꽉 차면 중간을 버림.
       
       Args:
           past_key_values: 현재 저장된 KV Cache (Tuple of tensors)
           sink_size: 절대 지우지 않을 앞부분 토큰 개수 (Attention Sink, 보통 4)
           window_size: 유지할 최근 대화 토큰 개수 (Rolling Window)
       """
       
       # 1. Key와 Value를 각각 가져옴 (Layer별로 처리됨)
       # shape: [batch_size, num_heads, seq_len, head_dim]
       keys, values = past_key_values 
       
       current_seq_len = keys.shape[2]
       max_capacity = sink_size + window_size
   
       # 2. 캐시가 허용 용량보다 작으면 아무것도 안 함 (Pass)
       if current_seq_len <= max_capacity:
           return past_key_values
   
       # 3. [핵심] 캐시 자르기 (Eviction Logic)
       # -------------------------------------------------------
       # A. Attention Sink: 맨 앞의 'sink_size'만큼 잘라냄 (절대 보존)
       # 예: keys[:, :, :4, :]
       sink_keys = keys[:, :, :sink_size, :]
       sink_values = values[:, :, :sink_size, :]
   
       # B. Rolling Window: 맨 뒤의 'window_size'만큼 잘라냄 (최신 대화)
       # 예: keys[:, :, -1020:, :]
       recent_keys = keys[:, :, -window_size:, :]
       recent_values = values[:, :, -window_size:, :]
   
       # C. 중간 삭제 후 결합 (Concatenate)
       # [Sink] + [Recent] 형태로 다시 붙임
       new_keys = torch.cat([sink_keys, recent_keys], dim=2)
       new_values = torch.cat([sink_values, recent_values], dim=2)
       # -------------------------------------------------------
   
       return (new_keys, new_values)
  ```
  - Positional Shift
  ```python
  def apply_rotary_pos_emb_streaming(x, cos, sin, position_ids):
      # 캐시가 잘려나갔으므로, 모델이 '캐시 내에서의 위치'를 알 수 있도록
      # position_ids를 조정하는 로직이 추가됩니다.
      # (실제 구현에서는 position_ids가 캐시 길이를 넘지 않도록 조정)
      
      # 일반적인 구현이 아닌 StreamingLLM용 패치 부분:
      # 캐시에 있는 토큰들의 위치 정보(cache_position)를 재정렬합니다.
      pass
  ```


- 실행 방법은 다음과 같습니다.
``` shell
CUDA_VISIBLE_DEVICES=3 python longbench_streaming_w_EMA_RoPE_regen.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \\
					 --enable_start_recent_kv_cache --start_size 160 --recent_use no --compress gaus
```
