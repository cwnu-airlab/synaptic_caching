import torch
import json
import math
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.utils import parse_args, load

device='cuda'
args = parse_args()

templates = json.load(open("streaming_llm/dataset2prompt.json", 'r'))
compress = json.load(open("streaming_llm/dataset2compress.json", 'r'))

model, tokenizer = load(args.model_name_or_path)
max_gen_len = 1000 
max_context_len = model.config.max_position_embeddings

recent_use = args.recent_use
chunk_size = 819 #int(max_context_len * 0.15)
cache_size = 6144 #int(max_context_len * 0.440)
trigger_size = 5324 #int(max_context_len * 0.350)
k_count = chunk_size * 4 
k_decod = chunk_size * 6 

output_filepath = f"results/LongBench/RoPE/streaming_{args.compress}_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_kE{k_count}_kD{k_decod}_recent{args.recent_use}"
print(f"Confirm: {chunk_size}, {cache_size}, {trigger_size} \nFile path: {output_filepath} \n(y/n)>>>", end='')
ans = input()
if ans != 'y' : exit()

os.makedirs(output_filepath, exist_ok=True)

#datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
#            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
#            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

datasets = [ "triviaqa", "musique", "hotpotqa", "2wikimqa", "qasper", 'samsum', 'dureader', 'lsht', 'gov_report']

data = dict()
for name in datasets:
    temp = load_dataset('zai-org/LongBench', name, split='test')
    data[name] = [x for x in temp]

attn_score = None
if args.enable_start_recent_kv_cache: 
    kv_cache = enable_streaming_llm(
            model, recent_use=recent_use, cache_size=cache_size, start_size=args.start_size, recent_size=args.recent_size, compress=args.compress)
else:
    kv_cache = None


if args.compress == 'gaus' :
    for key in compress :
        compress[key] = 'gaussian'
elif args.compress == 'ema' :
    for key in compress :
        compress[key] = 'moving_average'
elif args.compress == 'peak' :
    for key in compress :
        compress[key] = 'peak_finding'
else :
    for key in compress :
        compress[key] = 'top_k'

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

for name in datasets :
    if os.path.exists(f"{output_filepath}/{name}.jsonl"): 
        with open(f"{output_filepath}/{name}.jsonl", "r") as f:
            exist_len = [json.loads(d) for d in f]
        exist_len = len(exist_len)
        if exist_len == len(data[name]) :
            print(f"Skip {name}")
            continue
    else :
        exist_len = 0 
    
    with open(f"{output_filepath}/{name}.jsonl", 'a') as f :
        comp_way = compress[name]
        for i, item in enumerate(tqdm(data[name][exist_len:])):
            result = list() 
            prompt = templates[name]
            inputs = prompt.format(**item)
            
            """
            <Encoding Process>

            Input Sequence가 일정 비율에 맞게 Chunk 단위로 쪼개어져 KV Cache로 저장됨 
            """

            encodings = tokenizer.encode(inputs, add_special_tokens=True, return_tensors='pt').to(device)
            past_key_values = None   # KV Cache
            past_positions = None    # KV Cache의 절대 Position ID
            seq_len = encodings.shape[-1]

            chunks = encodings.split(chunk_size, dim=1)
            flg = 0
            
            for chunk_idx in range(len(chunks)):
                if chunk_idx == len(chunks)-1: continue   # 마지막 토큰은 Encoding에 포함하지 않음
                if flg == 1: continue                     # Memory Out Err 발생 시 Encoding 중단 

                chunk = chunks[chunk_idx]
                chunk_len = chunk.shape[-1]
                
                if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:   # Threshold를 넘으면 압축
                    past_key_values, past_positions = kv_cache.evict_for_space_analysis(past_key_values, attn_score, comp_way, k_count, past_positions)
                
                # 현재 Chunk의 절대 위치 계산
                if past_positions is None: start_pos = 0    
                else: start_pos = past_positions[0][-1].detach().item() + 1       # 마지막 위치 +1부터 시작 
                current_pos = torch.arange(start_pos, start_pos + chunk_len, device=device).unsqueeze(0) # [1, seq_len]
               
                # ⭐️ 현재 position_ids에 KV Cache의 절대 Position ID를 함께 붙여 전달 
                if past_positions is None : position_ids = None
                else : 
                    position_ids = torch.cat([past_positions, current_pos], dim=-1)
                
                with torch.no_grad():
                    #try:
                    # Self/Cross Attention 수행 및 KV Cache 업데이트 
                    c_output = model(input_ids=chunk.to(device), past_key_values=past_key_values, position_ids=position_ids, 
                                output_attentions=True, use_cache=True)
#                    except:
#                        print(chunks[chunk_idx].shape)
#                        print(f"Memory Out Err in encoding process: {chunk_idx}/{len(chunks)} ==> {encodings.shape} / {past_key_values[0][0].shape}")
#                        result.append(f"Memory Out Err in encoding process: {chunk_idx}/{len(chunks)} ==> {encodings.shape} / {past_key_values[0][0].shape}")
#                        flg = 1 
#                        continue
                    
                    past_key_values = c_output.past_key_values
                    
                    if chunk_idx == 0: past_positions = current_pos.clone()
                    else: past_positions = torch.cat([past_positions, current_pos], dim=1) 
                   
                    attn_score = list(c_output.attentions)
          
            """
            <Decoding Process>

            마지막 Chunk와 KV Cache 간 Cross Attention으로 토큰 생성 시작 
            토큰 하나 생성 시마다 treshold 초과 여부를 확인하고 압축 진행
            """

            if flg != 1 :
                encodings = chunks[-1]
                
                if past_key_values == None :
                    kv_len = 0
                else : kv_len = past_key_values[0][0].shape[-2]

                start_pos = past_positions[0][-1].detach().item()+1 
                current_pos = torch.arange(start_pos, start_pos + encodings.shape[-1], device=device).unsqueeze(0)
                position_ids = torch.cat([past_positions, current_pos], dim=-1)
                
                # ⭐️ Position_ids 크기가 Input Sequence 길이의 배수가 되도록 조정 
                gap = (math.floor(position_ids.shape[-1] / encodings.shape[-1])+1) * encodings.shape[-1] - position_ids.shape[-1]
                if gap > 0 :
                    padding = torch.full((1, gap), -999, device=device)
                    position_ids = torch.cat([position_ids, padding], dim=-1)
                    

                seq_len = encodings.shape[1]
                with torch.no_grad() :
                    try :
                        outputs = model(input_ids=encodings, past_key_values=past_key_values, position_ids=position_ids, use_cache=True)
                    except :
                        past_key_values, past_positions = kv_cache.evict_for_space_analysis(past_key_values, attn_score, 
                                comp_way, k_decod, past_positions)
                        kv_len = past_key_values[0][0].shape

                        start_pos = past_positions[0][-1].detach().item()+1
                        current_pos = torch.arange(start_pos, start_pos + encodings.shape[-1], device=device).unsqueeze(0)
                        position_ids = torch.cat([past_positions, current_pos], dim=-1)
                        
                        outputs = model(input_ids=encodings, past_key_values=past_key_values, position_ids=position_ids, use_cache=True)
                    
                    past_key_values = outputs.past_key_values
                    past_positions = torch.cat([past_positions, current_pos], dim=1)

                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_ids = [pred_token_idx.item()]
                    pos = 0
                    
                    for _  in range(max_gen_len - 1):
                        start_pos = past_positions[0][-1].detach().item()+1
                        current_pos = torch.arange(start_pos, start_pos+pred_token_idx.shape[-1], device=device).unsqueeze(0)
                        position_ids = torch.cat([past_positions, current_pos], dim=-1)

                        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, position_ids=position_ids, 
                                output_attentions=True,use_cache=True)
                        
                        past_key_values = outputs.past_key_values
                        past_positions = torch.cat([past_positions, current_pos], dim=-1)

                        attn_score = list(outputs.attentions)

                        if kv_cache is not None and past_key_values[0][0].shape[-2] >= trigger_size:
                            past_key_values, past_positions = kv_cache.evict_for_space_analysis(past_key_values, attn_score, 
                                    comp_way, k_decod, past_positions)

                        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        generated_ids.append(pred_token_idx.item())
                        generated_text = (
                                tokenizer.decode(generated_ids, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True, spaces_between_special_tokens=False).strip().split(" "))
                        
                        now = len(generated_text) - 1
                        if now > pos :
                            result.append(generated_text[pos:now][-1])
                            pos = now
                        if pred_token_idx == 128009:
                            break
            

                try: 
                    result.append(generated_text[pos:][-1])
                except :
                    result.append("")
            
            output_sentence = " ".join(result)
            print(output_sentence)
            exit()
            for_save = {
                    "input": item['input'], "pred": output_sentence, 'label': item['answers'], 
                    "kv_size": f"{kv_len}", 
                    'length': item['length'], 'dataset': item['dataset'], 'language': item['language'], 
                    'all_classes': item['all_classes'], '_id': item['_id']}
            f.write(json.dumps(for_save, ensure_ascii=False)+'\n')


f.close()
print(f"Saved to {output_filepath}")
