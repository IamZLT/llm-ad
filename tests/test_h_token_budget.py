import torch
import pytest
from models.h_token_budget import select_tokens,uniform_indices,pack_sparse_pair,greedy_decode_sparse

@pytest.mark.parametrize('hw,b', [((1,8),3),((8,1),5),((7,11),31),((4,4),16),((1,1),1)])
def test_uniform_exact(hw,b):
    idx=uniform_indices(hw,b)
    assert len(idx)==len(idx.unique())==b
    assert idx.min()>=0 and idx.max()<hw[0]*hw[1]
    assert torch.equal(idx,idx.sort().values)

def test_h_selection_is_spatial_and_keeps_global_coverage():
    h=torch.zeros(8,8);h[6:,6:]=1
    idx,stats=select_tokens(h,(8,8),16,'h',.5,0)
    assert set(uniform_indices((8,8),8).tolist())<=set(idx.tolist())
    assert {54,55,62,63}<=set(idx.tolist())
    assert len(idx.unique())==16

def test_flat_falls_back_and_full_identity():
    assert torch.equal(select_tokens(torch.zeros(8,8),(4,4),8)[0],uniform_indices((4,4),8))
    assert torch.equal(select_tokens(torch.rand(8,8),(4,4),8,'full')[0],torch.arange(16))

def test_sparse_qwen_positions_and_cached_decode():
    from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    from models.vision_cache import bind_cached_image_features
    torch.manual_seed(1)
    cfg=Qwen3_5Config(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,layer_types=['linear_attention','full_attention'],max_position_embeddings=128,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,
        patch_size=2,spatial_merge_size=2,temporal_patch_size=1),
        image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
    m=Qwen3_5ForConditionalGeneration(cfg).eval()
    ids=torch.tensor([[1,62]+[60]*4+[63,2,62]+[60]*4+[63,3,4]])
    batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=(ids==60).long(),
               image_grid_thw=torch.tensor([[1,4,4],[1,4,4]]),pixel_values=torch.zeros(32,12))
    merged=torch.randn(8,32);h=torch.arange(16.).reshape(4,4)
    full=pack_sparse_pair(m,batch,merged,h,'full')
    sparse=pack_sparse_pair(m,batch,merged,h,'h',.5)
    assert sparse['inputs_embeds'].shape[1]==ids.shape[1]-2
    assert torch.equal(sparse['position_ids'],full['position_ids'][:,:,sparse['sequence_indices']])
    assert sparse['next_position']==full['next_position']
    with torch.no_grad(),bind_cached_image_features(m,merged):
        ref=m(**batch,use_cache=False).logits
        got=m(inputs_embeds=full['inputs_embeds'],position_ids=full['position_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
        assert torch.allclose(ref,got,atol=1e-6)
        native=m.generate(**batch,max_new_tokens=3,do_sample=False,eos_token_id=None,pad_token_id=0)
    class Tok:
        eos_token_id=None
        def decode(self,ids,**kwargs):return str(ids)
    m.generation_config.eos_token_id=None
    explicit=greedy_decode_sparse(m,full,Tok(),3)
    assert explicit['ids']==native[0,ids.shape[1]:].tolist()
    # Sparse cached decoding must agree with repeated whole-prefix forwarding.
    generated=greedy_decode_sparse(m,sparse,Tok(),3)['ids']
    emb=sparse['inputs_embeds'];pos=sparse['position_ids']
    for i,token in enumerate(generated):
        with torch.no_grad():o=m(inputs_embeds=emb,position_ids=pos,attention_mask=torch.ones(1,emb.shape[1],dtype=torch.long),use_cache=False)
        assert token==int(o.logits[0,-1].argmax())
        emb=torch.cat([emb,m.get_input_embeddings()(torch.tensor([[token]]))],1)
        pos=torch.cat([pos,torch.full((3,1,1),sparse['next_position']+i,dtype=torch.long)],-1)
