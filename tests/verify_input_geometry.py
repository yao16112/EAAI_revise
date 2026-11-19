"""CPU-only geometry audit: actual trainer transforms and actual DINO patch embedding.

Stop before the first Transformer block; no model training or GPU allocation.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.VarAD.trainer import AnomalyMambaTrainer
from model.VarAD.tokenizer_backbones.dino.model import DINO
from config import DATASET_ROOTS


class TransformConstructionStub(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.placeholder=nn.Parameter(torch.zeros(1))


class StopBeforeTransformer(Exception):
    pass


def verify(output):
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    # Execute the current entry-point alignment expression, not a replacement formula.
    tree=ast.parse((ROOT/'main.py').read_text())
    expressions=[node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                         and t.value.id=='args' and t.attr=='image_size' for t in node.targets)]
    assert len(expressions)==1
    align=compile(ast.Expression(expressions[0]),str(ROOT/'main.py'),'eval')
    from types import SimpleNamespace
    real_path=next((Path(DATASET_ROOTS['mvtec'])/'bottle/train/good').glob('*.png'))
    with Image.open(real_path) as image:real_image=image.convert('RGB')
    dino=DINO('dinov2_vitb14').eval().cpu()
    assert dino.dino.patch_size==14 and tuple(dino.dino.patch_embed.patch_size)==(14,14)
    results=[]
    cases=[('resize_1022',1022,False,real_image,8),
           ('cli_1024',1024,False,real_image,8),
           ('resize_1036',1036,False,real_image,8),
           ('native_1024',1024,True,Image.new('RGB',(1024,1024),(100,100,100)),1),
           ('native_real_bottle',1024,True,real_image,1)]
    for name,requested,native,image,batch in cases:
        effective=requested if native else eval(align,{'args':SimpleNamespace(image_size=requested)})
        with patch('model.VarAD.trainer.AnomalyMamba',TransformConstructionStub):
            transform_owner=AnomalyMambaTrainer(image_size=effective,native_res=native,device='cpu')
        tensor=transform_owner.data_transform(image)
        x=tensor.unsqueeze(0).repeat(batch,1,1,1)
        record={'case':name,'requested_image_size':requested,'effective_image_size':effective,
                'native_res':native,'source_hw':[image.height,image.width],
                'transform':str(transform_owner.data_transform),'tensor_shape':list(x.shape),
                'evidence':'CPU_ACTUAL_TRANSFORM_AND_PATCH_EMBED_NO_TRANSFORMER_OR_TRAINING'}
        def patch_input(module,args):record['dino_patch_embed_input_shape']=list(args[0].shape)
        def patch_output(module,args,result):record['patch_embedding_shape']=list(result.shape)
        def first_block(module,args):
            record['first_transformer_input_shape']=list(args[0].shape)
            raise StopBeforeTransformer()
        handles=[dino.dino.patch_embed.register_forward_pre_hook(patch_input),
                 dino.dino.patch_embed.register_forward_hook(patch_output),
                 dino.dino.blocks[0].register_forward_pre_hook(first_block)]
        try:
            with torch.no_grad():dino(x)
        except StopBeforeTransformer:pass
        else:raise AssertionError('Expected stop before Transformer')
        finally:
            for handle in handles:handle.remove()
        h,w=x.shape[-2:];record['token_grid']=[h//14,w//14]
        assert record['patch_embedding_shape']==[batch,(h//14)*(w//14),768]
        assert record['first_transformer_input_shape']==[batch,(h//14)*(w//14)+1,768]
        if name=='native_1024':
            assert x.shape[-2:]==(1036,1036)
            assert torch.all(x[:,:,-12:,:]==0) and torch.all(x[:,:,:,-12:]==0)
        results.append(record)
    try:
        with torch.no_grad():dino(torch.zeros(1,3,1024,1024))
    except AssertionError as exc:raw_error=str(exc)
    else:raise AssertionError('Raw 1024 should fail divisibility assertion')
    assert not torch.cuda.is_initialized()
    result={'status':'PASS','python':sys.executable,'torch':str(torch.__version__),
            'cuda_initialized':torch.cuda.is_initialized(),'patch_size':14,
            'real_image':str(real_path),'cases':results,'raw_1024_error':raw_error,
            'batch8_full_gpu_training':'NOT_TESTED_TO_PRESERVE_RUNNING_PILOT'}
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    verify(parser.parse_args().output)
