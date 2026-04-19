#!/usr/bin/env python
"""
验证 SteelBERT 权重加载和推理是否正常 (GPU 环境)
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_PATH = "/internfs/Zy/Steelllm/ckpt/SteelBERT"

def test_config():
    """1. 验证 config.json 是否可读"""
    print("[1/5] 加载 config.json ...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    print(f"  ✓ 模型类型: {config.model_type}")
    print(f"  ✓ Hidden size: {config.hidden_size}")
    print(f"  ✓ 层数: {config.num_hidden_layers}")
    print(f"  ✓ 注意力头数: {config.num_attention_heads}")
    print(f"  ✓ 词表大小: {config.vocab_size}")
    return config

def test_tokenizer():
    """2. 验证分词器是否可用"""
    print("\n[2/5] 加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    test_text = "316L stainless steel with high tensile strength"
    tokens = tokenizer(test_text, return_tensors="pt")
    decoded = tokenizer.decode(tokens["input_ids"][0])
    print(f"  ✓ 词表大小: {tokenizer.vocab_size}")
    print(f"  ✓ 测试文本: '{test_text}'")
    print(f"  ✓ Token 数量: {len(tokens['input_ids'][0])}")
    print(f"  ✓ 解码还原: '{decoded}'")
    return tokenizer

def test_model_load():
    """3. 验证模型权重能否加载到 GPU"""
    print("\n[3/5] 加载模型到 GPU ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_PATH).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 设备: {device}")
    print(f"  ✓ 参数量: {param_count / 1e6:.1f}M")
    print(f"  ✓ 模型类: {model.__class__.__name__}")
    return model, device

def test_inference(model, tokenizer, device):
    """4. 验证推理是否正常"""
    print("\n[4/5] 执行推理测试 ...")
    texts = [
        "A composite steel plate for marine construction was fabricated using 316L stainless steel.",
        "The yield strength of the alloy reached 1200 MPa after quenching and tempering.",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    print(f"  ✓ 输入: {len(texts)} 条文本")
    print(f"  ✓ CLS 嵌入维度: {cls_embeddings.shape}")
    print(f"  ✓ 嵌入范围: [{cls_embeddings.min().item():.4f}, {cls_embeddings.max().item():.4f}]")
    return cls_embeddings

def test_gpu_memory():
    """5. GPU 显存使用情况"""
    print("\n[5/5] GPU 显存统计 ...")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  ✓ 已分配: {allocated:.0f} MB")
        print(f"  ✓ 已保留: {reserved:.0f} MB")
        print(f"  ✓ 总显存: {total:.0f} MB")
        print(f"  ✓ 剩余可用: {total - reserved:.0f} MB")

if __name__ == "__main__":
    print("=" * 60)
    print("SteelBERT 权重加载 & 推理验证")
    print("=" * 60)
    
    try:
        config = test_config()
        tokenizer = test_tokenizer()
        model, device = test_model_load()
        embeddings = test_inference(model, tokenizer, device)
        test_gpu_memory()
        
        print("\n" + "=" * 60)
        print("✅ 所有验证全部通过！模型可以正常使用。")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
