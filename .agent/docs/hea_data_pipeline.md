# HEA 私有数据 → SteelBERT 回归预测 完整处理方案

> 数据源：`datasets/hea-composition-performance.xlsx` (64 条 CoCrNi 高熵合金)
> 目标：复用 SteelBERT 的预训练知识，预测屈服强度 / 抗拉强度 / 延伸率

---

## 数据概览

| 维度 | 详情 |
|------|------|
| 样本数 | 64 条 |
| 主元素 | Co, Cr, Ni, Al, Ti, Ta (at%) |
| 其他元素 | W, Mo, C, Nb, Fe, Cu, Mn, Hf, V, Zr, B 等 (部分样本) |
| 相结构 | FCC, FCC+L12, FCC+L12+η, FCC+η, BCC, BCC+Laves, FCC+B2, FCC+FCC |
| 热处理 | 单段/多段时效、冷轧+时效、as-cast 等 |
| 预测目标 | 室温 YS/UTS/EL + 高温 YS/UTS/EL (部分有缺失) |

---

## 第一步：成分对齐 (Composition Alignment)

### 问题
- 原数据以 `at%` 为主，部分行有 `wt%`，部分行两者都有或都缺
- `其他元素` 列以字符串形式存储（如 `"W:1.76, Mo:1.52, C:0.10, Nb:1.31"`）
- SteelBERT 的 `add_ele_embed` 需要**所有元素列为数值型**

### 处理方案

```python
import pandas as pd
import re

df = pd.read_excel('datasets/hea-composition-performance.xlsx')

# 1. 解析 "其他元素" 列 → 独立的数值列
def parse_other_elements(s):
    """解析 'W:1.76, Mo:1.52' → {'W': 1.76, 'Mo': 1.52}"""
    if pd.isna(s) or s.strip() == '-':
        return {}
    pairs = re.findall(r'([A-Z][a-z]?)\s*:\s*([\d.]+)', s)
    return {elem: float(val) for elem, val in pairs}

other_parsed = df['其他元素'].apply(parse_other_elements)
other_df = pd.DataFrame(other_parsed.tolist()).fillna(0)

# 2. 合并主元素 + 其他元素，统一为 at%
main_elements = ['Co', 'Cr', 'Ni', 'Al', 'Ti', 'Ta']
for col in main_elements:
    df[col] = pd.to_numeric(df[f'{col}(at%)'].replace('-', 0), errors='coerce').fillna(0)

composition = pd.concat([df[main_elements], other_df], axis=1).fillna(0)

# 3. 确保所有列为 float 类型
composition = composition.astype(float)
```

### 关键决策：at% vs wt%
- **建议使用 at%**：因为你的大部分数据都只有 at%
- SteelBERT 原始训练用 wt%，但 `add_ele_embed` 只是把百分比当作**加权系数**
- 只要同一数据集内**统一使用同一单位**，对加权平均的语义质量影响不大

---

## 第二步：语义合成 (LLM-based Text Generation)

### 目标
将 `相结构 + 热处理状态 + 测试温度` 合成一段 SteelBERT 能高效编码的学术英文描述。

### LLM 调用规范

| 参数 | 设定 |
|------|------|
| 模型 | GPT-4o / Claude / 本地 Llama-3 (8B) |
| Temperature | **0.0** (确保可复现) |
| 调用方式 | **离线批处理**，一次生成全部 64 条 |
| 缓存 | 结果保存至 `datasets/hea_with_text.csv`，后续训练直接读取 |

### Prompt 模板

```python
SYSTEM_PROMPT = """You are a materials scientist. Convert alloy metadata into 
one concise academic English sentence. Rules:
1. Only describe facts given — do NOT infer or predict properties
2. Use standard metallurgical terminology
3. Include: alloy system, phase structure, heat treatment details, test temperature
4. Output exactly one sentence per alloy"""

def build_user_prompt(row):
    elements = []
    for el in ['Co','Cr','Ni','Al','Ti','Ta']:
        val = row.get(f'{el}(at%)', 0)
        if val and val != '-' and float(val) > 0:
            elements.append(f"{el}:{val}at%")
    
    other = row.get('其他元素', '-')
    if other and other != '-':
        elements.append(other)
    
    return f"""Alloy: {', '.join(elements)}
Phase structure: {row['相结构']}
Heat treatment: {row['热处理状态']}
Test temperature: {row.get('高温温度(°C)', 'Room temperature')}°C"""
```

### 调用示例 (OpenAI API)

```python
import openai

client = openai.OpenAI(api_key="YOUR_KEY")

def generate_text(row):
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)}
        ]
    )
    return resp.choices[0].message.content.strip()

# 批量生成并缓存
texts = []
for i, row in df.iterrows():
    text = generate_text(row)
    texts.append(text)
    print(f"[{i+1}/64] {row['合金名称']}: {text[:80]}...")

df['Text'] = texts
df.to_csv('datasets/hea_with_text.csv', index=False)
```

### 预期输出示例

| 合金 | 生成的 Text |
|------|-------------|
| AT00 | "This equiatomic CoCrNi medium-entropy alloy with a single FCC phase was aged at 850°C for 10 hours and tested at 700°C." |
| AT22 | "This Co31.97Cr31.97Ni31.97Al2.04Ti2.04 high-entropy alloy with FCC and L12 dual-phase structure was aged at 850°C for 10 hours and tested at 700°C." |

### 备选方案：纯模板拼接（无需 LLM，零成本）

```python
def template_text(row):
    phase = row['相结构'] if row['相结构'] != '-' else 'unknown'
    ht = row['热处理状态'] if row['热处理状态'] != '-' else 'as-received'
    temp = row.get('高温温度(°C)', 'room temperature')
    return (f"This alloy has {phase} phase structure, "
            f"heat treated at {ht}, tested at {temp} degrees Celsius.")

df['Text'] = df.apply(template_text, axis=1)
```

> **对比**：LLM 版本语义密度更高（能触发更多 SteelBERT 注意力），但模板版本**零噪声、完全可控**。建议先用模板版做 baseline，再用 LLM 版做对比实验。

---

## 第三步：特征提取 (SteelBERT Embedding)

复用 `reg_v1.py` 中的现有函数：

```python
from reg_v1 import get_embeddinngs, get_ele_embeddinngs, add_ele_embed

# 1. 文本特征 (768d) — 来自 gen_text_embed
#    对 Text 列的每一行调用 get_embeddinngs → [CLS] 向量
text_embeds = df['Text'].apply(get_embeddinngs)  # shape: (64, 768)

# 2. 成分特征 (768d) — 来自 add_ele_embed
#    对各元素符号调用 SteelBERT，按 at% 加权平均
comp_embeds = add_ele_embed(composition_df)  # shape: (64, 768)

# 3. Concatenate → 1536d 输入 MLP
features = concat([comp_embeds, text_embeds], dim=1)
```

---

## 第四步：模型适配与训练 (Adaptation)

### 架构选择
64 条数据 → **极简架构**，防止过拟合

```python
# 推荐配置
BEST_CONFIG_HEA = {
    'simple_layer_list': [128, 64],     # 仅 2 层，极窄
    'cnn_start': 32,
    'dropout': 0.2,                     # 较高 dropout
    'lr': 3e-4,
    'batch_size': 16,                   # 小 batch
    'epoch': 300,
    'patience': 30,
}
```

### 训练策略
- **SteelBERT 完全冻结**（64 条数据无法支撑解冻 1.88 亿参数）
- **多种子训练**（至少 5 个 seed），取 Val R² 最高的模型
- **数据划分**：8:2 (约 51 训练 + 13 验证)
- **Loss**：MSELoss
- **评估**：关注 Train-Val Gap，**Gap > 10% 即需减参数**

### 预测目标选择
由于部分属性有缺失值，建议分别训练：

| 目标 | 有效样本数 | 可行性 |
|------|-----------|--------|
| 高温屈服强度 | 52 | ✅ 首选 |
| 高温延伸率 | 50 | ✅ 可行 |
| 室温屈服强度 | 44 | ✅ 可行 |
| 室温延伸率 | 44 | ✅ 可行 |
| 室温抗拉强度 | 32 | ⚠️ 样本偏少 |
| 高温抗拉强度 | 32 | ⚠️ 样本偏少 |

---

## 风险与注意事项

1. **领域偏移**：SteelBERT 主要在"钢铁"语料上训练，对 HEA 术语（如 L12、η 相）的语义覆盖可能不完整。建议在 Text 中使用 SteelBERT 更熟悉的同义表达（如 "ordered intermetallic precipitates" 代替 "L12"）
2. **数据量极小**：64 条数据训练深度学习模型风险很高。如果效果不理想，可考虑：
   - 传统 ML 方法 (XGBoost/SVR) 作为 baseline 对比
   - 将 SteelBERT embedding 作为额外特征，喂给传统 ML
3. **at% vs wt%**：确保全数据集统一单位，不可混用
