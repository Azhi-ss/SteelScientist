# Supplemental Material

**Steel design based on a large language model**

Shaohan Tian$^{a}$, Xue Jiang$^{a,b,*}$, Weiren Wang$^{a}$, Zhihua Jing$^{a}$, Chi Zhang$^{a}$, Cheng Zhang$^{a}$, Turab Lookman$^{b,*}$, Yanjing Su$^{a,*}$

$^{a}$ *Beijing Advanced Innovation Center for Materials Genome Engineering, Institute for Advanced Materials and Technology, University of Science and Technology Beijing, Beijing 100083, China*
$^{b}$ *Liaoning Academy of Materials, Shenyang, 110000, Liaoning, China*
$^*$ *Corresponding author: E-mail: jiangxue@ustb.edu.cn, turablookman@gmail.com, yjsu@ustb.edu.cn*

-----

## Supplementary Figures

### Figure S1. Directed graph illustrating the diversity of processing actions and parameters in steels.

We identified 36 processing actions (after normalization) and 135 parameters from approximately 55,000 research papers on steels. By calculating the binary occurrence frequencies between different actions, we were able to represent these relationships in the graph. Each node in the figure represents a processing action, and the width of the arrow indicates its frequency, with broader lines denoting more frequently occurring pairs of actions. On average, each action is associated with three parameters, and the typical processing sequence involves five actions. Based on this analysis, we can infer that there is a vast potential space of processing routes, estimated to be no less than $(36 \times 3 \times 100)^5 \approx 1.5 \times 10^{20}$. It’s important to note, however, that actual steel manufacturing is far more intricate and encompasses a wide spectrum of knowledge.

> **【图片详细文字解析：图 S1】**
> 本图是一个复杂的**有向图（网络拓扑图）**，展示了钢铁领域海量的工艺动作和参数组合。
>
>   * **节点（Nodes）：** 图中包含数十个圆形的节点，每个节点代表一种归一化后的“加工动作（Processing action）”，例如：`heat`（加热）、`solution`（固溶）、`quench`（淬火）、`roll`（轧制）、`temper`（回火）、`anneal`（退火）、`cool`（冷却）、`forge`（锻造）等。在每个节点旁边，列出了该动作关联的典型工艺参数（如：温度、时间、冷却速率、压下量等）。
>   * **连线（Edges/Arrows）：** 节点之间由带箭头的连线相连，表示工艺的先后顺序。连线的粗细代表了这两种动作在55000篇文献中连续出现的频率——连线越粗，说明这两种工艺（如“热轧”后接“淬火”）在实际和研究中越常被搭配使用。
>   * **整体视觉：** 这是一个高度密集的网络，直观地证明了钢铁加工路线的排列组合空间极其庞大（论文估算至少有 $1.5 \times 10^{20}$ 种），解释了传统表格数据难以处理工艺序列的原因。

<br>

### Figure S2. Network architecture optimization of the prediction model for yield strength, ultimate tensile strength and elongation.

Each point represents a distinct network architecture variant tailored for mechanical property prediction. The optimization process involved fine-tuning various architectural parameters, including the number of layers, number of nodes, dropout rates, activation functions, pooling methods and the number of Convolutional neural network channels were optimized. The model’s performance significantly improved as we continuously adjusted and refined the network architecture.

> **【图片详细文字解析：图 S2】**
> 本图包含三个并排的**散点图（a, b, c）**，分别展示了网络架构优化过程中，模型对屈服强度 (YS)、极限抗拉强度 (UTS) 和延伸率 (EL) 的预测表现。
>
>   * **坐标轴：** X轴为训练集的决定系数（Training $R^2$），Y轴为验证集的决定系数（Validation $R^2$），坐标范围从 -1.0 到 1.0。
>   * **数据点：** 图中布满了红色的散点，每一个点代表一组特定的神经网络架构超参数配置（如不同的层数、节点数、Dropout率等）。
>   * **参考线：** 图中有一条虚线表示 $Y = X$（即训练集得分等于验证集得分）。
>   * **图形结论：** 大量的红色散点紧密地分布在这条 $Y = X$ 的虚线周围，并向右上角（$R^2$ 接近 1.0 的高分区域）聚集。这表明通过 Optuna 自动调参，模型没有出现严重的过拟合（否则点会偏向右下角），并且能够找到泛化能力极佳的架构。

<br>

### Figure S3. Euclidean distance matrix heatmap of integer numbers’ embeddings with different LLMs, covering the range from 0 to 2000.

(a) BERT model. (b) SciBERT model. (c) MatSciBert model. (d) SteelBERT model.

> **【图片详细文字解析：图 S3】**
> 本图展示了四个**热力图（Heatmaps，a, b, c, d）**，比较了四种不同语言模型对 0-2000 之间整数进行 Embedding 编码后的欧氏距离矩阵。
>
>   * **矩阵结构：** 坐标轴 X 和 Y 均代表数字 0 到 2000。热力图的颜色深浅代表两个数字向量之间的欧氏距离。
>   * **对角线：** 所有图的主对角线（左上到右下）都是深黑色，因为数字与自身的距离为 0。
>   * **模型对比：** \>   \* 图 (a) BERT、(b) SciBERT 和 (c) MatSciBERT 的热力图呈现出大量不规则的色块和条纹（甚至有十字形的亮带），这说明它们对数字大小的语义理解是混乱的，没有平滑的距离过渡。
>       * 图 (d) SteelBERT 呈现出非常均匀且有规律的色彩渐变——沿着对角线向外，颜色平滑地从暗变亮。这证明 SteelBERT 在向量空间中完美保留了数字大小递增的数学逻辑。

<br>

### Figure S4. Evaluation of number decoding task on LLMs.

A probing model to decode a number from its word embedding across a randomly selected 80% subset of integers within the range from 0 to 2000, exemplified by the conversion of “100” to 100.0.

> **【图片详细文字解析：图 S4】**
> 这是一个自下而上的**流程示意图（Flowchart）**，展示了模型如何将文字形态的数字解码回数学数值。
>
>   * **最下方：** 输入字符串字面量（例如文本形式的 `"9"` 或 `"100"`）。
>   * **向上的箭头 1：** 经过 `Model Embedding`（模型嵌入层），将字符串转化为高维向量。
>   * **向上的箭头 2：** 高维向量输入到一个 `MLP`（多层感知机 / 探测模型）中。
>   * **向上的箭头 3：** MLP 输出一个回归预测的浮点数值（例如框中显示的 `8.95`）。
>   * **最上方：** 最终映射或对齐为实际的连续数值目标（例如框中显示的 `9.0`）。

<br>

### Figure S5. Evaluation of the fine-tuning results and traditional machine learning models.

Our model shows superior prediction performance across all mechanical properties, achieving an $R^2$ of 89.85% (±6.17%), 88.34% (±5.95%) and 87.24% (±5.15%) for YS, UTS and EL, respectively.

> **【图片详细文字解析：图 S5】**
> 本图是一个**分组柱状图（Grouped Bar Chart）**，对比了微调后的深度学习模型与传统机器学习模型在实验数据集上的表现。
>
>   * **X 轴：** 列出了不同的预测模型，包括：GBR, KRR, MLP, SVR, XGB, RF，以及最后的 `Ft-model`（微调后的 SteelBERT 模型）。
>   * **Y 轴：** 为预测决定系数 $R^2$ (%)，范围从 20 到 120。
>   * **柱状图：** 每个模型都有三根并排的柱子，分别代表对 UTS（蓝色）、YS（橙色）和 EL（粉色）的预测 $R^2$ 分数。每根柱子上带有误差棒（Error bars）。
>   * **图形结论：** 大部分传统模型（如 MLP, SVR, KRR）的柱子较低且误差棒极大（表现出极大的不稳定性）。而最右侧的 `Ft-model` 的三根柱子最高（均接近 90%）且误差棒最小，视觉上直观地确立了微调模型在小样本上的绝对性能优势。

-----

## Supplementary Tables

**Table S1. The number of articles comprising both abstracts and full texts for the pre-training of SteelBERT language model.**
Note that the word count encompasses not only textual content, but also mathematical equations and chemical formulas. Approximately 88,000 full texts related to the topic of “steel” were sourced from the Web of Science database. Of these, approximately 55,000 full texts were successfully acquired, while the remaining 33,000 were only accessible in abstract form. The comprehensive training corpus encompasses an estimated 0.96 billion words.

| Type | Sub-type | Papers | Words |
| :--- | :--- | :--- | :--- |
| **Abstract** | Article | 2,373,726 | 398,038,432 |
| | patent | 1,324,883 | 259,234,858 |
| | Meeting | 550,447 | 82,101,497 |
| **Full text** | Elsevier | 37,778 | 560,154,423 |
| | Springer | 12,444 | 319,441,782 |
| | MDPI | 4,134 | 72,568,172 |
| | ASME | 226 | 5,967,200 |
| **In total** | **All** | **4,303,638** | **958,131,577** |

<br>

**Table S2. Evaluation on processing text classification task with different sampling ratio.**

| Ratio | Dataset | F1 score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **1:5** | Validation | 96.67±0.18 | 96.82±0.37 | 96.54±0.37 |
| | Testing | 97.12±0.16 | 97.35±0.24 | 96.90±0.32 |
| | Unseen samples | 98.29±0.16 | - | 96.65±0.30 |
| **1:10** | Validation | 98.06±0.11 | 97.44±0.14 | 98.68±0.20 |
| | Testing | 98.07±0.11 | 97.66±0.25 | 98.48±0.10 |
| | Unseen samples | 99.17±0.07 | - | 98.36±0.13 |
| **1:15** | Validation | 98.49±0.05 | 98.32±0.10 | 98.66±0.14 |
| | Testing | 98.40±0.08 | 98.18±0.14 | 98.62±0.21 |
| | Unseen samples | 99.37±0.04 | - | 98.74±0.08 |

<br>

**Table S3. Comparison of pretrained models on abstracts clustering task.**

| Model | SI | DBI | DI | CHI |
| :--- | :--- | :--- | :--- | :--- |
| Bert | 0.25±0.11 | 0.44±0.01 | 0.01±0.01 | 4192.07±976.52 |
| SciBERT | 0.37±0.01 | 0.46±0.01 | 0.02±0.01 | 1659.80±58.74 |
| MatSciBERT | 0.68±0.01 | 0.44±0.01 | 0.17±0.01 | 33807.73±1893.26 |
| SteelBERT | 0.72±0.01 | 0.37±0.01 | 0.22±0.02 | 75177.57±8760.06 |

<br>

**Table S4. Comparison of pretrained models on chemical elements visualization task.**

| Model | SI | DBI | DI | CHI |
| :--- | :--- | :--- | :--- | :--- |
| Bert | -0.23±0.01 | 18.82±2.75 | 0.02±0.01 | 1.24±0.06 |
| SciBERT | -0.22±0.01 | 9.16±0.91 | 0.03±0.02 | 3.48±0.14 |
| MatSciBERT | -0.18±0.02 | 14.54±1.24 | 0.02±0.01 | 1.86±0.17 |
| SteelBERT | -0.07±0.01 | 6.87±0.58 | 0.05±0.01 | 5.48±1.02 |

<br>

**Table S5. Words and chunks relevant to processing actions within each topic cluster.**

| Topic | Processing actions |
| :--- | :--- |
| **Material analysis techniques** | x-ray diffraction, electrochemical impedance spectroscopy, scanning electron microscopy, differential scanning calorimetry, thermogravimetric analysis, x-ray diffraction xrd analyses, electrochemical impedance spectroscopy eis tests, x-ray diffraction xrd measurements, x-ray photoelectron spectroscopy xps, electrochemical impedance spectroscopy eis measurements, x-ray diffraction xrd experiments, atom probe tomography apt, transmission electron microscopy tem studies, transmission electron microscopy tem, electrochemical impedance spectroscopy eis measurement, x-ray diffraction xrd analysis, electron backscattered diffraction ebsd, tem observations, tem analysis, apt experiments, xrd tests, apt measurements, lsp experiments, pbs solution, tem studies, xps analysis, xrd analyses, apt analyses, ebsd analysis, lcf tests, raman spectroscopy, xrd measurements, apt analysis, xrd measurement, xrd experiments, ebsd measurements, dilatometric measurements, ebsd measurement, eis measurements, ebsd scans, ebsd scanning, xrd analysis, dilatometric tests, tem observation, eis tests, dsc experiments, xps analyses, xps measurements |
| **Ion irradiation** | ion-milled, ion milling process, neutron irradiation, proton irradiation, irradiation experiments, ion irradiations, ion milling, ion implantation, ion irradiation, electron transparency, electron irradiation |
| **Solvents** | anhydrous ethanol, bi-distilled water, pure water, deionised water, millipore water, double distilled water, milli - q water, ultrapure water, double - distilled water, bidistilled water, triply distilled water, ultra-pure water, high purity water, distilled water, absolute ethanol, ultra pure water, demineralized water, absolute ethyl alcohol, sterile distilled water, de-ionized water, ethyl alcohol, de-ionised water, isopropyl alcohol, deionized water, sterilized water, doubly distilled water |
| **Electrochemical testing** | polarization experiments, impedance measurements, cathodic hydrogen charging, polarization test, potentiodynamic measurements, anodic polarization, potentiodynamic scans, electrochemical impedance measurements, electrochemical polarization tests, potentiodynamic polarization test, electrochemical experiments, electrochemical tests, cathodic polarization, potentiodynamic polarization measurements, electrochemical testing, electrochemical experiment, potentiodynamic polarisation tests, polarization measurements, potentiodynamic polarization, hydrogen charging, potentiostatic polarization test, anodic polarization tests, potentiodynamic polarization experiments, potentiodynamic tests, potentiostatic tests, electrochemical hydrogen charging, potentiostatic polarizations, potentiostatic test, potentiodynamic polarization tests, polarization tests, electrochemical test, potentiostatic experiments, electrochemical measurements, potentiostatic polarization, electrochemical measurement, cyclic voltammetry, magnetic measurements, potentiodynamic polarization measurement, electrochemical impedance spectroscopy measurements, potentiodynamic polarization studies |
| **Welding** | induction-brazed, wire-percussion-welded, fusion-welded, brazing, groove-welded, spot-welded, lap-welded, weld, resistance-welded, spot-welding, deposition process, welding experiment, friction stir welding, brazing experiments, welded joints, welding operations, laser treatment, laser surface alloying, laser cutting, brazing process, laser processing, resistance spot welding, spot welding, welding experiments, laser welding, welding process, welding trials, weld metal, laser cladding, post weld heat treatment |
| **Heat treatment** | tempered, soaking , normalized , melted , aging , austenizing, normalize, homogenization, premelted, quenching, overaging, pre-melted, warm-forged, vacuum-annealed, aged, reheat, exposure, die-quenching, reaustenitised, self-tempered, temper-rolled, soft-annealing, bainitising, cross-rolled, re-rolled, solutionised, overageing, rough-rolled, intermediate-quenching, overtempered, normalising, melt-quenching, heating, cooling, melting, solidified, solutionized, autotempering, step-quenched, recrystallization, double-tempered, carburized, soak, co-heated, watercooled, mill-annealed, bright-annealing, reversion-annealed, die-quenched, heated, solution-annealed, austenised, hyperquenched, steam-quenched, solid-solution-treated, semi-annealing, reheating, semi-melted, hot-rolled, pre-quenched, cool-rolled, cryotreated, partitioning, austenitizied, preaustenitized, pit-treated, batch-annealed, melt, presoaking, pre-heated, austempering, unsm-treated, peak-aging, well-homogenized, first-annealed, forgeding, press-forged, re-forging, pretreated, carbonitrided, rough-machined, furnace-heating, post-heating, vacuum-sealed, sandblast-treated, vacuum-processed, hot-roll-bonding, roll-bonded, roll-bonding, electrodepositing, plasma-nitrided, austenitezed, flame-hardened, pre-tempering, argon-cooling, pre-nitrided, micro-alloyed, bake-hardening, freezing, heat-sealed, solid-solutioned, rolling, calcined, pre-soaked, nitrogenized, prebaking, baking, prebaked, cold-swaged, oven-drying, degassing, degreasing, preoxidized, degassed, solidifying, hardening, preheating, overheated, homogenize, rolled, casting, pre-forging, sintered, low-rolled, finish-rolled, cryorolling, re-crystallizing, post-aged, over-heating, homogenised, bright-annealed, carbonitriding, smelting, superheated, sinterhardening, hot-dipping, overheating, over-carburizing, forged, self-tempering, nitriding-treated, cryogenic-rolled, recrystalised, re-aged, groove-rolled, forging, induction-hardened, groove-rolling, ion-nitrided, presoaked, vacuum-drying, hydrotreated, smelt, re-solidify, cold-forged, recrystallized, shape-rolled, vacuum-sintered, re-austenitisized, solutionising, surface-oxidized, thixoforged, oxynitrocarburized, ingot-casted, pre-aged, warm-rolled, sinter-hardening, drop-casting, vacuum-dried, hot-pressing, hot-compacted, dry-sintered, co-sintered, calcining, sinterized, hot-compressed, solution heat treatment, normalizing treatment, subsequent quenching, immediate water quenching, aging treatment, nitriding treatments, surface cleaning, isothermal annealing, rapid cooling, diffusion annealing, cold rolling, deep cryogenic treatment, hot rolling, air drying, steel samples, water quenching, recrystallization annealing, rolling process, shot peening, plasma nitriding, free cooling, direct quenching, final thinning, ultrasonic cleaning, final polish, cloth polishing, bristle brush, isothermal aging, surface grinding, sand blasting, copious amount, solution treatment, thermal treatment, controlled rolling, homogenization heat treatment, thermal treatments, oil quenching, furnace cooling, hot forging, thermal aging, coiling temperature, plasma nitriding treatment, whole process, cryogenic treatment, coated samples, rough rolling, thermal annealing, acetone degreasing, reversion annealing, heat treatment, immediate quenching, hot extrusion, grinding operation, continuous cooling, different heat treatments, isothermal holding, rapid quenching, colloidal silica, surface treatment, warm rolling |
| **Material characterization methods** | depth profiling, microstructural observation, sample preparation, phase identification, chemical analysis, microstructural examination, metallographic analysis, metallographic examination, elemental analysis, thermal analysis, microstructural examinations, phase analysis, microstructural analysis, microstructural observations, microstructure analysis, microstructure observation, metallographic preparation, chemical analyses, metallographic observation |
| **Wear and friction** | reciprocating sliding tests, dry sliding wear tests, wear testing, dry sliding tests, tribological tests, wear tests, erosion tests, sliding tests, tribocorrosion tests, friction tests, sliding wear tests, tribological experiments, cavitation erosion tests, wear test, wear experiments |
| **Corrosion experiment** | leaching experiments, salt spray tests, salt spray test, corrosion experiments, oxidation experiments, corrosion studies, oxidation tests, corrosion test, corrosion testing, corrosion tests, static corrosion tests, weight loss measurements, weight loss tests, exposure experiments, corrosion measurements, immersion tests, immersion test, immersion experiments, oxidation test |
| **Fatigue testing** | corrosion fatigue tests, cyclic tests, rotating bending fatigue tests, fatigue test, fatigue tests, low cycle fatigue tests, fatigue pre-cracking, creep testing, uniaxial creep tests, fatigue testing, thermal fatigue tests, fatigue experiments, fatigue precracking, creep test |
| **Mechanical testing** | standard tensile test, torsion tests, hot compression tests, shear tests, uniaxial tension tests, dynamic tensile tests, tensile deformation, uniaxial tension, tensile specimens, tensile testing, tensile coupon test, interrupted tensile tests, standard tensile tests, tensile test, dl - epr test, dl - epr tests, high temperature tensile tests, static tensile tests, quasi-static tensile tests, isothermal hot compression tests, static tensile test, hot deformation tests, tensile experiments, coupon tests, tensile tests, quasi-static uniaxial tensile tests, fracture tests, tensile shear tests, tension tests, uniaxial compression tests, impact testing, uniaxial hot compression tests, tensile coupon tests, hot tensile tests, tensile samples, uniaxial tensile tests, compression tests, monotonic tensile tests, uniaxial tensile test, compression test, uniaxial tension test, uniaxial tensile testing, charpy v - notch impact tests, fcg tests, charpy tests, three - point bending tests, charpy impact tests, scc tests, room - temperature tensile tests, low - cycle fatigue tests, low cycle fatigue tests, charpy impact test, vickers hardness tests, tensile - shear tests, ssrt tests |

<br>

**Table S6. Comparison of pretrained models on number decoding task.**

| Model | Dataset | R2 | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | Training | 93.57±0.28 | 115.46±1.97 | 146.68±3.11 |
| | Testing | 93.13±0.27 | 117.77±1.61 | 151.06±2.88 |
| **SciBERT** | Training | 94.80±0.38 | 101.17±3.95 | 131.69±4.72 |
| | Testing | 94.64±0.39 | 100.86±3.91 | 133.17±4.86 |
| **MatSciBERT** | Training | 79.95±1.02 | 185.87±6.24 | 258.87±6.40 |
| | Testing | 79.49±0.94 | 186.11±5.01 | 260.91±5.88 |
| **SteelBERT** | Training | 98.35±0.15 | 50.32±3.21 | 73.70±3.37 |
| | Testing | 98.11±0.19 | 54.10±3.34 | 79.09±4.10 |

<br>

**Table S7. Performance comparison of fine-tuned LLMs for property prediction.**

| Model | Dataset | Yield strength ($R^2$/%) | YS (MAE/MPa) | UTS ($R^2$/%) | UTS (MAE/MPa) | Elongation ($R^2$/%) | EL (MAE/%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3.1-8B** | Training | 51.23±9.01 | 179.12±21.43 | 36.35±11.07 | 210.12±17.06 | 16.58±24.97 | 9.77±1.92 |
| | Validation | 37.50±8.20 | 204.74±19.05 | 23.39±10.42 | 232.38±14.86 | 4.70±24.99 | 11.00±1.79 |
| | Testing | -7.61±5.18 | 216.35±4.54 | -0.39±8.34 | 215.23±10.60 | -15.06±27.32 | 12.16±1.51 |
| **Llama-3.1-8B-Instruct** | Training | 49.48±5.24 | 191.20±8.43 | 52.17±8.99 | 183.08±15.76 | 38.35±22.40 | 8.37±1.67 |
| | Validation | 26.25±4.52 | 217.91±6.75 | 36.08±7.19 | 215.21±11.80 | 24.93±23.84 | 9.41±1.58 |
| | Testing | 2.71±17.40 | 196.41±18.97 | 1.82±15.28 | 219.45±16.85 | -6.72±20.95 | 11.82±1.15 |
| **ChatGPT-3.5-turbo-1106** | Training | 67.83±3.17 | 92.11±2.63 | 81.57±1.32 | 79.20±1.48 | 73.03±3.27 | 4.72±21.84 |
| | Validation | 68.60±3.19 | 123.41±4.11 | 79.93±2.04 | 105.56±6.75 | 51.84±9.91 | 6.62±0.42 |
| | Testing | 7.81±13.34 | 199.39±18.15 | 1.82±16.06 | 237.86±22.53 | -10.45±11.28 | 10.30±0.80 |

<br>

**Table S8. Chemical compositions of the studied steels.**

| Composition /Wt. % | C | Cr | Ni | Mn | Si | Mo | Nb | Cu | B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Type 1** | 0.048 | 15.13 | 8.09 | 0.65 | 0.27 | 2 | - | 1.22 | - |
| **Type 2** | 0.092 | 15.01 | 7.92 | 0.83 | 0.45 | 1.99 | 0.043 | 1.49 | - |
| **Type 3** | 0.063 | 15.06 | 7.91 | 0.83 | 0.45 | 1.99 | 0.0005 | 2.89 | 0.0011 |

<br>

**Table S9. Mean results of the predictive models from 5 partitions of experimental datasets.**

| Model | Datasets | YS ($R^2$/%) | YS (MAE/MPa) | UTS ($R^2$/%) | UTS (MAE/MPa) | EL ($R^2$/%) | EL (MAE/%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GBR** | Training | 89.40±0.26 | 52.53±1.47 | 98.22±0.52 | 14.78±1.90 | 81.50±1.45 | 3.28±0.10 |
| | Testing | 80.74±13.05 | 61.45±12.72 | 82.25±10.23 | 36.36±12.67 | 65.02±9.65 | 4.14±0.43 |
| **KRR** | Training | 90.67±0.60 | 49.86±1.68 | 87.15±1.70 | 44.48±2.81 | 79.48±1.44 | 3.21±0.19 |
| | Testing | 76.44±12.99 | 67.97±12.47 | 59.22±28.42 | 66.36±11.96 | 37.45±19.91 | 5.01±1.03 |
| **MLP** | Training | 80.93±9.5 | 68.49±17.27 | 62.58±9.79 | 72.04±11.27 | 52.86±4.24 | 5.12±0.35 |
| | Testing | 57.22±21.06 | 92.90±26.38 | 50.17±30.22 | 84.44±14.49 | 27.20±20.36 | 6.25±0.55 |
| **SVR** | Training | 53.10±3.62 | 88.61±2.53 | 52.85±7.41 | 78.72±4.43 | 61.83±2.27 | 4.63±13.28 |
| | Testing | 20.69±34.54 | 118.27±21.51 | 35.58±9.47 | 90.90±10.44 | 28.09±20.17 | 5.94±1.00 |
| **XGB** | Training | 98.87±0.15 | 15.37±1.07 | 97.84±0.33 | 16.14±1.24 | 98.57±0.36 | 0.79±0.09 |
| | Testing | 82.99±11.39 | 51.46±13.07 | 81.86±12.04 | 42.79±10.72 | 60.46±13.11 | 4.12±0.39 |
| **RF** | Training | 96.94±0.54 | 26.69±1.92 | 95.93±1.17 | 22.28±2.83 | 90.94±2.22 | 1.85±0.20 |
| | Testing | 81.72±10.52 | 54.86±14.90 | 83.45±7.71 | 42.34±9.35 | 66.11±10.29 | 3.62±0.33 |
| **Fine-tuned model** | Training | 91.63±1.63 | 45.55±4.31 | 92.21±1.73 | 30.75±4.99 | 90.04±3.7 | 2.14±0.48 |
| | Testing | 89.85±6.17 | 41.60±8.77 | 88.34±5.95 | 43.38±8.89 | 87.24±5.15 | 2.81±0.62 |

<br>

**Table S10. Performance of fine-tuned LLMs for property prediction on experimental data.**

| Model | Datasets | YS ($R^2$/%) | YS (MAE/MPa) | UTS ($R^2$/%) | UTS (MAE/MPa) | EL ($R^2$/%) | EL (MAE/%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT** | Training | 73.97±7.87 | 77.75±10.01 | 66.27±3.23 | 69.47±3.72 | 65.02±3.76 | 4.39±0.21 |
| | Testing | 69.33±7.69 | 87.14±10.34 | 61.91±3.40 | 63.85±4.01 | 64.42±3.79 | 4.37±0.48 |
| **SciBERT** | Training | 80.06±1.71 | 73.87±3.15 | 64.07±6.73 | 67.86±5.23 | 59.50±6.72 | 4.67±0.44 |
| | Testing | 71.14±2.61 | 74.66±4.10 | 57.42±5.64 | 72.24±8.18 | 58.54±6.53 | 4.43±0.79 |
| **MatSciBERT** | Training | 80.83±2.13 | 70.16±3.99 | 68.99±2.36 | 67.79±3.00 | 67.60±6.01 | 4.15±0.27 |
| | Testing | 75.24±2.23 | 75.08±4.80 | 61.82±3.26 | 63.26±3.77 | 65.32±5.69 | 4.41±0.64 |
| **Llama-3.1-8B** | Training | 80.59±1.75 | 71.50±4.03 | 69.30±2.37 | 65.93±2.96 | 53.40±12.43 | 5.08±0.74 |
| | Testing | 75.89±2.29 | 73.81±4.02 | 64.82±2.80 | 65.99±4.93 | 52.48±12.57 | 4.11±0.44 |
| **Llama-3.1-8B-Instruct** | Training | 75.73±7.83 | 72.65±8.54 | 63.99±5.32 | 69.95±3.96 | 59.14±7.00 | 4.83±0.41 |
| | Testing | 71.85±7.58 | 81.86±11.35 | 60.52±4.91 | 71.39±5.98 | 51.00±6.75 | 4.52±0.35 |

<br>

**Table S11. R2 score on testing set with varying amounts of training sets ranging from 15 to 50 for yield strength.**

| Amount | SVR | RF | XGB | GBR | KRR | MLP | Ft-model |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 15 | -1.66±2.44 | 0.70±0.08 | 0.70±0.16 | 0.71±0.12 | -0.87±1.76 | -0.64±1.71 | 0.58±0.23 |
| 20 | -3.02±8.39 | 0.77±0.07 | 0.76±0.11 | 0.76±0.09 | -2.63±12.42 | -0.30±0.84 | 0.57±0.13 |
| 25 | -2.13±8.72 | 0.77±0.10 | 0.77±0.10 | 0.78±0.10 | -0.34±3.51 | -0.0±0.64 | 0.73±0.16 |
| 30 | -0.25±1.34 | 0.79±0.11 | 0.81±0.07 | 0.80±0.08 | 0.58±0.35 | 0.33±0.32 | 0.74±0.19 |
| 35 | -0.49±2.12 | 0.83±0.06 | 0.84±0.04 | 0.84±0.08 | 0.63±0.35 | 0.35±0.54 | 0.84±0.15 |
| 40 | -0.86±2.89 | 0.84±0.09 | 0.84±0.06 | 0.86±0.07 | 0.63±0.42 | 0.27±0.83 | 0.82±0.08 |
| 45 | -0.95±4.8 | 0.84±0.11 | 0.85±0.07 | 0.86±0.10 | 0.63±0.52 | 0.45±0.32 | 0.88±0.23 |
| 50 | -0.56±2.54 | 0.82±0.15 | 0.81±0.31 | 0.78±0.36 | 0.33±1.65 | 0.12±1.48 | 0.87±0.16 |
*(Note: Excerpted interval values (every 5 points) from the complete Table S11 for display brevity. The full table spans amounts 15 to 50 linearly.)*

<br>

**Table S12. R2 score on testing set with varying amounts of training sets ranging from 15 to 50 for ultimate tensile strength.**

| Amount | SVR | RF | XGB | GBR | KRR | MLP | Ft-model |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 15 | -1.25±1.24 | 0.63±0.11 | 0.64±0.16 | 0.65±0.12 | -0.87±1.35 | -0.52±1.07 | 0.53±0.25 |
| 20 | -3.39±10.19 | 0.66±0.09 | 0.65±0.17 | 0.65±0.16 | -1.7±6.99 | -0.21±0.54 | 0.68±0.13 |
| 25 | -8.83±36.17 | 0.7±0.11 | 0.7±0.17 | 0.69±0.16 | -3.17±14.76 | -0.14±0.54 | 0.65±0.11 |
| 30 | -0.65±1.57 | 0.71±0.13 | 0.72±0.14 | 0.74±0.13 | -0.05±2.28 | -0.1±0.87 | 0.68±0.15 |
| 35 | -0.11±0.94 | 0.76±0.11 | 0.79±0.10 | 0.81±0.07 | 0.51±0.45 | 0.34±0.52 | 0.77±0.15 |
| 40 | -0.81±2.3 | 0.81±0.09 | 0.82±0.06 | 0.83±0.06 | 0.5±0.50 | 0.27±0.77 | 0.76±0.14 |
| 45 | 0.01±0.94 | 0.8±0.15 | 0.82±0.08 | 0.83±0.08 | 0.48±0.83 | 0.25±1.25 | 0.87±0.23 |
| 50 | -0.37±1.64 | 0.79±0.19 | 0.75±0.38 | 0.82±0.15 | 0.39±0.88 | 0.14±1.13 | 0.87±0.18 |
*(Note: Excerpted interval values for brevity.)*

<br>

**Table S13. R2 score on testing set with varying amounts of training sets ranging from 15 to 50 for elongation.**

| Amount | SVR | RF | XGB | GBR | KRR | MLP | Ft-model |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 15 | -2.95±7.94 | 0.57±0.13 | 0.52±0.12 | 0.53±0.16 | -2.33±5.76 | -0.37±0.78 | 0.63±0.12 |
| 20 | -2.41±6.67 | 0.6±0.10 | 0.54±0.13 | 0.57±0.08 | -4.11±16.33 | 0.09±0.28 | 0.66±0.08 |
| 25 | -3.18±10.76 | 0.63±0.11 | 0.6±0.13 | 0.6±0.13 | -3.08±12.86 | 0.27±0.25 | 0.66±0.11 |
| 30 | -1.56±5.47 | 0.67±0.08 | 0.64±0.11 | 0.64±0.12 | -1.11±4.99 | 0.31±0.23 | 0.74±0.08 |
| 35 | -0.31±1.55 | 0.69±0.08 | 0.65±0.09 | 0.66±0.08 | 0.09±1.19 | 0.28±0.13 | 0.78±0.09 |
| 40 | -0.5±1.96 | 0.69±0.12 | 0.62±0.12 | 0.66±0.11 | 0.41±0.58 | 0.30±0.25 | 0.80±0.11 |
| 45 | 0.03±0.88 | 0.70±0.15 | 0.62±0.18 | 0.67±0.13 | 0.55±0.18 | 0.24±0.25 | 0.79±0.06 |
| 50 | -1.16±5.85 | 0.67±0.25 | 0.61±0.31 | 0.64±0.22 | 0.43±0.56 | 0.22±0.46 | 0.86±0.11 |
*(Note: Excerpted interval values for brevity.)*

<br>

**Table S14. Processing parameters of base steels and optimized steels for Type 2.**

*(Part 1: Primary Processing)*

| Sample | solution treatment temperature/℃ | solution treatment time/mins | cold-rolled thickness reduction% | tempering temperature/℃ |
| :--- | :--- | :--- | :--- | :--- |
| **Base steel 1** | 1050 | 240 | 79 | 590 |
| **Optimized steel 1-1** | 1050 | 240 | 79 | 625 |
| **Optimized steel 1-2** | 1050 | 240 | 79 | 625 |
| **Optimized steel 1-3** | 1050 | 240 | 79 | 625 |
| **Base steel 2** | 1050 | 240 | 57 | 660 |
| **Optimized steel 2** | 1050 | 240 | 57 | 625 |

*(Part 2: Secondary Processing)*

| Sample | tempering time/mins | Secondary round of cold-rolled thickness reduction% | Secondary round of tempering temperature/℃ | Secondary round of tempering time/mins |
| :--- | :--- | :--- | :--- | :--- |
| **Base steel 1** | 60 | - | - | - |
| **Optimized steel 1-1** | 30 | 75 | 660 | 30 |
| **Optimized steel 1-2** | 30 | 75 | 625 | 60 |
| **Optimized steel 1-3** | 30 | 25 | 625 | 60 |
| **Base steel 2** | 15 | - | - | - |
| **Optimized steel 2** | 30 | 75 | 625 | 60 |

<br>

**Table S15. Results of experimental tensile tests and predicted values for optimized and base steels.**

| Experimental sample | Yield strength/ MPa | Ultimate tensile strength/ MPa | Elongation/ % |
| :--- | :--- | :--- | :--- |
| **Base steel 1** | 910 | 1086 | 23.8 |
| **Optimized steel 1-1** | 960 (898.44)\* | 1138 (1090.87)\* | 32.5 (30.32)\* |
| **Optimized steel 1-2** | 1028 (1055.73)\* | 1234 (1163.30)\* | 25.8 (27.39)\* |
| **Optimized steel 1-3** | 913 (947.07)\* | 1060 (1100.22)\* | 29.7 (28.30)\* |
| **Base steel 2** | 782 | 989 | 25.2 |
| **Optimized steel 2** | 923 (936.99)\* | 1090 (1079.42)\* | 28.4 (26.63)\* |
*\* The value in the bracket represents the predicted value from fine-tune model.*

-----

## Supplementary Note

### Note S1.

Our labelled dataset for classification task consists of 494 positive samples and 39,358 negative samples, drawn from 39,852 sentences across 258 articles. We experimented with various negative sampling ratios (1:5, 1:10, and 1:15), yielding 2,470, 4,940, and 7,410 negative samples, respectively. The final dataset retained all positive samples and randomly sampled negatives, which were shuffled and split into training, validation, and test sets in a 6:2:2 ratio. Notably, the large pool of unused negative samples was not included in training data. These unsampled negatives reflect the diversity and potential edge cases that the model might encounter in practice, effectively simulating the challenge of unseen negative examples.

Our classification model, based on the SteelBERT backbone with a dropout layer and a single MLP layer, was evaluated across 10 random seeds to assess performance using F1 score, precision, and recall, as shown in Table S2. Results indicate that the model performs well on classifying negative samples, regardless of the specific ratio of positive to negative samples. Furthermore, as the ratio of negative samples increases, the model’s overall classification performance improves, even when encountering previously unseen negative examples.

### Note S2.

For the clustering task, we evaluated various models using several metrics, including the Silhouette Index (SI)[1], Davies-Bouldin Index (DBI)[2], Dunn Index (DI)[3] and Calinski-Harabasz Index (CHI)[4], which are given by the equations below. This allows us to better gauge model performance across different dimensions.

The $SI$ is defined by equation (1), where $a(i)$ represents the average distance between $i$ and other points in the same cluster, $b(i)$ denotes the average distance between $i$ and points in the nearest neighbor cluster.
$$SI = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} \tag{1}$$

The $DBI$ is defined by equation (2), where $N$ is the number of clusters, $s_i$ denotes the average intra-cluster distance for cluster $i$, $s_j$ represents the average intra-cluster distance for cluster $j$, $d_{ij}$ is the distance between the centroids of clusters $i$ and $j$.
$$DBI = \frac{1}{N} \sum_{i=1}^{k} \max_{j \neq i} \frac{s_i + s_j}{d_{ij}} \tag{2}$$

The $DI$ is defined by equation (3), where $d(c_i, c_j)$ is the distance between the centroids of clusters $i$ and $j$, where $\delta(C_i)$ represents the maximum intra-cluster distance for cluster $i$.
$$DI = \frac{\min_{i \neq j} d(c_i, c_j)}{\max_{1 \le i \le k} \delta(C_i)} \tag{3}$$

The $CHI$ is defined by equation (4), where $N$ represents the total number of data points, $k$ is the number of clusters, $Tr(B_k)$ is the trace of the between-cluster dispersion matrix, which measures the spread between clusters, and $Tr(W_k)$ denotes the trace of the within-cluster dispersion matrix, measuring the spread within each cluster.
$$CHI = \frac{Tr(B_k)}{Tr(W_k)} \cdot \frac{N - k}{k - 1} \tag{4}$$

We utilized the first token of each sequence embedding to represent the abstracts, as it typically captures a global summary of the entire input, offering a compact yet effective representation. This approach allows us to maintain the essential information while reducing computational complexity. We applied the Uniform Manifold Approximation and Projection (UMAP) algorithm to reduce the dimensionality of the embeddings generated by various pretrained models, ensuring the preservation of key features. Following this, we used the Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm for clustering. To ensure robustness, we conducted the clustering task with ten different random seeds and calculated the mean and standard deviation for the SI, DBI, DI, and CHI, as shown in Table S3. For four metrics, except DBI, higher values indicate better clustering performance. The SteelBERT model consistently outperformed across all four metrics, demonstrating its superior ability to capture textual knowledge specific to the steel materials domain.

### Note S3.

We utilized the same dataset for this study, dividing it into a training set and a validation set with an 8:2 ratio. The dataset extracted from literatures from the year 2022 to 2023 was designated as the test dataset. To predict the mechanical properties of steel, we built a regression task in an instruction, converting multimodal data that includes both tabular and text information. The prompt for fine-tuning is `“The elemental composition of the steel in weight % as obtained is: {weight percentage1}{element1}…{weight percentagen}{elementn}.{processing text}. What is the {property name} of this steel?###”` And the answer is `“The {property name} of this steel is {property value}{property unit}.”`

The “property name” in this prompt comprises “ultimate tensile strength”, “yield strength” and “elongation”. Subsequently, we fine-tuned on the GPT-3.5-turbo-1106 model, setting the number of epochs to 6 and the default learning rate. The sample for fine-tuning is provided below.

**Prompt:** The elemental composition of the steel in weight % as obtained is: 0.27 C, 0.7 Al, 1.7 Si, 2.3 Mn, 0.4 Ni. The steel was first austenitized at 930 °C for 45 min. Then the No. 3 steel was continuously cooled from 325 °C to 295 °C at a cooling rate of 0.25 °C/min. Finally, all samples were tempered at 320 °C for 1 h, followed by air cooling to room temperature. What is the yield strength of this steel?\#\#\#
**Answer:** The yield strength of this steel is 958 MPa.@@@

### Note S4.

We also fine-tuned several versions of large open-source generative models, such as LLaMa, using LoRA (Low-Rank Adaptation)[5], a method that enables efficient model adaptation. LoRA achieves this by freezing most of the pre-trained model’s parameters and introducing learnable low-rank matrices for task-specific updates, significantly reducing the number of parameters updated during training. In our case, we set the rank parameter to 8, which provided sufficient capacity to capture the complex relationships in the steel data, while an alpha value of 16 amplified the effect of these low-rank matrices, enhancing fine-tuning without excessive computational cost.

However, using generative models for mechanical property prediction presents several challenges, especially with a small dataset. While large generative models excel in creative tasks such as text generation, summarization, and other open-ended outputs, mechanical property prediction demands precision and consistency. Generative models inherently introduce variability in their responses, which can be advantageous in creative domains but problematic for tasks requiring deterministic and reproducible predictions. In materials science, where small changes in steel composition and processing can significantly impact mechanical properties, this variability risks producing unstable or inconsistent outputs.

To address this, we carefully optimized key hyperparameters, including the temperature and top-p, which control the model’s behavior during prediction. The temperature parameter adjusts the randomness of model outputs by controlling the sharpness of the probability distribution over possible predictions. A high temperature introduces more randomness, making the model explore less likely options, while a low temperature focuses the model on the most probable outputs, reducing unpredictability. Given that mechanical property prediction requires accuracy and reliability, we set the temperature to 0.2. This low temperature ensures the model favors high-probability outputs, minimizing the risk of unrealistic predictions and maintaining results grounded in the steel data’s physical and chemical constraints.

Similarly, the top-p parameter (nucleus sampling) influences the model’s output by controlling the cumulative probability of predictions it considers. A lower top-p value restricts the model to only the most likely predictions, while a higher value allows it to explore a broader range of possibilities. For our task, we set a top-p value of 0.85, striking a balance between ensuring the model focused on probable outcomes while still allowing it to capture minor but significant variations, such as the influence of microalloying elements or subtle differences in steel processing.

To ensure the models could handle our highly specific task of steel property prediction, we designed a specialized prompt for fine-tuning. The prompt was structured to explicitly guide the model to use both elemental composition and processing details as input for generating precise property predictions. Our fine-tuning prompt was as follows: `“The elemental composition of the steel in weight percentage is as follows: {compositions} {processing text}. Based on empirical data and predictive models for steel alloys with similar composition and processing, please determine the {property} of this steel. Provide a precise numerical value along with the appropriate unit.”`

This prompt ensured that the model focused on the most relevant aspects of steel data—composition and processing—while demanding a specific numerical output with the correct unit, which is crucial in scientific prediction tasks. By guiding the model in this way, we aimed to minimize the inherent variability typical of generative models and ensure its predictions were both accurate and actionable within the materials science domain.

These optimizations, along with training the model over 5 epochs, ensured that the generative model produced consistent and high-quality predictions without overfitting. Despite these efforts, SteelBERT consistently outperformed the fine-tuned generative models, especially in cases requiring a deep understanding of steel-specific interactions between composition and processing, as shown in Table S7. While fine-tuning the generative models yielded reasonable results, they lacked the specialized knowledge necessary for precise steel property predictions and were significantly less efficient. Our SteelBERT model has around 200 million parameters, whereas the LLaMa-8B model, for example, contains 8 billion parameters—about 40 times more. This difference makes SteelBERT not only more resource-efficient but also better suited for the task, as it captures the intricate patterns and relationships specific to steel data with far fewer parameters. This efficiency, combined with SteelBERT’s pre-training in materials science, allowed us to achieve superior performance without the massive computational overhead required by larger, general-purpose generative models.

-----

## Supplementary Video

**Movie S1 (separate file).** The supporting video demonstrates the prediction of mechanical properties for steel with various chemical compositions and processing routes, while also comparing them with other steel samples reported in the literature using scatter plots.

-----

## References

[1] P.J. Rousseeuw, Silhouettes: A graphical aid to the interpretation and validation of cluster analysis, J. Comput. Appl. Math. 20 (1987) 53–65. [https://doi.org/10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427\(87\)90125-7).
[2] D.L. Davies, D.W. Bouldin, A cluster separation measure, IEEE Trans. Pattern Anal. Mach. Intell. (1979) 224–227. [https://doi.org/10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909).
[3] J.C. Dunn, A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters, J. Cybern. 3 (1973) 32–57. [https://doi.org/10.1080/01969727308546046](https://doi.org/10.1080/01969727308546046).
[4] T. Calinski, J. Harabasz, A dendrite method for cluster analysis, Commun. Stat.-Theory Methods 3 (1974) 1–27. [https://doi.org/10.1080/03610927408827101](https://doi.org/10.1080/03610927408827101).
[5] E.J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of large language models, (2021). [http://arxiv.org/abs/2106.09685](http://arxiv.org/abs/2106.09685).