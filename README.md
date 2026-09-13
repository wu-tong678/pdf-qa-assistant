# PDF问答助手

上传PDF文档，用自然语言提问，AI从文档中检索并回答。

## 核心功能

- 📄 支持 PDF 文档上传与解析
- 🔍 基于 RAG 架构的智能检索
- 💬 自然语言问答交互
- 📊 内置检索效果评估体系

## 技术栈

- Python 3.10+
- Streamlit（Web 界面）
- LangChain（RAG 流程编排）
- Chroma（向量数据库，持久化存储）
- 智谱 GLM-4（大模型）
- 智谱 GLM-Rerank（精排优化）

## 在线体验

[https://pdf-app-assistant-xogchwtvcwzl4dv88wmljj.streamlit.app/](https://pdf-app-assistant-xogchwtvcwzl4dv88wmljj.streamlit.app/)

## 项目截图

<img src="images/screenshot.png" alt="项目截图" width="600">

## 评估结果

### 评估体系演进

- **v1（19 题，单一 PDF）**：初版评估，文档短、题目简单，Rerank 无提升
- **v2（58 题，3 份 PDF，4 档难度）**：重建评估集，覆盖不同类型文档和难度

使用 58 道问题、3 份 PDF（Transformer 论文、GLM-4 技术报告、LangChain Agents 文档）测试检索效果（Top-5 Hit Rate）。

| 测试集 | 检索方式 | Top-5 命中率 |
|--------|----------|--------------|
| v1（19 题，单一 PDF） | 纯 Chroma | 73.68% |
| v1（19 题，单一 PDF） | Chroma + Rerank | 73.68% |
| **v2（58 题，3 份 PDF）** | **纯 Chroma** | **55.17%** |
| **v2（58 题，3 份 PDF）** | **Chroma + Rerank** | **67.24%** |

**结论：** v1 测试集区分度不足，Rerank 无提升；重建 v2 后，Rerank 将 Top-5 命中率提升 **+12.07%**。

**观察：** Rerank 在 GLM-4 报告上从 65% 降到 55%，说明 Rerank 效果依赖文档结构——在相似术语密集的技术报告中，重排序可能引入噪声。

### 按 PDF 分

| PDF | 纯 Chroma | Chroma + Rerank |
|-----|-----------|-----------------|
| attention_is_all_you_need.pdf | 40.00% | 60.00% |
| glm4_report.pdf | 65.00% | 55.00% |
| langchain_agents.pdf | 61.11% | 88.89% |

### 按难度分

| 难度 | 纯 Chroma | Chroma + Rerank |
|------|-----------|-----------------|
| easy | 58.82% | 76.47% |
| cross_paragraph | 47.06% | 70.59% |
| vague | 66.67% | 60.00% |
| distractor | 44.44% | 55.56% |

## 项目结构

```text
pdf-qa-assistant/
├── app.py                      # Streamlit 主程序
├── build_index.py              # 向量库构建脚本
├── evaluate.py                 # v1 基线评估
├── evaluate_with_rerank.py     # v1 Rerank 评估
├── evaluate_v2.py              # v2 评估（多 PDF + 多难度）
├── data/
│   ├── pdf/                    # 原始 PDF
│   │   ├── attention_is_all_you_need.pdf
│   │   ├── glm4_report.pdf
│   │   ├── langchain_agents.pdf
│   │   └── learn.pdf
│   ├── v1/                     # v1 测试集（19 题）
│   │   └── eval_data.json
│   └── v2/                     # v2 测试集（58 题）
│       ├── eval_data_v2.json
│       └── eval_result_v2.txt
├── requirements.txt
└── README.md
