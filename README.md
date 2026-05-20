# Enterprise RAG Knowledge System

企业知识库RAG系统 - 多Agent协作 + 长链推理

## 核心特性

- **多Agent协作**: DocumentAgent → RetrievalAgent → GeneratorAgent → VerifierAgent
- **长链推理**: 文档加载 → 文本切分 → 向量检索 → 答案生成 → 准确性验证
- **智能检索**: 基于向量相似度的文档检索
- **答案验证**: 自动验证答案置信度和可靠性

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行查询
python main.py -q "公司年假政策是什么?"

# 指定文档目录
python main.py -q "密码要求是什么?" -d sample_docs
```

## 架构

```
├── agents/              # Agent模块
│   ├── document_agent.py   # 文档处理Agent
│   ├── retrieval_agent.py  # 检索Agent
│   ├── generator_agent.py  # 答案生成Agent
│   └── verifier_agent.py   # 验证Agent
├── chains/            # 推理链
│   └── rag_chain.py
├── sample_docs/       # 示例文档
│   ├── employee_handbook.txt
│   └── it_security_policy.txt
└── main.py
```

## 长链推理流程

1. **DocumentAgent**: 加载文档 → 文本切分 → 生成向量嵌入
2. **RetrievalAgent**: 理解查询 → 生成查询向量 → 相似度匹配 → 返回Top-K结果
3. **GeneratorAgent**: 构建上下文 → 基于资料生成答案
4. **VerifierAgent**: 验证答案准确性 → 计算置信度 → 输出最终结果

## License

MIT