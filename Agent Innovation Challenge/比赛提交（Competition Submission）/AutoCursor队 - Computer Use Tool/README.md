# AutoCursor

AutoCursor - 智能桌面自动化助手 The AI-Powered Desktop Automation Agent

**项目描述：**
AutoCursor 是一个架起人类语言与直接电脑控制之间桥梁的智能体。它通过调用大语言模型来理解用户的自然语言指令，并实时分析屏幕元素，从而将用户的高级意图转化为精确、自主的操作。它能够直接控制鼠标和键盘，完成从简单的点击、文本输入到复杂的多步骤工作流程等一系列任务——将口头指令转化为无缝的数字执行。本项目重新定义了人机交互，为所有人带来直观、易用且功能强大的桌面自动化体验。

**核心机制：**
人类指令 -> 大模型推理与屏幕分析 -> 自动化鼠标/键盘控制


**Description:**
AutoCursor is an intelligent agent that bridges the gap between human language and direct computer control. By leveraging a large language model (LLM) to understand natural language instructions and analyze on-screen elements in real-time, AutoCursor translates high-level user intent into precise, autonomous actions. It directly controls the mouse and keyboard to perform tasks—from simple clicks and text input to complex, multi-step workflows—turning verbal commands into seamless digital execution. Our project redefines human-computer interaction, making desktop automation intuitive, accessible, and powerful for everyone.

**Core Mechanism:**
Human Instruction -> LLM Reasoning & Screen Analysis -> Automated Mouse/Keyboard Control

## 快速开始 Quick Start

### 1. 使用Conda（推荐）配置安装环境
```bash
# 1.1 克隆仓库
git clone https://gitcode.com/your_name/AutoCursor.git
cd AutoCursor

# 1.2 创建并激活环境 (推荐使用python>=3.11）
conda create -n AutoCursor python==3.11.4
conda activate AutoCursor

# 1.3 下载依赖库
pip install -r requirements.txt
```
### 2. 配置大模型API key
```bash
# 2.1 本地新建.env文件
# 2.2 添加自己的API配置
API_KEY= "your-api-key"
DATABASE_URL= "https://your-database-url"
# 此文件已加入.gitignore 不会随git提交
```

### 3. 运行示例代码
```bash
python ./core/autoCursor.py
```
