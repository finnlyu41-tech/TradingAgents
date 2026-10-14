# Codex CLI 订阅接入（实验性）

`codex_cli` 是直接调用**官方 Codex CLI** 的模型后端，不是 HTTP 中转，也不读取或复制登录令牌。TradingAgents 保留原有分析师、工具、辩论和风控流程；Codex 返回报告、结构化决策或工具请求，工具仍由 TradingAgents 执行。

## 使用

1. 在**运行 TradingAgents 的同一台可信电脑、同一用户环境**安装官方 Codex CLI，运行 `codex login` 并选择 ChatGPT 登录。已经登录的不必重新登录。
2. 将仓库根目录 `.env.codex.example` 的设置合并到现有 `.env`。不要直接覆盖已有 `.env`；不要复制、提交 `auth.json`。
3. 原有启动方式不变：`tradingagents`。也可以在交互菜单选择 **Codex CLI (ChatGPT subscription; experimental)**。

最核心的设置是：

```dotenv
TRADINGAGENTS_LLM_PROVIDER=codex_cli
TRADINGAGENTS_QUICK_THINK_PROVIDER=codex_cli
TRADINGAGENTS_DEEP_THINK_PROVIDER=codex_cli
TRADINGAGENTS_QUICK_THINK_LLM=codex-default
TRADINGAGENTS_DEEP_THINK_LLM=codex-default
```

同时设置两个模型层的 provider，避免现有配置仍将其中一层指向付费 API。旧 `backend_url`、`temperature`、`max_tokens`、`llm_max_retries` 设置必须清空；本后端不静默忽略这些设置。

`codex-default` 使用 Codex 的内置默认模型，**不继承个人 config.toml 的模型或 provider**。要固定模型，用当前账户实际支持的 Codex 模型 ID 替换两个 `codex-default`。默认推理强度为 `medium`，可通过 `TRADINGAGENTS_CODEX_REASONING_EFFORT` 调整；模型不支持时会报错，不自动换模型。

`TRADINGAGENTS_CODEX_COMMAND` 可以是官方可执行文件的完整路径，无须为了本项目改变全局 PATH。例如某些 macOS ChatGPT 安装包内附：

```dotenv
TRADINGAGENTS_CODEX_COMMAND=/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex
```

该路径因安装方式而异；本次仅在此路径验证过 `codex-cli 0.162.0-alpha.17.2` 的版本、参数和 ChatGPT 登录状态。未证明所有旧版本兼容。Windows 请使用原生 `codex.exe` 或 WSL；为避免批处理参数解释风险，不执行 `.cmd` / `.bat` 启动器。

## 不消耗模型额度的检查

`check_available()` 只调用 CLI 帮助和登录状态，不生成文本，也不会读取令牌内容：

```python
from tradingagents.llm_clients.codex_cli_client import CodexCLIChatModel

model = CodexCLIChatModel(command="codex")  # 或官方可执行文件完整路径
print(model.check_available())
# {'provider': 'codex_cli', 'authentication': 'chatgpt', 'status': 'ready'}
```

`ready` 只表示安装与登录检查通过，**不表示模型可用、额度充足或真实分析已验证**。实际执行 `model.invoke(...)` 或股票分析会使用 Codex 额度；不是免费无限调用，也不绕过订阅限额。若账户启用了额外 credits，其消耗仍遵循 Codex 账户规则，本后端不管理账单。

## 边界与失败方式

- 仅接受 ChatGPT 登录。子进程不接收模型 API Key 或模型端点覆盖变量，且固定官方 provider 和 ChatGPT 登录方式。失败时不会转到 Platform API、中转或另一个模型。
- 每次请求在空临时目录运行，忽略个人 `config.toml`，使用只读沙盒、`never` 审批策略和临时会话。禁用 shell、个人插件、hooks、记忆、子代理、自动技能工具安装及网页搜索；不修改全局 Codex 配置。
- 报告和工具请求通过 JSON 输出契约传回；未知工具、缺少参数、坏 JSON、错误决策、超时、退出失败均停止。主程序注入的股票代码等参数仍归主程序处理，不能让模型覆盖。
- 本后端错误不会触发 TradingAgents 原有的“结构化失败后再试自由文本”。其他 provider 的原有回退行为不变。
- 同一个 Python 进程内串行调用 Codex，避免多个分析师争用订阅会话。不同进程之间不共享此锁；不要同时启动多份分析。超时按每次 Codex 调用计算，不是整份多角色报告的总时限。
- 当前支持文本消息、现有工具回合和 Pydantic 结构化决策。不支持图片输入、真正的逐 token 流式输出、任意 API 参数或自动重试。只记录 CLI 实际返回的 token 数，不伪造零费用。
- **本地 CLI 不等于本地模型**：提示词和选取的金融资料仍发送给 OpenAI。行情和新闻数据源也仍需要网络，其 Key、权限、收费及限额与模型接入分开。
- 不添加定时任务，不接入券商或真实下单。不要把个人登录态放进公开 CI、公共服务器、镜像或 GitHub Secrets；仓库原有 CI 只跑模拟测试。

## 验证与尚未验证

自动测试使用合成数据及临时假 CLI，不访问模型或交易账户。覆盖模型工厂、两个模型层、配置、环境过滤、CLI 检查、标准输入/最终文件、工具调用往返、主程序注入参数、结构化决策、无 API 回退、超时、输出上限、原生工具异常事件及完整多角色流程。

本次验证使用 Python 3.13 / macOS。**尚未执行真实订阅模型生成或真实行情的端到端分析；Windows 和 Linux 的实机 CLI 运行也未验证。** 原生工具禁用设置已按上述安装版支持的配置核对，但模拟测试不能替代真实 CLI 执行验证。保持实验性标识，先做一次小规模合成输入验证，再运行股票分析。

```bash
pytest -q tests/test_codex_cli.py tests/test_graph_end_to_end.py
```

## 官方依据

- [Codex 登录方式：ChatGPT 订阅与 API Key 分开](https://developers.openai.com/codex/auth)
- [官方非交互模式 codex exec](https://developers.openai.com/codex/noninteractive)
- [CLI 参数：输出契约、临时会话和忽略用户配置](https://developers.openai.com/codex/cli/reference)
- [配置参考：provider、登录方式和功能开关](https://developers.openai.com/codex/config-reference)

这里提供的是本 fork 的兼容实现，不代表 OpenAI 或 TradingAgents 上游认可投资分析结果。研究报告不保证投资收益。
