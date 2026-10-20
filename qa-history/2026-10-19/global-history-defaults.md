# 全局问答保存默认配置

日期：2026-10-19（Asia/Shanghai）

## 用户问题

默认设置所有项目在任意电脑上生成问答或者历史记录随时同步保存，将该配置写到默认配置中，应用于所有的项目及以后的所有问答。

用户随后提供了更新后的 AGENTS.md，包含全局问答保存与跨电脑同步偏好。

## 实际交付的完整回答

已写入本机的全局默认设置：

- [全局规则](C:/Users/lvzhe/.codex/AGENTS.md)：所有读取该规则的项目，后续每轮问答保存到项目的 `qa-history`，并记录生成成果。
- [历史配置](C:/Users/lvzhe/.codex/config.toml)：已设置 `history.persistence = "save-all"`，配置读取检查通过。

**跨电脑实时同步尚未启用。** 这些本机设置不会自动传到其他电脑，也不能补回其他电脑已有的历史。官方 Work 同步需要单独开通，且不迁移已有聊天、与 Codex 历史分开。[官方说明](https://help.openai.com/en/articles/20001548-agent-security-and-local-work-sync-in-chatgpt)

要完成跨电脑同步，还需要可访问的另一台电脑，以及确定的官方同步渠道或共享存储位置。

## 验证与未完成事项

- 已回读全局 AGENTS.md 与 config.toml。
- 已运行本机 Codex 的配置读取命令，退出码为 0。
- 未验证新会话中的历史实际写入；history.persistence 不等于完整回答导出或跨电脑同步。
- 本记录在项目可写目录内，未修改 sources/。
- 未连接另一台电脑，未设置共享目录或服务端同步权限，未迁移历史。
