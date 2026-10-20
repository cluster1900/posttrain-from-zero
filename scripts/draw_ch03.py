#!/usr/bin/env python3
"""第 3 章原创结构示意图。无实测数据，不绘制虚构指标；仅使用标准库。"""
from html import escape
from pathlib import Path

DATA_SOURCE = "docs/apexo/00_progress.md §1.2、§6.1–§6.4（提交 ca57456）；历史实现 fbf621e"
OUT = Path(__file__).resolve().parents[1] / "assets/images"
FONT = "Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei, sans-serif"


class Figure:
    def __init__(self, title, height):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img">',
                      f'<title>{escape(title)}</title>', f'<rect width="1200" height="{height}" fill="#f7f8fc"/>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#52657e"/></marker></defs>',
                      f'<g font-family="{FONT}" fill="#18243b">']
        self.text(36, 38, title, 27, bold=True)

    def text(self, x, y, value, size=20, center=False, bold=False, color="#18243b"):
        anchor = "middle" if center else "start"
        weight = "bold" if bold else "normal"
        self.parts.append(f'<text x="{x}" y="{y}" dy="0.35em" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')

    def box(self, x, y, w, h, title, lines=(), tint="#ffffff"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{tint}" stroke="#a0b2cd"/>')
        self.text(x+w/2, y+30, title, 22, center=True, bold=True)
        for i, line in enumerate(lines):
            self.text(x+w/2, y+66+28*i, line, 18, center=True)

    def arrow(self, path):
        self.parts.append(f'<path d="{path}" fill="none" stroke="#52657e" stroke-width="2.3" marker-end="url(#arrow)"/>')

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text("\n".join(self.parts + ["</g></svg>"]) + "\n", encoding="utf-8")


def pipeline():
    f = Figure("从占位流程到可追踪的更新链路", 475)
    f.text(36, 87, "v1：有流程外观，缺少学习链路", 22, bold=True, color="#a34234")
    f.text(36, 263, "v2：输出、奖励、概率与参数更新必须接通", 22, bold=True, color="#087f83")
    rows = [
        (110, [("载入模型", ["权重确实被加载"]), ("固定回答", ["dummy_response"]),
               ("打印奖励", ["没有损失与反传"]), ("保存完成产物", ["不能证明发生了学习"])]),
        (290, [("真实多轮生成", ["保存行为 logprob"]), ("沙箱与验证器", ["真实结果奖励"]),
               ("DPPO 与反传", ["更新 LoRA 参数"]), ("重新同步与评测", ["检验链路与效果"])])]
    for y, boxes in rows:
        for i, (title, lines) in enumerate(boxes):
            x = 36 + i*294
            f.box(x, y, 246, 111, title, lines, "#fff1ed" if y == 110 else "#e9f5f2")
            if i < 3:
                f.arrow(f"M{x+248} {y+55} H{x+282}")
    f.text(36, 444, "结构示意。v2 代码具备链路，不代表当时已经完成 GPU、真实沙箱或模型质量验收。", 19)
    f.save("ch03_pipeline.svg")


def evidence():
    f = Figure("验证结论必须停留在证据覆盖的层级", 405)
    boxes = [
        ("静态审核", ["代码路径与接口", "不能证明路径被执行"]),
        ("CPU 单元测试", ["受控输入与边界", "不能证明 GPU 兼容"]),
        ("目标环境冒烟", ["真实生成与沙箱", "不能证明质量提升"]),
        ("正式质量评测", ["同任务集、完整日志", "再判断效果与门禁"])]
    for i, (title, lines) in enumerate(boxes):
        x = 36 + 294*i
        f.box(x, 112, 246, 146, title, lines)
        if i < 3:
            f.arrow(f"M{x+248} 185 H{x+282}")
    f.text(36, 306, "PR #1 的已验范围：73 项 CPU 测试，以及编译、Shell 语法、空白检查。", 21)
    f.text(36, 349, "其 GPU / vLLM / Harbor 验证明确待完成；本图不按测试数量推算模型能力。", 19)
    f.save("ch03_evidence.svg")


def memory():
    f = Figure("同卡串行：生成阶段与更新阶段分开占用 GPU", 470)
    f.box(36, 93, 535, 255, "生成阶段", [], "#e9f5f2")
    f.box(629, 93, 535, 255, "更新阶段", [], "#eef0fb")
    f.text(65, 166, "GPU：vLLM 权重与 KV cache", 23)
    f.text(65, 211, "CPU：HF 训练器、优化器状态", 23)
    f.text(65, 257, "冻结输出头保留 CPU 主副本", 21)
    f.text(65, 307, "输出：轨迹与行为 logprob", 21)
    f.text(658, 166, "GPU：HF 训练器与本步计算", 23)
    f.text(658, 211, "vLLM：进入 sleep 模式", 23)
    f.text(658, 257, "冻结输出头按阶段搬入", 21)
    f.text(658, 307, "输出：更新后的 LoRA 参数", 21)
    f.arrow("M574 222 H620")
    f.arrow("M895 352 V392 H303 V355")
    f.text(599, 422, "下一轮前：推送新权重、唤醒缓存、重置前缀缓存并检查同步", 21, center=True)
    f.save("ch03_memory_phases.svg")


def evaluation():
    f = Figure("评测先验输入，再判质量：缺证据不能补一个成功", 550)
    f.box(36, 166, 220, 120, "读取真实结果", ["检查输入完整性"])
    branches = [
        (76, "没有 trial", ["退出码 2", "不写新报告"], "#fff1ed"),
        (218, "证据不完整", ["timing 缺失或损坏", "任务集合不一致"], "#fff1ed"),
        (360, "证据齐全且可比", ["继续计算指标", "进入正式质量门禁"], "#e9f5f2")]
    for y, title, lines, tint in branches:
        f.box(414, y, 300, 115, title, lines, tint)
        f.arrow(f"M260 226 H337 V{y+57} H406")
    f.box(816, 218, 348, 150, "禁止的捷径", ["无日志 → 造模拟样本", "有报告文件 → 当成 PASSED"], "#fff1ed")
    f.text(36, 518, "教学完整性报告也明确标为 DEMO_ONLY；代码路径测试与真实评测报告不可混用。", 19)
    f.save("ch03_eval_gate.svg")


def main():
    pipeline()
    evidence()
    memory()
    evaluation()
    print("Generated 4 chapter-3 SVG schematics")


if __name__ == "__main__":
    main()
