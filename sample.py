from time import perf_counter

import dspy

lm = dspy.LM('deepseek/deepseek-flash', api_key=DEEPSEEK_API_KEY)
dspy.configure(lm=lm)

class Outline(dspy.Signature):
    """Outline a thorough overview of a topic."""

    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(desc="mapping from section headings to subheadings")

class DraftSection(dspy.Signature):
    """Draft a top-level section of an article."""

    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section")

class DraftArticle(dspy.Module):
    def __init__(self):
        super().__init__()
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        started = perf_counter()
        print("正在生成大纲……", flush=True)
        outline = self.build_outline(topic=topic)
        total = len(outline.section_subheadings)
        print(
            f"大纲生成完成，共 {total} 章，耗时 {perf_counter() - started:.1f} 秒",
            flush=True,
        )
        sections = []
        for i, (heading, subheadings) in enumerate(
            outline.section_subheadings.items(), start=1
        ):
            section_started = perf_counter()
            print(f"正在生成第 {i}/{total} 章：{heading}", flush=True)
            section = self.draft_section(
                topic=outline.title,
                section_heading=f"## {heading}",
                section_subheadings=[f"### {subheading}" for subheading in subheadings],
            )
            sections.append(section.content)
            print(
                f"第 {i}/{total} 章完成，耗时 {perf_counter() - section_started:.1f} 秒",
                flush=True,
            )
        print(f"文章生成完成，总耗时 {perf_counter() - started:.1f} 秒", flush=True)
        return dspy.Prediction(title=outline.title, sections=sections)

draft_article = DraftArticle()
article = draft_article(topic="World Cup 2002")
print(article)
