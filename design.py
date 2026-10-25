"""
设计智能体的输入输出格式以及提示词
"""
import dspy

PROMPT_PREMISE = ("你要解决的问题是优化一个二维机翼的气动性能，下文中的所有公式的符号均需遵循以下规则："
                  "升力系数：C_L"
                  "阻力系数：C_D"
                  "上表面前缘半径 R_up"
                  "下表面前缘半径 R_low"
                  "上表面最大厚度位置 X_up"
                  "上表面最大厚度 Y_up"
                  "上表面最大厚度处曲率 z_up"
                  "下表面最大厚度位置 X_low"
                  "下表面最大厚度 Y_low"
                  "下表面最大厚度处曲率 z_low"
                  "后缘方向角 alpha_te"
                  "后缘楔角 beta_te"
                  "后缘高度 z_te"
                  "后缘厚度 Delta z_te"
                  "不得引入除上文外的任何物理参数")

INPUT_REQUIREMENTS = "输入一段话，用自然语言描述了翼形条件的约束"
OUTPUT_REQUIREMENTS = ("将输入转化为数学约束,每条约束包含一个等式或者不等式。每行一条约束，每条约束用纯"
                        "数学（latex格式）的公式给出且两侧被$包裹，使用max，min等优化算子时，函数放在其正右方用（）包裹，"
                        "变量不用表示出来，每行以回车符结尾")
class Read(dspy.Signature):
    requirements: str = dspy.InputField(desc=INPUT_REQUIREMENTS)
    fomula: str = dspy.OutputField(desc=OUTPUT_REQUIREMENTS)

Read = Read.with_instructions(PROMPT_PREMISE)
