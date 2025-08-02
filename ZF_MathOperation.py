import torch
from typing import Literal, Union, Any
from decimal import Decimal, getcontext
import math

class ZF_MathOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"default": "0.000"}),
                "b": ("STRING", {"default": "1.000"}),
                "operation": ([
                    "+", "-", "×", "÷",  # 基础运算
                    "^", "√", "log",     # 幂运算和对数
                    "%", "max", "min"     # 其他运算
                ], {"default": "+"}),
                "precision": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            },
            "hidden": {"widget": "WIDGET"}  # 添加隐藏的结果显示组件
        }

    RETURN_TYPES = ("FLOAT", "STRING",)  # 返回计算值和显示字符串
    RETURN_NAMES = ("Float", "display",)
    FUNCTION = "calculate"
    CATEGORY = "math"

    def auto_convert(self, value: Any) -> Union[Decimal, int, float]:
            """自动类型转换"""
            try:
                # 尝试转换为Decimal
                return Decimal(str(value))
            except:
                # 如果失败，尝试eval计算
                try:
                    return eval(str(value))
                except:
                    raise ValueError(f"无法转换值: {value}")

    def calculate(self, a, b, operation, precision):
            try:
                # 设置精度
                getcontext().prec = precision + 10  # 额外精度用于计算
                
                # 自动转换输入值
                x = self.auto_convert(a)
                y = self.auto_convert(b)
                
                # 处理除法异常
                if operation == "divide" and y == 0:
                    raise ValueError("除数不能为零")
                
                operations = {
                    "+": lambda x,y: x + y,
                    "-": lambda x,y: x - y,
                    "×": lambda x,y: x * y,
                    "÷": lambda x,y: x / y if y != 0 else Decimal('inf'),
                    "^": lambda x,y: x ** y,
                    "√": lambda x,y: x ** (Decimal('1') / y) if y != 0 else Decimal('inf'),
                    "log": lambda x,y: Decimal(str(math.log(float(x), float(y)))) if x > 0 and y > 0 and y != 1 else Decimal('inf'),
                    "%": lambda x,y: x % y if y != 0 else Decimal('inf'),
                    "max": lambda x,y: max(x, y),
                    "min": lambda x,y: min(x, y)
                }
                
                # 检查除零错误
                if (operation in ["÷", "√", "%"] and y == 0) or \
                (operation == "log" and (x <= 0 or y <= 0 or y == 1)):
                    raise ValueError("无效的运算参数")

                # 计算结果并控制精度
                result = Decimal(float(operations[operation](x, y))).quantize(
                    Decimal(f"0.{'0' * precision}")
                )
                # 格式化显示结果
                display = f"{a} {operation} {b} = {result}"
            
                # 更新widget显示
                self.widget = display
                
                return (str(result), display)
                
            except Exception as e:
                error_msg = f"计算错误: {str(e)}"
                self.widget = error_msg
                return ("0.0", error_msg)