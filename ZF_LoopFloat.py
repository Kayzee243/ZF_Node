from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple, Dict, Any, Union

class ZF_LoopFloat:
    """生成浮点数列表的ComfyUI自定义节点。
    支持正向和反向遍历,可控制步长和精度。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "from_this": ("FLOAT", {
                    "default": 0.00,
                    "min": 0.00,
                    "max": 10.00,
                    "step": 0.01
                }),
                "to_that": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.00,
                    "max": 10.00,
                    "step": 0.01
                }),
                "jump": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.00,
                    "max": 10.00,
                    "step": 0.01
                }),
                "jump_switch": (["enable", "disable"],),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "create_ZF_loop_float_copilot"
    CATEGORY = "ZF_node2"

    def create_ZF_loop_float_copilot(self, from_this: float, to_that: float, 
                            jump: float, jump_switch: str) -> Tuple[List[float]]:
        """生成浮点数列表。

        Args:
            from_this: 起始值
            to_that: 结束值  
            jump: 步长
            jump_switch: 是否启用步进

        Returns:
            包含浮点数序列的元组
        """
        try:
            # 初始化存储结果的空列表
            range_values = []
            # 将输入值转换为Decimal类型并保留两位小数，确保精确计算
            current = Decimal(str(from_this)).quantize(Decimal('0.01'))
            target = Decimal(str(to_that)).quantize(Decimal('0.01')) 
            step = Decimal(str(jump)).quantize(Decimal('0.01'))
            
            # 如果步进开关关闭，只返回起始值
            if jump_switch == "disable":
                return ([float(current)],)
                
            # 根据起始值和目标值的大小关系确定步长方向
            if current > target:
                step = -abs(step)  # 反向遍历使用负步长
            else:
                step = abs(step)   # 正向遍历使用正步长
                
            # 主循环：根据步长正负决定遍历方向和终止条件
            while (step > 0 and current <= target) or \
                  (step < 0 and current >= target):
                # 将当前值添加到结果列表中，转换回float类型
                range_values.append(float(current))
                # 按步长更新当前值
                current += step

                # 如果最后一个值不等于目标值,则添加目标值
            if range_values[-1] != float(target):
                range_values.append(float(target))
            
            # 返回结果列表的元组
            return (range_values,)
            
        except Exception as e:
            print(f"Error in create_ZF_loop_float_copliot: {str(e)}")
            return ([0.0],)