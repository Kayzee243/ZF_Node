import torch
import nodes
import comfy.utils
from comfy.cli_args import LatentPreviewMethod
from PIL import Image, ImageOps, ImageChops

class ZF_MaskOverlayV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }

    CATEGORY = "image/postprocessing"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "overlay_mask"

    def overlay_mask(self, image, mask, alpha):
        # 将image和mask转换为Tensor格式
        image = image.clone().movedim(-1, 1)
        mask = mask.clone()
        
        # 自动缩放mask到image尺寸
        if mask.shape[-2:] != image.shape[-2:]:
            mask = comfy.utils.common_upscale(
                mask.unsqueeze(1), 
                image.shape[3], 
                image.shape[2], 
                "bilinear", 
                "center"
            ).squeeze(1)

        # 将mask扩展到3通道
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        
        # 创建彩色蒙版（这里使用红色，可以修改颜色）
        color_mask = torch.zeros_like(image)
        color_mask[:, 0] = 1.0  # R通道设为1（红色）

        # 应用透明度
        blended = image * (1 - alpha) + color_mask * mask * alpha

        # 限制数值范围
        blended = torch.clamp(blended, 0, 1)
        
        # 恢复通道顺序
        blended = blended.movedim(1, -1)
        mask = mask[:, 0].unsqueeze(-1)  # 恢复原始mask格式
        
        return (blended, mask)