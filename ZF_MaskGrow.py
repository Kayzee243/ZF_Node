import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps

class ZF_MaskGrow:
    """
    一个ComfyUI自定义节点，用于扩大或反转蒙版。
    可以将处理后的蒙版作为形状，用一张输入的图像进行填充，并可自定义蒙版底色。
    [Numpy Rewrite Version]
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "expansion_size": ("INT", { "default": 10, "min": 0, "max": 1024, "step": 1 }),
                "mask_color": ("STRING", {"default": "#FFFFFF"}),
            },
            "optional": {
                "image_to_paste": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "process_mask"
    CATEGORY = "Mask/Processing"

    def tensor_to_pil(self, tensor):
        if tensor is None: return None
        return Image.fromarray(np.clip(255. * tensor[0].cpu().numpy(), 0, 255).astype(np.uint8))

    def pil_to_tensor(self, pil_image):
        if pil_image is None: return None
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def pil_mask_to_tensor(self, pil_image):
        if pil_image is None: return None
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def process_mask(self, mask, invert_mask, expansion_size, mask_color, image_to_paste=None):
        # 1. 准备蒙版
        mask_pil = self.tensor_to_pil(mask)
        if mask_pil is None: return (None, None)
        if invert_mask: mask_pil = ImageOps.invert(mask_pil)
        if expansion_size > 0:
            kernel_size = expansion_size * 2 + 1
            expanded_mask_pil = mask_pil.filter(ImageFilter.MaxFilter(kernel_size))
        else:
            expanded_mask_pil = mask_pil

        # 将处理好的蒙版转换为Numpy数组 (0-1范围)
        mask_np = np.array(expanded_mask_pil).astype(np.float32) / 255.0
        h, w = mask_np.shape

        # 2. 准备颜色
        try:
            hex_color = mask_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            color_to_fill = np.array([r, g, b, 255], dtype=np.uint8)
        except:
            color_to_fill = np.array([255, 255, 255, 255], dtype=np.uint8)

        # 3. [核心重写] 直接用Numpy构建最终图像
        # 创建一个4通道的透明画布 (RGBA)
        final_array = np.zeros((h, w, 4), dtype=np.uint8)

        # 找到蒙版中所有“白色”（非黑）的区域
        mask_area = mask_np > 0

        # 将这些区域填充上指定的颜色
        final_array[mask_area] = color_to_fill

        # 4. 如果有图像输入，则直接覆盖像素
        if image_to_paste is not None:
            pasted_tensor = image_to_paste[0]
            pasted_np = np.clip(255. * pasted_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            
            # 如果输入图像是3通道的RGB，为其增加一个不透明的Alpha通道
            if pasted_np.shape[2] == 3:
                alpha_channel = np.full((h, w, 1), 255, dtype=np.uint8)
                pasted_np = np.concatenate((pasted_np, alpha_channel), axis=2)

            # 找到输入图像中所有不透明的区域
            pasted_alpha_area = pasted_np[:, :, 3] > 0

            # 找到最终需要被覆盖的区域（蒙版区域 AND 输入图像不透明区域）
            composite_area = mask_area & pasted_alpha_area
            
            # 直接将输入图像的像素值复制到最终画布的对应区域
            final_array[composite_area] = pasted_np[composite_area]

        # 5. 将最终的Numpy数组转换回Pillow图像，再转换为Tensor
        final_pil = Image.fromarray(final_array, 'RGBA')
        
        output_image_tensor = self.pil_to_tensor(final_pil)
        output_mask_tensor = self.pil_mask_to_tensor(expanded_mask_pil)
        
        return (output_image_tensor, output_mask_tensor)