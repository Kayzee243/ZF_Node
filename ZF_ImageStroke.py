import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2 # 确保已安装 opencv-python

class ZF_ImageStroke:
    """
    一个为图像添加内外描边的ComfyUI自定义节点。
    [终极简化修复版]
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "stroke_type": (["Outer", "Inner", "Center"],),
                "stroke_size": ("INT", {"default": 10, "min": 1, "max": 256, "step": 1}),
                "stroke_color": ("STRING", {"default": "#FFFFFF"}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "output_mask_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask_out",)
    FUNCTION = "apply_stroke"
    CATEGORY = "Image/Processing"

    def tensor_to_pil(self, tensor, is_mask=False):
        if tensor is None: return None
        image_np = tensor[0].cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        if is_mask:
            return Image.fromarray(image_np, mode='L')
        
        if image_np.ndim == 2: return Image.fromarray(image_np, mode='L')
        elif image_np.ndim == 3:
            if image_np.shape[2] == 1: return Image.fromarray(image_np.squeeze(axis=2), mode='L')
            elif image_np.shape[2] == 3: return Image.fromarray(image_np, mode='RGB')
            elif image_np.shape[2] == 4: return Image.fromarray(image_np, mode='RGBA')
        raise ValueError(f"Unsupported tensor shape: {image_np.shape}")

    def pil_to_tensor(self, pil_image):
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert("RGBA")
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def apply_stroke(self, image, stroke_type, stroke_size, stroke_color, feather, invert_mask, output_mask_only, mask=None):
        # 1. 准备图像和蒙版
        original_image_pil = self.tensor_to_pil(image)
        if original_image_pil.mode != 'RGBA':
            original_image_pil = original_image_pil.convert('RGBA')

        if mask is not None:
            mask_pil = self.tensor_to_pil(mask, is_mask=True)
        else:
            mask_pil = original_image_pil.getchannel('A')

        if invert_mask:
            mask_pil = ImageOps.invert(mask_pil)
        
        mask_np = np.array(mask_pil)

        # 2. 生成描边形状
        kernel = np.ones((3, 3), np.uint8)
        if stroke_type == "Outer":
            dilated = cv2.dilate(mask_np, kernel, iterations=stroke_size)
            stroke_shape_np = cv2.subtract(dilated, mask_np)
        elif stroke_type == "Inner":
            eroded = cv2.erode(mask_np, kernel, iterations=stroke_size)
            stroke_shape_np = cv2.subtract(mask_np, eroded)
        else: # Center
            half_size = max(1, stroke_size // 2)
            dilated = cv2.dilate(mask_np, kernel, iterations=half_size)
            eroded = cv2.erode(mask_np, kernel, iterations=half_size)
            stroke_shape_np = cv2.subtract(dilated, eroded)

        # 3. 准备颜色和羽化
        try:
            hex_color = stroke_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            r, g, b = (255, 255, 255)

        stroke_shape_pil = Image.fromarray(stroke_shape_np, mode='L')
        if feather > 0:
            stroke_shape_pil = stroke_shape_pil.filter(ImageFilter.GaussianBlur(feather))

        # 4. [核心逻辑重写] 使用最简单、最基础的粘贴方法
        # 创建一个纯色的图层用于填充描边
        stroke_color_fill = Image.new("RGBA", original_image_pil.size, (r, g, b, 255))
        
        # 创建一个透明的画布，作为最终输出的底层
        final_image_pil = Image.new("RGBA", original_image_pil.size, (0, 0, 0, 0))
        
        # 第一步：将颜色填充层，通过描边形状作为蒙版，粘贴到底层画布上。
        # 这一步完成后，final_image_pil 上就只有带颜色的描边。
        final_image_pil.paste(stroke_color_fill, (0, 0), stroke_shape_pil)

        # 第二步：如果不是只输出蒙版，则将原始图像完整地粘贴到描边层之上。
        if not output_mask_only:
            # 这一步使用图像自身的透明通道进行粘贴，将图像叠加在描边之上。
            final_image_pil.paste(original_image_pil, (0, 0), original_image_pil)
        
        # 5. 输出
        output_image_tensor = self.pil_to_tensor(final_image_pil)
        output_mask_tensor = self.pil_to_tensor(stroke_shape_pil.convert("RGBA"))
        
        return (output_image_tensor, output_mask_tensor)