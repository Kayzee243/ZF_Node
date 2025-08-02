import torch
import numpy as np
from PIL import Image, ImageOps

class ZF_LayerMaster:
    """
    一个ComfyUI自定义节点，可以将最多四个纹理/贴图图像合成到底图上。
    每个图层都有一个总开关，并可以独立进行缩放、位置、旋转调整，连接独立的蒙版和反转蒙版。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        transform_controls = {
            "enable_texture": ("BOOLEAN", {"default": True}),
            "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
            "x_percent": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01}),
            "y_percent": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01}),
            "invert_mask": ("BOOLEAN", {"default": False})
        }

        return {
            "required": {
                "base_image": ("IMAGE",),
                "enable_texture_1": transform_controls["enable_texture"],
                "texture_1": ("IMAGE",),
                "invert_mask_1": transform_controls["invert_mask"],
                "scale_1": transform_controls["scale"],
                "rotation_1": transform_controls["rotation"],
                "x_percent_1": transform_controls["x_percent"],
                "y_percent_1": transform_controls["y_percent"],
                "enable_texture_2": transform_controls["enable_texture"],
                "texture_2": ("IMAGE",),
                "invert_mask_2": transform_controls["invert_mask"],
                "scale_2": transform_controls["scale"],
                "rotation_2": transform_controls["rotation"],
                "x_percent_2": transform_controls["x_percent"],
                "y_percent_2": transform_controls["y_percent"],
                "enable_texture_3": transform_controls["enable_texture"],
                "texture_3": ("IMAGE",),
                "invert_mask_3": transform_controls["invert_mask"],
                "scale_3": transform_controls["scale"],
                "rotation_3": transform_controls["rotation"],
                "x_percent_3": transform_controls["x_percent"],
                "y_percent_3": transform_controls["y_percent"],
                "enable_texture_4": transform_controls["enable_texture"],
                "texture_4": ("IMAGE",),
                "invert_mask_4": transform_controls["invert_mask"],
                "scale_4": transform_controls["scale"],
                "rotation_4": transform_controls["rotation"],
                "x_percent_4": transform_controls["x_percent"],
                "y_percent_4": transform_controls["y_percent"],
            },
            "optional": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "mask_3": ("MASK",),
                "mask_4": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_layers"
    CATEGORY = "Image/Layering"

    def tensor_to_pil(self, tensor):
        if tensor is None: return None
        # 从批处理张量中取出第一张图像，并转换为numpy数组
        image_np = tensor[0].cpu().numpy()
        # 将0-1范围的浮点数转换为0-255范围的8位整数
        image_np = (image_np * 255).astype(np.uint8)
        
        # 根据数组维度和通道数创建Pillow图像
        if image_np.ndim == 2: # (H, W) -> 灰度图/蒙版
            return Image.fromarray(image_np, mode='L')
        elif image_np.ndim == 3: # (H, W, C)
            if image_np.shape[2] == 1: # (H, W, 1) -> 灰度图/蒙版
                return Image.fromarray(image_np.squeeze(axis=2), mode='L')
            elif image_np.shape[2] == 3: # (H, W, 3) -> RGB彩色图
                return Image.fromarray(image_np, mode='RGB')
            elif image_np.shape[2] == 4: # (H, W, 4) -> RGBA带透明通道图
                return Image.fromarray(image_np, mode='RGBA')
        
        raise ValueError(f"Unsupported tensor shape: {image_np.shape}")

    def pil_to_tensor(self, pil_image):
        if pil_image is None: return None
        # 修正：使用更稳健的convert方法处理透明度，避免乱码
        if pil_image.mode != 'RGB':
             pil_image = pil_image.convert("RGB")
        
        # 转换为numpy数组并归一化
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        # 增加批处理维度并转换为torch张量
        return torch.from_numpy(image_np).unsqueeze(0)

    def composite_layers(self, base_image, 
                         enable_texture_1, texture_1, invert_mask_1, scale_1, rotation_1, x_percent_1, y_percent_1,
                         enable_texture_2, texture_2, invert_mask_2, scale_2, rotation_2, x_percent_2, y_percent_2,
                         enable_texture_3, texture_3, invert_mask_3, scale_3, rotation_3, x_percent_3, y_percent_3,
                         enable_texture_4, texture_4, invert_mask_4, scale_4, rotation_4, x_percent_4, y_percent_4,
                         mask_1=None, mask_2=None, mask_3=None, mask_4=None):
        
        base_pil = self.tensor_to_pil(base_image).convert('RGBA')

        textures_data = [
            (enable_texture_1, texture_1, mask_1, invert_mask_1, scale_1, rotation_1, x_percent_1, y_percent_1),
            (enable_texture_2, texture_2, mask_2, invert_mask_2, scale_2, rotation_2, x_percent_2, y_percent_2),
            (enable_texture_3, texture_3, mask_3, invert_mask_3, scale_3, rotation_3, x_percent_3, y_percent_3),
            (enable_texture_4, texture_4, mask_4, invert_mask_4, scale_4, rotation_4, x_percent_4, y_percent_4),
        ]

        output_pil = base_pil.copy()

        for enable_bool, tex_tensor, mask_tensor, invert_bool, scale, rotation, x_p, y_p in textures_data:
            if not enable_bool or tex_tensor is None: continue

            tex_pil = self.tensor_to_pil(tex_tensor)
            mask_pil = self.tensor_to_pil(mask_tensor)

            if invert_bool and mask_pil:
                mask_pil = ImageOps.invert(mask_pil)

            new_width = int(tex_pil.width * scale)
            new_height = int(tex_pil.height * scale)
            if new_width == 0 or new_height == 0: continue
            
            tex_transformed = tex_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if mask_pil:
                mask_transformed = mask_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if rotation != 0:
                # 确保贴图是RGBA以支持透明填充
                if tex_transformed.mode != 'RGBA':
                    tex_transformed = tex_transformed.convert('RGBA')
                tex_transformed = tex_transformed.rotate(rotation, expand=True, fillcolor=(0,0,0,0), resample=Image.Resampling.BICUBIC)
                
                if mask_pil:
                    # 蒙版是单通道'L'模式，填充色为0（黑色）
                    mask_transformed = mask_transformed.rotate(rotation, expand=True, fillcolor=0, resample=Image.Resampling.BICUBIC)
            
            # 确保贴图是RGBA模式以应用蒙版
            if tex_transformed.mode != 'RGBA':
                tex_transformed = tex_transformed.convert('RGBA')

            if mask_pil:
                # 确保蒙版尺寸和旋转后的贴图尺寸完全一致
                mask_transformed = mask_transformed.resize(tex_transformed.size, Image.Resampling.LANCZOS)
                tex_transformed.putalpha(mask_transformed)

            base_width, base_height = output_pil.size
            paste_x = int(base_width * x_p - tex_transformed.width / 2)
            paste_y = int(base_height * y_p - tex_transformed.height / 2)
            
            output_pil.paste(tex_transformed, (paste_x, paste_y), tex_transformed)

        final_tensor = self.pil_to_tensor(output_pil)
        
        return (final_tensor,)