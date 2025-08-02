from .ZF_LoopFloat import ZF_LoopFloat
from .ZF_MathOperation import ZF_MathOperation
from .ZF_MaskOverlayV1 import ZF_MaskOverlayV1
from .ZF_BooleanONorOFF import ZF_BooleanONorOFF
from .ZF_LayerMaster import ZF_LayerMaster
from .ZF_MaskGrow import ZF_MaskGrow
from .ZF_ImageStroke import ZF_ImageStroke

NODE_CLASS_MAPPINGS = {
    "ZF_LoopFloat":ZF_LoopFloat,
    "ZF_MathOperation":ZF_MathOperation,
    "ZF_MaskOverlayV1":ZF_MaskOverlayV1,
    "ZF_BooleanONorOFF":ZF_BooleanONorOFF,
    "ZF_LayerMaster":ZF_LayerMaster,
    "ZF_ImageStroke":ZF_ImageStroke,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZF_LoopFloat":"🌪️ ZF_LoopFloat",
    "ZF_MathOperation":"🌪️ ZF_MathOperation",
    "ZF_MaskOverlayV1":"🌪️ ZF_MaskOverlayV1",
    "ZF_BooleanONorOFF":"🌪️ ZF_BooleanONorOFF",
    "ZF_ImageStroke":"🌪️ ZF_ImageStroke",
}