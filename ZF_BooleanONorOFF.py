from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple, Dict, Any, Union

class ZF_BooleanONorOFF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_state": (["On", "Off"], {"default": "off"}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("Boolean",)
    FUNCTION = "get_boolean"
    CATEGORY = "Custom Nodes/Logic"

    def get_boolean(self, switch_state):
        return (True,) if switch_state == "On" else (False,)