from typing import Dict, Type
from app.intent.base import BaseIntent
from app.intent.cert_verify import CertVerify
from app.intent.product_find import ProductFind
from app.intent.outfit_reco import OutfitReco
from app.intent.material_explain import MaterialExplain
from app.intent.fallback import Fallback

REGISTRY: Dict[str, BaseIntent] = {
    "CERT_VERIFY":      CertVerify(),
    "PRODUCT_FIND":     ProductFind(),
    "OUTFIT_RECO":      OutfitReco(),
    "MATERIAL_EXPLAIN": MaterialExplain(),
    "FALLBACK":         Fallback(),
}

def get_intent_handler(tool: str) -> BaseIntent:
    return REGISTRY.get(tool, REGISTRY["FALLBACK"])
