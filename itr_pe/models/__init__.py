try:
    from models.isnet import ISNetGTEncoder, ISNetDIS
except ImportError:
    from .isnet import ISNetGTEncoder, ISNetDIS
