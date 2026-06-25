import os
import binascii
from typing import Tuple


def _get_cipher():
    """Lazy import of AESGCM to avoid hard dependency at module load."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM


def _get_key() -> bytes:
    hex_key = os.environ.get("BOT_ENCRYPTION_KEY")
    if not hex_key:
        raise RuntimeError("BOT_ENCRYPTION_KEY not set in environment")
    return binascii.unhexlify(hex_key)


def encrypt(plaintext: bytes, associated_data: bytes | None = None) -> bytes:
    """Encrypt bytes with AES-256-GCM. Returns nonce + ciphertext (concatenated).
    Nonce length: 12 bytes.
    """
    AESGCM = _get_cipher()
    key = _get_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    return nonce + ct


def decrypt(blob: bytes, associated_data: bytes | None = None) -> bytes:
    """Decrypt a blob produced by `encrypt` (nonce + ciphertext)."""
    AESGCM = _get_cipher()
    key = _get_key()
    aesgcm = AESGCM(key)
    nonce = blob[:12]
    ct = blob[12:]
    return aesgcm.decrypt(nonce, ct, associated_data)
