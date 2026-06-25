import os
import binascii
import tempfile

from src.crypto import encrypt, decrypt


def test_encrypt_decrypt_roundtrip(monkeypatch):
    # set a 32-byte hex key in env
    key = binascii.hexlify(b"\x11" * 32).decode()
    monkeypatch.setenv("BOT_ENCRYPTION_KEY", key)

    plaintext = b"hello rika"
    blob = encrypt(plaintext)
    assert isinstance(blob, (bytes, bytearray))
    out = decrypt(blob)
    assert out == plaintext
