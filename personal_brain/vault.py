from __future__ import annotations

import base64
import ctypes
import hashlib
import os
import sqlite3
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from .schema import BrainSchema


CRYPTPROTECT_UI_FORBIDDEN = 0x01
KDF_NAME = "pbkdf2_hmac_sha256"
KDF_ITERATIONS = 260_000
ENCRYPTION_SCHEME = "windows_dpapi_with_master_password_entropy_v1"


@dataclass(frozen=True)
class SecureItemSummary:
    id: int
    label: str
    secret_type: str
    username: str | None
    note: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SecureItemSecret:
    summary: SecureItemSummary
    secret: str


class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class SecureVault:
    """Local encrypted vault for secrets.

    Secrets must never go through AI extraction, Router manifests, embeddings,
    or Markdown exports. This class stores only encrypted values in SQLite.
    """

    def __init__(self, schema: BrainSchema):
        if os.name != "nt":
            raise RuntimeError("SecureVault V0 uses Windows DPAPI and requires Windows.")
        self.schema = schema

    def add_item(
        self,
        label: str,
        secret_type: str,
        secret: str,
        master_password: str,
        username: str | None = None,
        note: str | None = None,
    ) -> int:
        clean_label = required_text(label, "label")
        clean_secret_type = required_text(secret_type, "secret_type")
        clean_secret = required_text(secret, "secret")
        clean_master = required_text(master_password, "master_password")

        self.schema.initialize()
        salt = os.urandom(16)
        entropy = derive_entropy(clean_master, salt)
        encrypted = dpapi_protect(clean_secret.encode("utf-8"), entropy)

        with self.schema.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO secure_items (
                    label, secret_type, username, encrypted_value,
                    encryption_scheme, kdf_name, kdf_salt, kdf_iterations, note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(label) DO UPDATE SET
                    secret_type = excluded.secret_type,
                    username = excluded.username,
                    encrypted_value = excluded.encrypted_value,
                    encryption_scheme = excluded.encryption_scheme,
                    kdf_name = excluded.kdf_name,
                    kdf_salt = excluded.kdf_salt,
                    kdf_iterations = excluded.kdf_iterations,
                    note = excluded.note,
                    updated_at = datetime('now', 'localtime')
                """,
                (
                    clean_label,
                    clean_secret_type,
                    optional_text(username),
                    base64.b64encode(encrypted).decode("ascii"),
                    ENCRYPTION_SCHEME,
                    KDF_NAME,
                    base64.b64encode(salt).decode("ascii"),
                    KDF_ITERATIONS,
                    optional_text(note),
                ),
            )
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = conn.execute("SELECT id FROM secure_items WHERE label = ?", (clean_label,)).fetchone()
            return int(row["id"])

    def list_items(self) -> list[SecureItemSummary]:
        self.schema.initialize()
        with self.schema.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, label, secret_type, username, note, created_at, updated_at
                FROM secure_items
                ORDER BY updated_at DESC, label ASC
                """
            ).fetchall()
        return [row_to_summary(row) for row in rows]

    def get_item(self, label: str, master_password: str) -> SecureItemSecret:
        clean_label = required_text(label, "label")
        clean_master = required_text(master_password, "master_password")
        self.schema.initialize()
        with self.schema.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM secure_items
                WHERE label = ?
                """,
                (clean_label,),
            ).fetchone()
        if row is None:
            raise KeyError(f"secure item not found: {clean_label}")
        if row["encryption_scheme"] != ENCRYPTION_SCHEME:
            raise RuntimeError(f"unsupported encryption scheme: {row['encryption_scheme']}")
        if row["kdf_name"] != KDF_NAME:
            raise RuntimeError(f"unsupported kdf: {row['kdf_name']}")
        salt = base64.b64decode(row["kdf_salt"])
        entropy = derive_entropy(clean_master, salt, int(row["kdf_iterations"]))
        encrypted = base64.b64decode(row["encrypted_value"])
        try:
            secret = dpapi_unprotect(encrypted, entropy).decode("utf-8")
        except OSError as exc:
            raise ValueError("failed to decrypt secure item; master password may be wrong") from exc
        return SecureItemSecret(summary=row_to_summary(row), secret=secret)


def derive_entropy(
    master_password: str,
    salt: bytes,
    iterations: int = KDF_ITERATIONS,
) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        master_password.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )


def dpapi_protect(data: bytes, entropy: bytes) -> bytes:
    data_blob = bytes_to_blob(data)
    entropy_blob = bytes_to_blob(entropy)
    out_blob = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(data_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    return blob_to_bytes_and_free(out_blob)


def dpapi_unprotect(data: bytes, entropy: bytes) -> bytes:
    data_blob = bytes_to_blob(data)
    entropy_blob = bytes_to_blob(entropy)
    out_blob = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(data_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    return blob_to_bytes_and_free(out_blob)


def bytes_to_blob(data: bytes) -> DATA_BLOB:
    buffer = ctypes.create_string_buffer(data)
    blob = DATA_BLOB()
    blob.cbData = len(data)
    blob.pbData = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    blob._buffer = buffer  # type: ignore[attr-defined]
    return blob


def blob_to_bytes_and_free(blob: DATA_BLOB) -> bytes:
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob.pbData)


def row_to_summary(row: sqlite3.Row) -> SecureItemSummary:
    return SecureItemSummary(
        id=int(row["id"]),
        label=str(row["label"]),
        secret_type=str(row["secret_type"]),
        username=row["username"],
        note=row["note"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def required_text(value: str | None, label: str) -> str:
    text = optional_text(value)
    if not text:
        raise ValueError(f"{label} is required")
    return text


def optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

