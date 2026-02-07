import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
import zipfile
import tempfile
import shutil
import tarfile
import subprocess

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters,
)

try:
    import rarfile  # type: ignore
except ImportError:
    rarfile = None

try:
    import py7zr  # type: ignore
except ImportError:
    py7zr = None

# ==== CONFIG ====
# Use BOT_TOKEN from env on Railway (set in dashboard), else fallback to value below
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8380922566:AAFCAn2Y-iClE7aMtue9ypNayWro464u8tg")
DOWNLOAD_DIR = "downloads"
RESULTS_DIR = "results"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==== HELPERS ====
def get_timestamped_folder(base_filename: str) -> str:
    """Create and return a timestamped folder path like results/20260205_122345_filename/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = os.path.splitext(base_filename)[0].replace(" ", "_")[:50]
    folder_name = f"{timestamp}_{clean_name}"
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def remove_duplicates_from_file(file_path: str) -> None:
    """Remove duplicate lines from a file while preserving order."""
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        seen = set()
        unique_lines = []
        for line in lines:
            if line.strip() and line not in seen:
                seen.add(line)
                unique_lines.append(line)
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(unique_lines)
    except Exception:
        pass


def format_timedelta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


from telegram.error import BadRequest

import re


async def safe_edit(message, text):
    """Edit a Telegram message and ignore 'Message is not modified' errors."""
    try:
        await message.edit_text(text)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            return
        raise


def debug_log(msg: str) -> None:
    """Append a timestamped debug message to results/debug.log (best-effort)."""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "debug.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def find_extraction_tool() -> str | None:
    """Find and return path to 7z, UnRAR, or WinRAR on PATH (Linux + Windows) or common Windows locations."""
    # Check PATH first (Linux: 7z, 7za, unrar; Windows: .exe variants)
    for tool in ("7z", "7za", "unrar", "7z.exe", "7za.exe", "unrar.exe", "WinRAR.exe", "Rar.exe"):
        path = shutil.which(tool)
        if path:
            return path

    # Common Windows install locations (skip on Linux)
    candidates = []
    pf = os.environ.get("ProgramFiles", "C:\\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    
    for base_dir in [pf, pf86, "C:\\Program Files", "C:\\Program Files (x86)"]:
        candidates.extend([
            os.path.join(base_dir, "7-Zip", "7z.exe"),
            os.path.join(base_dir, "WinRAR", "WinRAR.exe"),
            os.path.join(base_dir, "WinRAR", "Rar.exe"),
            os.path.join(base_dir, "UnRAR", "unrar.exe"),
        ])
    
    for c in candidates:
        try:
            if c and os.path.isfile(c):
                debug_log(f"[TOOLS] Found extraction tool: {c}")
                return c
        except Exception:
            pass
    
    return None


def extract_with_available_tool(archive_path: str, extract_to_dir: str, password: str | None) -> bool:
    """Try to extract RAR with any available tool (7z, UnRAR, WinRAR). Returns True on success."""
    tool = find_extraction_tool()
    if not tool:
        debug_log(f"[TOOLS] No extraction tool found")
        return False
    
    debug_log(f"[TOOLS] Using extraction tool: {tool}")
    
    try:
        # Build command based on tool
        if "7z" in tool.lower():
            cmd = [tool, "x", "-y", f"-o{extract_to_dir}", archive_path]
            if password:
                cmd.insert(2, f"-p{password}")
        elif "unrar" in tool.lower():
            cmd = [tool, "x", "-y", "-o+"]  # -o+ means overwrite without prompt
            if password:
                cmd.append(f"-p{password}")
            cmd.extend([archive_path, extract_to_dir + os.sep])
        elif "winrar" in tool.lower() or "rar.exe" in tool.lower():
            cmd = [tool, "x", "-y", "-o+"]
            if password:
                cmd.append(f"-p{password}")
            cmd.extend([archive_path, extract_to_dir + os.sep])
        else:
            debug_log(f"[TOOLS] Unknown tool: {tool}")
            return False
        
        debug_log(f"[TOOLS] Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            debug_log(f"[TOOLS] Extraction succeeded with {os.path.basename(tool)}")
            return True
        else:
            stderr = result.stderr.strip() if result.stderr else ""
            stdout = result.stdout.strip() if result.stdout else ""
            debug_log(f"[TOOLS] Extraction failed. Return code: {result.returncode}")
            if stderr:
                debug_log(f"[TOOLS] stderr: {stderr}")
            if stdout:
                debug_log(f"[TOOLS] stdout: {stdout}")
            return False
    except Exception as e:
        debug_log(f"[TOOLS] Extraction exception: {e}")
        return False


def extract_archive_once(archive_path: str, extract_to_dir: str, password: str | None) -> bool:
    """Extract archive to a folder. Returns True on success, False on failure."""
    try:
        first_bytes = b""
        try:
            with open(archive_path, "rb") as fh:
                first_bytes = fh.read(4096)
        except Exception:
            first_bytes = b""

        def looks_like_html(b: bytes) -> bool:
            b2 = b.lstrip().lower()
            return b2.startswith(b"<!doctype html") or b2.startswith(b"<html") or b"<html" in b2[:1024]

        if looks_like_html(first_bytes):
            raise RuntimeError("Downloaded content looks like HTML (not an archive).")

        is_zip = zipfile.is_zipfile(archive_path)
        is_rar = False
        if not is_zip and rarfile is not None:
            try:
                is_rar = rarfile.is_rarfile(archive_path)  # type: ignore
            except Exception:
                is_rar = False

        is_7z = False
        if not is_zip and not is_rar:
            is_7z = first_bytes.startswith(b"7z\xbc\xaf\x27\x1c")

        is_tar = False
        if not is_zip and not is_rar and not is_7z:
            try:
                is_tar = tarfile.is_tarfile(archive_path)
            except Exception:
                is_tar = False

        if is_zip:
            with zipfile.ZipFile(archive_path) as z:
                pwd = password.encode("utf-8") if password else None
                z.extractall(extract_to_dir, pwd=pwd)
                return True
        elif is_rar:
            if rarfile is None:
                raise RuntimeError("RAR archive detected but 'rarfile' library not installed.")
            try:
                rf = rarfile.RarFile(archive_path)  # type: ignore
                if password:
                    rf.extractall(extract_to_dir, pwd=password)
                else:
                    rf.extractall(extract_to_dir)
                return True
            except Exception as e:
                debug_log(f"[EXTRACT] rarfile failed: {e}; trying tool fallback...")
                if extract_with_available_tool(archive_path, extract_to_dir, password):
                    return True
                else:
                    # Try to find what tools are available
                    tool_status = find_extraction_tool()
                    if not tool_status:
                        debug_log(f"[EXTRACT] No extraction tool found on system")
                    raise RuntimeError(
                        f"RAR extraction failed. Please install one of:\n"
                        "• 7-Zip (https://www.7-zip.org)\n"
                        "• WinRAR (https://www.winrar.com)\n"
                        "• UnRAR (https://www.rarlab.com/rar_add.htm)"
                    )
        elif is_7z:
            if py7zr is None:
                raise RuntimeError("7Z archive detected but 'py7zr' is not installed. Install with: pip install py7zr")
            try:
                if password:
                    with py7zr.SevenZipFile(archive_path, mode="r", password=password) as z:  # type: ignore
                        z.extractall(path=extract_to_dir)
                else:
                    with py7zr.SevenZipFile(archive_path, mode="r") as z:  # type: ignore
                        z.extractall(path=extract_to_dir)
                return True
            except Exception as e:
                raise RuntimeError(f"7Z extraction failed: {e}")
        elif is_tar:
            try:
                with tarfile.open(archive_path, mode="r:*") as tf:
                    tf.extractall(path=extract_to_dir)
                return True
            except Exception as e:
                raise RuntimeError(f"TAR extraction failed: {e}")
        else:
            raise RuntimeError("Unsupported archive type: not ZIP/RAR/7Z/TAR (or file is corrupted/incomplete).")
    except Exception as e:
        debug_log(f"[EXTRACT] Failed: {e}")
        raise


def search_cc_in_extracted(extracted_dir: str, result_path: str) -> int:
    """
    Search extracted folder for CreditCards files and merge them FAST.
    
    Logic:
    1. Single pass through directory tree
    2. Finds ALL "creditcards" folders (any depth) and reads their .txt files
    3. Finds ALL "creditcards.txt" files directly
    4. Deduplicates during reading (not after)
    
    Returns: count of unique CC lines written
    """
    merged_lines = 0
    seen_lines = set()
    cc_files = []  # List of tuples: (file_path, is_folder_file)
    
    try:
        # SINGLE PASS: Collect all CC-related files
        for root, dirs, files in os.walk(extracted_dir):
            # Find CreditCards folders
            if os.path.basename(root).lower() == "creditcards":
                for fname in files:
                    if fname.lower().endswith(".txt"):
                        cc_files.append((os.path.join(root, fname), True))
            
            # Find CreditCards.txt files anywhere
            for fname in files:
                if fname.lower() == "creditcards.txt":
                    cc_files.append((os.path.join(root, fname), True))
        
        # Fast file reading and deduplication
        with open(result_path, "w", encoding="utf-8", errors="ignore") as out:
            for file_path, _ in cc_files:
                if not os.path.exists(file_path) or not os.path.isfile(file_path):
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line_stripped = line.rstrip("\n").strip()
                            # Only write non-empty, unique lines
                            if line_stripped and line_stripped not in seen_lines:
                                seen_lines.add(line_stripped)
                                out.write(line_stripped + "\n")
                                merged_lines += 1
                except Exception as e:
                    debug_log(f"[CC_SEARCH] Error reading {file_path}: {e}")
                    continue
    
    except Exception as e:
        debug_log(f"[CC_SEARCH] Error: {e}")
    
    return merged_lines


def search_combos_in_extracted(extracted_dir: str, keyword: str, result_path: str) -> int:
    """Search extracted folder for combos. Synchronous."""
    combo_count = 0
    seen_combos = set()
    try:
        with open(result_path, "w", encoding="utf-8") as out:
            for root, dirs, files in os.walk(extracted_dir):
                for fname in files:
                    if not fname.lower().endswith(".txt"):
                        continue
                    
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                if keyword and keyword.lower() not in line.lower():
                                    continue
                                
                                combo = None
                                # Format 1: Username: X | Password: Y
                                if " | " in line:
                                    parts = [p.strip() for p in line.split("|")]
                                    username = None
                                    password_field = None
                                    for part in parts:
                                        if part.lower().startswith(("username:", "login:")):
                                            username = part.split(":", 1)[1].strip()
                                        elif part.lower().startswith("password:"):
                                            password_field = part.split(":", 1)[1].strip()
                                    if username and password_field:
                                        combo = f"{username}:{password_field}"
                                
                                # Format 2: url:login:pass
                                if not combo and line.count(":") >= 2:
                                    parts = line.split(":")
                                    if len(parts) >= 3:
                                        login = parts[-2].strip()
                                        passwd = parts[-1].strip()
                                        if login and passwd and not login.startswith("http") and not passwd.startswith("http"):
                                            combo = f"{login}:{passwd}"
                                
                                if combo and combo not in seen_combos:
                                    seen_combos.add(combo)
                                    out.write(combo + "\n")
                                    combo_count += 1
                    except Exception:
                        pass
    except Exception as e:
        debug_log(f"[COMBO_SEARCH] Error: {e}")
    
    remove_duplicates_from_file(result_path)
    return combo_count


def validate_creditcard(cc_number: str) -> bool:
    """Validate credit card using Luhn algorithm. Accepts 13-19 digit cards."""
    cc_number = cc_number.replace(" ", "").replace("-", "").strip()
    
    # Check if it's all digits and proper length
    if not cc_number.isdigit() or not (13 <= len(cc_number) <= 19):
        return False
    
    # Luhn algorithm
    def luhn_checksum(card_num):
        def digits_of(n):
            return [int(d) for d in str(n)]
        digits = digits_of(card_num)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10
    
    return luhn_checksum(cc_number) == 0


def clean_creditcards_file(input_file: str, output_file: str) -> int:
    """Clean CC file: validate format, Luhn algorithm, remove duplicates.
    
    Returns count of valid CCs written to output file.
    If Luhn validation finds 0 valid cards, falls back to just deduping raw data.
    """
    valid_ccs = set()
    count = 0
    all_lines = []
    
    try:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f_in:
            all_lines = f_in.readlines()
    except Exception as e:
        debug_log(f"[CC_CLEAN] Error reading input: {e}")
        return 0
    
    if not all_lines:
        return 0
    
    # First pass: try Luhn validation
    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for line in all_lines:
                # Handle various CC formats: "4111111111111111|CVV|MM/YY" or just "4111111111111111"
                parts = line.strip().split("|")
                if not parts:
                    continue
                
                cc_number = parts[0].strip()
                
                # Validate CC number with Luhn
                if not validate_creditcard(cc_number) or cc_number in valid_ccs:
                    continue
                
                valid_ccs.add(cc_number)
                
                # Write full line (with CVV/date if present)
                f_out.write(line.rstrip("\n") + "\n")
                count += 1
    except Exception as e:
        debug_log(f"[CC_CLEAN] Error writing output: {e}")
    
    # If Luhn validation failed (0 results), fallback to raw dedup
    if count == 0:
        debug_log(f"[CC_CLEAN] Luhn validation returned 0 valid cards; using raw dedup fallback")
        seen_lines = set()
        try:
            with open(output_file, "w", encoding="utf-8") as f_out:
                for line in all_lines:
                    line_stripped = line.rstrip("\n").strip()
                    if line_stripped and line_stripped not in seen_lines:
                        seen_lines.add(line_stripped)
                        f_out.write(line_stripped + "\n")
                        count += 1
        except Exception as e:
            debug_log(f"[CC_CLEAN] Fallback error: {e}")
    
    debug_log(f"[CC_CLEAN] Cleaned CC file: {count} valid/unique cards (method: {'Luhn' if count == valid_ccs else 'raw dedup'})")
    return count


async def download_file(
    url: str,
    dest_path: str,
    progress_message,
    context: ContextTypes.DEFAULT_TYPE,
    retries: int = 3,
):
    """Stream download with progress updates to Telegram.

    Enhanced behavior:
    - Respects system proxy env (trust_env=True)
    - Retries on transient errors with exponential backoff
    - On SSL errors, retries once with SSL verification disabled (useful when Windows reports "Access is denied")
    - Supports stop flag to cancel download
    """
    chunk_size = 1024 * 1024 * 64  # 64 MB for max download speed (Railway/server)

    # ssl_setting: None -> default verification, False -> disable verification
    ssl_setting = None

    for attempt in range(1, retries + 1):
        # Check if user requested stop
        if context.user_data.get("stop_requested", False):
            raise RuntimeError("Download cancelled by user")
        
        start = asyncio.get_running_loop().time()
        downloaded = 0
        last_update = start
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                timeout = aiohttp.ClientTimeout(total=None)  # No timeout - infinite
                async with session.get(url, timeout=timeout, ssl=ssl_setting) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")

                    total = int(resp.headers.get("Content-Length", 0))

                    # write to a temp file first to avoid leaving partial file on failures
                    tmp_path = dest_path + ".part"
                    with open(tmp_path, "wb") as f:
                        while True:
                            # Check if stop was requested
                            if context.user_data.get("stop_requested", False):
                                raise RuntimeError("Download cancelled by user")
                            
                            chunk = await resp.content.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)

                            now = asyncio.get_running_loop().time()
                            # Update message every 2 seconds or on finish
                            if now - last_update >= 2 or (total > 0 and downloaded == total):
                                elapsed = now - start
                                speed = downloaded / elapsed if elapsed > 0 else 0  # bytes/s
                                speed_mb = speed / 1024 / 1024

                                if total > 0 and speed > 0:
                                    eta = (total - downloaded) / speed
                                    eta_text = format_timedelta(eta)
                                    total_mb = total / 1024 / 1024
                                else:
                                    eta_text = "unknown"
                                    total_mb = 0

                                downloaded_mb = downloaded / 1024 / 1024

                                text_lines = [
                                    "📥 Downloading file...",
                                    f"Downloaded: {downloaded_mb:.2f} MB"
                                    + (f" / {total_mb:.2f} MB" if total > 0 else ""),
                                    f"Speed: {speed_mb:.2f} MB/s",
                                    f"ETA: {eta_text}",
                                ]

                                try:
                                    await progress_message.edit_text("\n".join(text_lines))
                                except Exception:
                                    # Ignore edit errors (e.g., message too old)
                                    pass

                                last_update = now

                    # move tmp to final path
                    try:
                        os.replace(tmp_path, dest_path)
                    except Exception:
                        # fallback to rename
                        os.rename(tmp_path, dest_path)

            # success
            try:
                await progress_message.edit_text(
                    f"Download complete ✅\nSaved as: `{dest_path}`"
                )
            except Exception:
                pass
            return

        except aiohttp.ClientSSLError as e:
            err = f"SSL error: {e}"
            # On first SSL error attempt, retry with verification disabled
            if ssl_setting is not False:
                try:
                    await progress_message.edit_text(
                        "SSL error encountered. Retrying with SSL verification disabled..."
                    )
                except Exception:
                    pass
                ssl_setting = False
                await asyncio.sleep(1)
                continue

        except aiohttp.ClientConnectorError as e:
            err = f"Connection error: {e}"
        except asyncio.TimeoutError:
            err = "Timeout"
        except PermissionError as e:
            err = f"Permission denied: {e}"
        except Exception as e:
            err = str(e)

        # retry / final failure handling
        if attempt < retries:
            try:
                await progress_message.edit_text(f"Download failed ({err}), retrying {attempt}/{retries}...")
            except Exception:
                pass
            await asyncio.sleep(2 ** attempt)
        else:
            try:
                await progress_message.edit_text(f"Download fail ho gaya ❌: {err}")
            except Exception:
                pass
            raise RuntimeError(err)


def extract_user_pass(
    source_path: str, keyword: str, result_path: str
) -> int:
    """Scan file for lines with keyword, extract last two colon-separated fields as user:pass.

    This is a synchronous function intended to be run in a thread via
    `loop.run_in_executor(...)` to avoid blocking the event loop.
    """
    count = 0
    with open(source_path, "r", encoding="utf-8", errors="ignore") as src:
        with open(result_path, "w", encoding="utf-8") as out:
            for line in src:
                if keyword in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 3:
                        user_pass = ":".join(parts[-2:])
                        out.write(user_pass + "\n")
                        count += 1
    return count


def extract_combo_from_password_file(source_path: str, keyword: str | None, result_path: str) -> int:
    """
    Extract combos from password files with various formats.
    Supports:
    - Username: X | Password: Y | URL: Z → X:Y
    - Login: X | Password: Y → X:Y
    - url:login:pass → login:pass (extract last two colon parts)
    - Lines containing keyword only
    
    If keyword is None, extract all combos.
    """
    count = 0
    with open(source_path, "r", encoding="utf-8", errors="ignore") as src:
        with open(result_path, "w", encoding="utf-8") as out:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                
                # Filter by keyword if provided
                if keyword and keyword.lower() not in line.lower():
                    continue
                
                combo = None
                
                # Format 1: Username: X | Password: Y | URL: Z
                if " | " in line and ":" in line:
                    try:
                        parts = [p.strip() for p in line.split("|")]
                        username = None
                        password = None
                        for part in parts:
                            if part.lower().startswith("username:"):
                                username = part.split(":", 1)[1].strip()
                            elif part.lower().startswith("login:"):
                                username = part.split(":", 1)[1].strip()
                            elif part.lower().startswith("password:"):
                                password = part.split(":", 1)[1].strip()
                        if username and password:
                            combo = f"{username}:{password}"
                    except Exception:
                        pass
                
                # Format 2: url:login:pass (last two colon parts)
                if not combo and line.count(":") >= 2:
                    try:
                        parts = line.split(":")
                        # Take last two parts if it looks like url:login:pass
                        if len(parts) >= 3:
                            login = parts[-2].strip()
                            passwd = parts[-1].strip()
                            # Basic heuristic: avoid URL-like parts
                            if not login.startswith("http") and not passwd.startswith("http"):
                                combo = f"{login}:{passwd}"
                    except Exception:
                        pass
                
                if combo:
                    out.write(combo + "\n")
                    count += 1
    
    return count


def extract_combos_from_archive(archive_path: str, keyword: str | None, result_path: str, password: str | None) -> int:
    """
    Extract archive and search ALL .txt files recursively for combos.
    Supports multiple formats and removes duplicates.
    """
    tmp_dir = tempfile.mkdtemp(prefix="combo_extract_", dir=RESULTS_DIR)
    combo_count = 0
    try:
        # Extract archive
        first_bytes = b""
        try:
            with open(archive_path, "rb") as fh:
                first_bytes = fh.read(4096)
        except Exception:
            first_bytes = b""

        def looks_like_html(b: bytes) -> bool:
            b2 = b.lstrip().lower()
            return b2.startswith(b"<!doctype html") or b2.startswith(b"<html") or b"<html" in b2[:1024]

        if looks_like_html(first_bytes):
            raise RuntimeError("Downloaded content looks like HTML (not an archive).")

        is_zip = zipfile.is_zipfile(archive_path)
        is_rar = False
        if not is_zip and rarfile is not None:
            try:
                is_rar = rarfile.is_rarfile(archive_path)  # type: ignore
            except Exception:
                is_rar = False

        if is_zip:
            with zipfile.ZipFile(archive_path) as z:
                pwd = password.encode("utf-8") if password else None
                z.extractall(tmp_dir, pwd=pwd)
        elif is_rar:
            if rarfile is None:
                raise RuntimeError("RAR archive detected but 'rarfile' library not installed.")
            try:
                rf = rarfile.RarFile(archive_path)  # type: ignore
                if password:
                    rf.extractall(tmp_dir, pwd=password)
                else:
                    rf.extractall(tmp_dir)
            except Exception as e:
                # Try 7-Zip/WinRAR/UnRAR command-line as a fallback
                debug_log(f"[COMBO] rarfile extraction failed: {e}; attempting tool fallback...")
                if extract_with_available_tool(archive_path, tmp_dir, password):
                    debug_log(f"[COMBO] Tool extraction succeeded")
                else:
                    raise RuntimeError(
                        f"RAR extraction failed. Please install one of:\n"
                        "• 7-Zip (https://www.7-zip.org)\n"
                        "• WinRAR (https://www.winrar.com)\n"
                        "• UnRAR (https://www.rarlab.com/rar_add.htm)"
                    )

        # Search ALL .txt files recursively in archive
        seen_combos = set()
        with open(result_path, "w", encoding="utf-8") as out:
            for root, dirs, files in os.walk(tmp_dir):
                for fname in files:
                    if not fname.lower().endswith(".txt"):
                        continue
                    
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                if keyword and keyword.lower() not in line.lower():
                                    continue
                                
                                combo = None
                                # Format 1: Username: X | Password: Y
                                if " | " in line:
                                    parts = [p.strip() for p in line.split("|")]
                                    username = None
                                    password_field = None
                                    for part in parts:
                                        if part.lower().startswith(("username:", "login:")):
                                            username = part.split(":", 1)[1].strip()
                                        elif part.lower().startswith("password:"):
                                            password_field = part.split(":", 1)[1].strip()
                                    if username and password_field:
                                        combo = f"{username}:{password_field}"
                                
                                # Format 2: url:login:pass
                                if not combo and line.count(":") >= 2:
                                    parts = line.split(":")
                                    if len(parts) >= 3:
                                        login = parts[-2].strip()
                                        passwd = parts[-1].strip()
                                        if login and passwd and not login.startswith("http") and not passwd.startswith("http"):
                                            combo = f"{login}:{passwd}"
                                
                                if combo and combo not in seen_combos:
                                    seen_combos.add(combo)
                                    out.write(combo + "\n")
                                    combo_count += 1
                    except Exception:
                        pass

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # Remove duplicates from output file
    remove_duplicates_from_file(result_path)
    return combo_count


async def download_from_context_url(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    progress_message=None,
) -> str | None:
    """
    Reusable download helper.
    Uses URL stored in context.user_data["url"], same logic as before
    (download_file + curl fallback). Returns dest_path on success,
    or None on failure.
    If progress_message is provided, it will be reused instead of creating a new one.
    """
    url = context.user_data.get("url")
    if not url:
        await update.message.reply_text("Koi URL stored nahi mila. Pehle link bhejo.")
        return None

    file_name = os.path.basename(url.split("?")[0]) or "file.txt"
    dest_path = os.path.join(DOWNLOAD_DIR, file_name)

    if progress_message is None:
        progress_message = await update.message.reply_text("Starting download...")

    download_succeeded = False
    try:
        await download_file(url, dest_path, progress_message, context)
        download_succeeded = True
    except Exception as e:
        await safe_edit(progress_message, f"Download failed ❌: {e}")

    if not download_succeeded:
        context.user_data.clear()
        return None

    await safe_edit(progress_message, f"Download complete ✅\nSaved as: `{dest_path}`")
    context.user_data["file_path"] = dest_path
    return dest_path


def extract_urls(text: str) -> list[str]:
    # Simple URL regex (http/https)
    pattern = r"https?://[^\s]+"
    return re.findall(pattern, text)


async def process_single_cc(url: str, password: str | None, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Download one URL and run CC extraction; does not clear full user_data."""
    # Preserve existing url in context
    orig_url = context.user_data.get("url")
    orig_file_path = context.user_data.get("file_path")

    context.user_data["url"] = url
    dest_path = await download_from_context_url(update, context)
    if dest_path is None:
        # restore
        if orig_url is not None:
            context.user_data["url"] = orig_url
        else:
            context.user_data.pop("url", None)
        if orig_file_path is not None:
            context.user_data["file_path"] = orig_file_path
        else:
            context.user_data.pop("file_path", None)
        return

    # Check if stop was requested
    if context.user_data.get("stop_requested", False):
        await update.message.reply_text("Process stopped by user.")
        return

    await update.message.reply_text("Archive downloaded ✅\nMerging CreditCards...")

    base_name = os.path.splitext(os.path.basename(dest_path))[0]
    result_path = os.path.join(RESULTS_DIR, f"{base_name}_CreditCards_merged.txt")

    loop = asyncio.get_running_loop()
    try:
        merged_files = await loop.run_in_executor(
            None, merge_creditcards_from_archive, dest_path, result_path, password
        )
    except Exception as e:
        await update.message.reply_text(f"Error while extracting/merging CreditCards: {e}")
        # restore context
        if orig_url is not None:
            context.user_data["url"] = orig_url
        else:
            context.user_data.pop("url", None)
        if orig_file_path is not None:
            context.user_data["file_path"] = orig_file_path
        else:
            context.user_data.pop("file_path", None)
        return

    if merged_files == 0 or not os.path.exists(result_path):
        await update.message.reply_text(
            "Koi 'CreditCards' folder ke andar .txt file nahi mili. Archive structure alag ho sakta hai."
        )
        # restore
        if orig_url is not None:
            context.user_data["url"] = orig_url
        else:
            context.user_data.pop("url", None)
        if orig_file_path is not None:
            context.user_data["file_path"] = orig_file_path
        else:
            context.user_data.pop("file_path", None)
        return

    try:
        with open(result_path, "rb") as f:
            await update.message.reply_document(
                document=InputFile(f, filename=os.path.basename(result_path)),
                caption=f"Merged CreditCards ({merged_files} file(s))",
            )
    except FileNotFoundError:
        await update.message.reply_text("Internal error: merged result file missing. Please try again.")

    # restore original context values
    if orig_url is not None:
        context.user_data["url"] = orig_url
    else:
        context.user_data.pop("url", None)
    if orig_file_path is not None:
        context.user_data["file_path"] = orig_file_path
    else:
        context.user_data.pop("file_path", None)



def merge_creditcards_from_archive(
    archive_path: str,
    result_path: str,
    password: str | None,
) -> int:
    """
    Extract archive (zip/rar), search for folders named 'CreditCards',
    collect all .txt files inside them, and merge into result_path.

    Returns number of text files merged.
    """
    tmp_dir = tempfile.mkdtemp(prefix="cc_extract_", dir=RESULTS_DIR)
    merged_files = 0
    try:
        # Detect archive type by content, not by extension
        first_bytes = b""
        try:
            with open(archive_path, "rb") as fh:
                first_bytes = fh.read(4096)
        except Exception:
            first_bytes = b""

        def looks_like_html(b: bytes) -> bool:
            b2 = b.lstrip().lower()
            return b2.startswith(b"<!doctype html") or b2.startswith(b"<html") or b"<html" in b2[:1024]

        if looks_like_html(first_bytes):
            raise RuntimeError(
                "Downloaded content looks like HTML (not an archive). "
                "Most likely your link is not a direct-download link."
            )

        is_zip = zipfile.is_zipfile(archive_path)
        is_rar = False
        if not is_zip and rarfile is not None:
            try:
                is_rar = rarfile.is_rarfile(archive_path)  # type: ignore
            except Exception:
                is_rar = False

        is_7z = False
        if not is_zip and not is_rar:
            # 7z signature: 37 7A BC AF 27 1C
            is_7z = first_bytes.startswith(b"7z\xbc\xaf\x27\x1c")

        is_tar = False
        if not is_zip and not is_rar and not is_7z:
            try:
                is_tar = tarfile.is_tarfile(archive_path)
            except Exception:
                is_tar = False

        if is_zip:
            with zipfile.ZipFile(archive_path) as z:
                pwd = password.encode("utf-8") if password else None
                z.extractall(tmp_dir, pwd=pwd)
        elif is_rar:
            if rarfile is None:
                raise RuntimeError(
                    "RAR archive detected but 'rarfile' library is not installed."
                )
            try:
                rf = rarfile.RarFile(archive_path)  # type: ignore
                if password:
                    rf.extractall(tmp_dir, pwd=password)
                else:
                    rf.extractall(tmp_dir)
            except Exception as e:
                # Try 7-Zip/WinRAR/UnRAR command-line as a fallback
                debug_log(f"[CC] rarfile extraction failed: {e}; attempting tool fallback...")
                if extract_with_available_tool(archive_path, tmp_dir, password):
                    debug_log(f"[CC] Tool extraction succeeded")
                else:
                    raise RuntimeError(
                        f"RAR extraction failed. Please install one of:\n"
                        "• 7-Zip (https://www.7-zip.org)\n"
                        "• WinRAR (https://www.winrar.com)\n"
                        "• UnRAR (https://www.rarlab.com/rar_add.htm)"
                    )
        elif is_7z:
            if py7zr is None:
                raise RuntimeError(
                    "7Z archive detected but 'py7zr' is not installed. Install with: pip install py7zr"
                )
            try:
                if password:
                    with py7zr.SevenZipFile(archive_path, mode="r", password=password) as z:  # type: ignore
                        z.extractall(path=tmp_dir)
                else:
                    with py7zr.SevenZipFile(archive_path, mode="r") as z:  # type: ignore
                        z.extractall(path=tmp_dir)
            except Exception as e:
                raise RuntimeError(f"7Z extraction failed: {e}")
        elif is_tar:
            # Supports .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz, etc.
            try:
                with tarfile.open(archive_path, mode="r:*") as tf:
                    tf.extractall(path=tmp_dir)
            except Exception as e:
                raise RuntimeError(f"TAR extraction failed: {e}")
        else:
            raise RuntimeError(
                "Unsupported archive type: not ZIP/RAR/7Z/TAR (or file is corrupted/incomplete)."
            )

        seen_lines = set()
        with open(result_path, "w", encoding="utf-8", errors="ignore") as out:
            # Search for CreditCards folders
            for root, dirs, files in os.walk(tmp_dir):
                base = os.path.basename(root)
                if base.lower() == "creditcards":
                    for name in files:
                        if not name.lower().endswith(".txt"):
                            continue
                        src_path = os.path.join(root, name)
                        try:
                            with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                                for line in f:
                                    line_stripped = line.rstrip("\n")
                                    if line_stripped and line_stripped not in seen_lines:
                                        seen_lines.add(line_stripped)
                                        out.write(line_stripped + "\n")
                                        merged_files += 1
                        except Exception:
                            continue
            
            # Search for CreditCards.txt files
            for root, dirs, files in os.walk(tmp_dir):
                for name in files:
                    if name.lower() == "creditcards.txt":
                        src_path = os.path.join(root, name)
                        try:
                            with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                                for line in f:
                                    line_stripped = line.rstrip("\n")
                                    if line_stripped and line_stripped not in seen_lines:
                                        seen_lines.add(line_stripped)
                                        out.write(line_stripped + "\n")
                                        merged_files += 1
                        except Exception:
                            continue
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # Remove duplicates from output file
    remove_duplicates_from_file(result_path)
    return merged_files


def show_initial_options_keyboard():
    """Show the 5 initial option buttons after user sends a link."""
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("No Password 🔓", callback_data="init_no_pwd"), InlineKeyboardButton("With Password 🔐", callback_data="init_with_pwd")],
            [InlineKeyboardButton("Cookie Checker 🍪", callback_data="init_cookies"), InlineKeyboardButton("Get Combos (From .txt) 📋 ✨", callback_data="init_get_combo")],
            [InlineKeyboardButton("❌ Cancel ⛔", callback_data="init_cancel")],
        ]
    )
    return keyboard


def show_multiselect_keyboard(selected: set | None = None):
    """Show the 7 multi-select buttons. selected = set of button names that are ON."""
    if selected is None:
        selected = set()

    buttons_info = [
        ("unzip", "Unzip", "unzip_off"),
        ("get_cc", "Get CC 🚀", "get_cc_off"),
        ("get_combo", "Get Combos (FromZips) 📋", "get_combo_off"),
        ("cookies", "Get Cookies 🍪", "cookies_off"),
        ("ulp", "Get ULP (All)", "ulp_off"),
    ]

    rows = []
    button_row = []
    
    for i, (key, label, _) in enumerate(buttons_info):
        if key in selected:
            emoji = "✅"
            callback = f"toggle_{key}_off"
            button_text = f"{emoji} {label}"
        else:
            callback = f"toggle_{key}_on"
            button_text = label
        
        button = InlineKeyboardButton(button_text, callback_data=callback)
        
        if key == "ulp":
            # ULP goes on its own row
            rows.append([button])
        elif len(button_row) < 2:
            # Add to current row (max 2 buttons per row)
            button_row.append(button)
            if len(button_row) == 2:
                rows.append(button_row)
                button_row = []

    rows.append([InlineKeyboardButton("Back 🔙", callback_data="multiselect_back"), InlineKeyboardButton("Done ✓", callback_data="multiselect_done")])

    return InlineKeyboardMarkup(rows)


async def download_and_prepare_combo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Old behaviour: download file then ask for keyword and run extract_user_pass.
    """
    dest_path = await download_from_context_url(update, context)
    if dest_path is None:
        return

    context.user_data["awaiting"] = "keyword"

    await update.message.reply_text(
        "Download complete ✅\n"
        "Now send the keyword to search for in the downloaded file.\n"
        "Example: `username`"
    )


async def download_and_extract_cc_archive(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    password: str | None,
):
    """
    New CC logic:
    - download archive (zip/rar) from stored URL
    - extract it (with/without password)
    - find all 'CreditCards' folders and 'CreditCards.txt' files
    - merge all into one result file in timestamped folder
    """
    dest_path = await download_from_context_url(update, context)
    if dest_path is None:
        return

    await update.message.reply_text(
        "Archive downloaded ✅\n"
        "Ab CreditCards folders ke andar wale saare .txt merge kiye jaa rahe hain..."
    )

    base_name = os.path.splitext(os.path.basename(dest_path))[0]
    timestamped_folder = get_timestamped_folder(base_name)
    result_path = os.path.join(timestamped_folder, f"{base_name}_CreditCards_merged.txt")

    loop = asyncio.get_running_loop()
    try:
        merged_files = await loop.run_in_executor(
            None, merge_creditcards_from_archive, dest_path, result_path, password
        )
    except Exception as e:
        await update.message.reply_text(f"Error while extracting/merging CreditCards: {e}")
        context.user_data.clear()
        return

    if merged_files == 0 or not os.path.exists(result_path):
        await update.message.reply_text(
            "Koi 'CreditCards' folder ya 'CreditCards.txt' file nahi mili. "
            "Archive structure alag ho sakta hai."
        )
        context.user_data.clear()
        return

    await update.message.reply_text(
        f"Done ✅\nMerged {merged_files} line(s) from 'CreditCards' folders/files.\nSending file..."
    )

    try:
        with open(result_path, "rb") as f:
            await update.message.reply_document(
                document=InputFile(f, filename=os.path.basename(result_path)),
                caption=f"Merged CreditCards ({merged_files} line(s))",
            )
    except FileNotFoundError:
        await update.message.reply_text("Internal error: merged result file missing. Please try again.")
        context.user_data.clear()
        return

    context.user_data.clear()
    await update.message.reply_text(f"Ho gaya ✅\nResults saved to:\n`{timestamped_folder}`\n\nNaya link bhejo ya /start karo.")


# ==== HANDLERS ====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    # Cleaner, two-column layout; removed inline About button per request
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Get Combo 🔎", callback_data="get_combo"),
                InlineKeyboardButton("Get CC 💳", callback_data="get_cc"),
            ],
            [
                InlineKeyboardButton("With Password 🔐", callback_data="cc_with_pwd"),
                InlineKeyboardButton("Without Password 🔓", callback_data="cc_without_pwd"),
            ],
            [
                InlineKeyboardButton("Stop ⛔", callback_data="stop"),
                InlineKeyboardButton("Clear 🧹", callback_data="clear_cmd"),
            ],
            [
                InlineKeyboardButton("Help ❓", callback_data="help_cmd"),
                InlineKeyboardButton("Status 📊", callback_data="status_cmd"),
            ],
        ]
    )

    await update.message.reply_text(
        "✨ Welcome! ✨\nSend a direct download link (http:// or https://), then pick an action.\n\n"
        "Commands:\n"
        "/start ▶️ — Show this menu\n"
        "/stop ⛔ — Stop current interaction\n"
        "/help ❓ — Usage and commands\n"
        "/status 📊 — Session + storage info\n"
        "/clear 🧹 — Prepare to clear files\n"
        "/clear_confirm ✅ — Confirm clear\n"
        "/settings ⚙️ — Bot settings (placeholder)",
        reply_markup=keyboard,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Reset and start a new session\n"
        "/help - Show this help message\n"
        "/status - Show current session status and storage usage\n"
        "/clear - Prepare to clear all downloaded and result files (requires /clear_confirm)\n"
        "/clear_confirm - Confirm and delete all downloaded/result files\n\n"
        "Usage:\n"
        "1) Send a URL (http/https)\n"
        "2) Choose 1 (Get combo) or 2 (Get CC)\n"
        "   - Combo: after download, send keyword to extract user:pass\n"
        "   - CC: after download+extract, bot merges all 'CreditCards'/*.txt into one file."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_path = context.user_data.get("file_path")
    awaiting = context.user_data.get("awaiting")

    def dir_info(path):
        try:
            files = os.listdir(path)
        except Exception:
            return 0, 0
        total_size = 0
        for f in files:
            p = os.path.join(path, f)
            try:
                total_size += os.path.getsize(p)
            except Exception:
                pass
        return len(files), total_size

    d_count, d_size = dir_info(DOWNLOAD_DIR)
    r_count, r_size = dir_info(RESULTS_DIR)

    def fmt_size(b):
        return f"{b/1024/1024:.2f} MB"

    await update.message.reply_text(
        f"Session status:\nCurrent file: {file_path or 'None'}\nAwaiting: {awaiting or 'none'}\n\n"
        f"Downloads: {d_count} files ({fmt_size(d_size)})\nResults: {r_count} files ({fmt_size(r_size)})"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["clear_pending"] = True
    await update.message.reply_text(
        "Are you sure you want to delete all downloaded and result files?\n"
        "If yes, send /clear_confirm to confirm."
    )


async def clear_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Delete files in download and results folders
    deleted = 0
    for folder in (DOWNLOAD_DIR, RESULTS_DIR):
        try:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                try:
                    os.remove(fpath)
                    deleted += 1
                except Exception:
                    pass
        except Exception:
            pass

    context.user_data.clear()
    await update.message.reply_text(f"Cleared {deleted} files. Storage freed.")


async def clean_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Delete all files and folders in results and downloads directories."""
    deleted_count = 0
    error_list = []
    
    # Delete all files/folders in DOWNLOADS_DIR
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for item in os.listdir(DOWNLOAD_DIR):
                item_path = os.path.join(DOWNLOAD_DIR, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    deleted_count += 1
                except Exception as e:
                    error_list.append(f"downloads/{item}: {str(e)}")
    except Exception as e:
        error_list.append(f"Error accessing downloads folder: {str(e)}")
    
    # Delete all files/folders in RESULTS_DIR
    try:
        if os.path.exists(RESULTS_DIR):
            for item in os.listdir(RESULTS_DIR):
                item_path = os.path.join(RESULTS_DIR, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    deleted_count += 1
                except Exception as e:
                    error_list.append(f"results/{item}: {str(e)}")
    except Exception as e:
        error_list.append(f"Error accessing results folder: {str(e)}")
    
    # Send confirmation message
    response = f"✅ Cleanup completed!\n\n"
    response += f"🗑️ Deleted {deleted_count} file(s)/folder(s)\n"
    response += f"📁 Both downloads and results folders are now empty."
    
    if error_list:
        response += f"\n\n⚠️ Some errors occurred:\n"
        for error in error_list:
            response += f"  • {error}\n"
    
    await update.message.reply_text(response)


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop any ongoing process (download, extraction, etc.)."""
    context.user_data["stop_requested"] = True
    context.user_data.clear()
    await update.message.reply_text("⛔ Process stopped! Any ongoing download or processing has been cancelled.\n\nUse /start to begin a new session.")


async def about_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "About: This bot downloads files and extracts combos or CreditCards from archives.\n"
        "Improved UI with inline buttons."
    )


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Settings placeholder: No configurable settings yet. You can clear storage with /clear."
    )


async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data if query else None
    if query:
        await query.answer()

    if not data:
        return

    class DummyUpdate:
        def __init__(self, message):
            self.message = message

    # ===== INITIAL 5-OPTION BUTTONS =====
    if data == "init_no_pwd":
        context.user_data["awaiting"] = "multiselect"
        context.user_data["selected_actions"] = set()
        context.user_data["main_message"] = query.message
        keyboard = show_multiselect_keyboard(set())
        await query.message.edit_text("Choose actions (can select multiple):", reply_markup=keyboard)
        return

    if data == "init_with_pwd":
        context.user_data["awaiting"] = "multiselect"
        context.user_data["selected_actions"] = set()
        context.user_data["archive_password_mode"] = True
        context.user_data["main_message"] = query.message
        keyboard = show_multiselect_keyboard(set())
        await query.message.edit_text("Choose actions (can select multiple):", reply_markup=keyboard)
        return

    if data == "init_get_combo":
        context.user_data["awaiting"] = "keyword"
        await query.message.reply_text("Send the keyword to search for in the file:")
        return

    if data == "init_cookies":
        await query.message.reply_text("Cookies Checker placeholder (logic coming soon).")
        return

    if data == "init_cancel":
        context.user_data.clear()
        await query.message.reply_text("Cancelled. Send a new link to start.")
        return

    # ===== MULTISELECT TOGGLE BUTTONS =====
    if data.startswith("toggle_"):
        parts = data.split("_")
        action_name = "_".join(parts[1:-1])  # e.g., "get_combo"
        action_state = parts[-1]  # "on" or "off"

        selected = context.user_data.get("selected_actions", set())
        if action_state == "on":
            selected.add(action_name)
        else:
            selected.discard(action_name)
        
        context.user_data["selected_actions"] = selected
        keyboard = show_multiselect_keyboard(selected)
        await query.message.edit_text("Choose actions (can select multiple):", reply_markup=keyboard)
        return

    if data == "multiselect_back":
        context.user_data.pop("selected_actions", None)
        context.user_data.pop("archive_password_mode", None)
        context.user_data["awaiting"] = "initial_choice"
        keyboard = show_initial_options_keyboard()
        await query.message.reply_text("Back to initial menu:", reply_markup=keyboard)
        return

    if data == "multiselect_done":
        selected = context.user_data.get("selected_actions", set())
        if not selected:
            await query.message.reply_text("Please select at least one action.")
            return

        context.user_data["selected_actions"] = selected
        has_pwd_mode = context.user_data.get("archive_password_mode", False)
        
        if has_pwd_mode:
            context.user_data["awaiting"] = "archive_password"
            await query.message.edit_text("📝 Send the password for the archive (or /skip for no password):")
        else:
            # No password mode, proceed to process
            context.user_data["archive_password"] = None
            # Check if get_combo action is selected
            if "get_combo" in selected:
                context.user_data["awaiting"] = "combo_keyword"
                await query.message.edit_text("🔍 Send the keyword to search for combos:")
            else:
                context.user_data["awaiting"] = None
                dummy_update = DummyUpdate(query.message)
                await process_selected_actions(dummy_update, context)
        return

    # ===== LEGACY BUTTONS (for backward compat) =====
    if data == "clear_cmd":
        context.user_data["clear_pending"] = True
        await query.message.reply_text(
            "Are you sure you want to delete all downloaded and result files?\nIf yes, send /clear_confirm to confirm."
        )
        return

    # ===== CLEAN COMMAND CALLBACKS (removed - clean now deletes everything at once) =====

    if data == "stop":
        context.user_data.clear()
        await query.message.reply_text("Stopped. Use /start to start again.")
        return

    if data == "help_cmd":
        await help_cmd(DummyUpdate(query.message), context)
        return

    if data == "status_cmd":
        await status(DummyUpdate(query.message), context)
        return

    await query.message.reply_text("Unknown action.")


async def process_selected_actions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process all selected actions one by one. Extract archive ONCE, reuse for all actions.
    Supports stop flag to cancel processing.
    """
    selected = context.user_data.get("selected_actions", set())
    url = context.user_data.get("url")
    pwd = context.user_data.get("archive_password")
    keyword = context.user_data.get("combo_keyword")
    main_msg = context.user_data.get("main_message")
    
    if not url:
        await update.message.reply_text("No URL found. Please start over.")
        return
    
    # Clear stop flag at start of processing
    context.user_data["stop_requested"] = False
    
    # Download ONCE at the beginning
    if main_msg:
        await safe_edit(main_msg, "📥 Downloading file...")
    else:
        main_msg = await update.message.reply_text("📥 Downloading file...")
        context.user_data["main_message"] = main_msg
    
    dest_path = await download_from_context_url(update, context, progress_message=main_msg)
    if not dest_path or not os.path.exists(dest_path):
        await safe_edit(main_msg, "❌ Download failed or file not found.")
        context.user_data.clear()
        return
    
    await safe_edit(main_msg, f"✅ Downloaded successfully!\nExtracting archive...")
    
    # Extract ONCE here
    extracted_dir = None
    try:
        extracted_dir = tempfile.mkdtemp(prefix="archive_extract_", dir=RESULTS_DIR)
        loop = asyncio.get_running_loop()
        debug_log(f"[MAIN] Starting extraction to: {extracted_dir}")
        await loop.run_in_executor(None, extract_archive_once, dest_path, extracted_dir, pwd)
        debug_log(f"[MAIN] Extraction completed")
        await safe_edit(main_msg, f"✅ Archive extracted!\nProcessing actions...")
    except Exception as e:
        debug_log(f"[MAIN] Extraction failed: {e}")
        await safe_edit(main_msg, f"❌ Extraction failed: {str(e)}")
        if extracted_dir and os.path.exists(extracted_dir):
            try:
                shutil.rmtree(extracted_dir, ignore_errors=True)
            except Exception:
                pass
        context.user_data.clear()
        return
    
    # Now process each action with the SAME extracted folder
    for action in sorted(selected):
        # Check if stop was requested
        if context.user_data.get("stop_requested", False):
            await safe_edit(main_msg, "⛔ Processing stopped by user.")
            break
        
        try:
            if action == "unzip":
                await safe_edit(main_msg, "🔄 Processing: Unzip...\n✅ Done!")

            elif action == "get_cc":
                await safe_edit(main_msg, "🔄 Processing: Get CC...\n✅ Searching for CreditCards...")

                base_name = os.path.splitext(os.path.basename(dest_path))[0]
                timestamped_folder = get_timestamped_folder(base_name)
                result_path = os.path.join(timestamped_folder, f"{base_name}_CreditCards_merged.txt")
                cleaned_path = os.path.join(timestamped_folder, f"{base_name}_CreditCards_CLEAN.txt")

                loop = asyncio.get_running_loop()
                try:
                    debug_log(f"[CC] Searching in extracted folder")
                    merged_files = await loop.run_in_executor(
                        None, search_cc_in_extracted, extracted_dir, result_path
                    )
                    debug_log(f"[CC] Found {merged_files} creditcard lines")
                except Exception as e:
                    debug_log(f"[CC] Search error: {e}")
                    await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ Search error: {str(e)}")
                    continue

                if merged_files <= 0 or not os.path.exists(result_path):
                    debug_log(f"[CC] No creditcards found")
                    await safe_edit(main_msg, "🔄 Processing: Get CC...\n❌ No CreditCards found.")
                    continue

                # Clean CCs: validate format and Luhn algorithm
                await safe_edit(main_msg, f"🔄 Processing: Get CC...\n✅ Got {merged_files} CC, cleaning...")
                try:
                    debug_log(f"[CC] Cleaning creditcards with Luhn validation")
                    cleaned_count = await loop.run_in_executor(
                        None, clean_creditcards_file, result_path, cleaned_path
                    )
                    debug_log(f"[CC] Cleaned: {cleaned_count} valid cards out of {merged_files}")
                except Exception as e:
                    debug_log(f"[CC] Clean error: {e}")
                    cleaned_count = 0

                if cleaned_count <= 0:
                    debug_log(f"[CC] No valid creditcards after cleaning")
                    await safe_edit(main_msg, "🔄 Processing: Get CC...\n❌ No valid CreditCards after cleaning.")
                    continue

                await safe_edit(main_msg, f"🔄 Processing: Get CC...\n✅ Cleaned CC ({cleaned_count} valid), sending...")
                try:
                    with open(cleaned_path, "rb") as f:
                        try:
                            await update.message.reply_document(
                                document=InputFile(f, filename=os.path.basename(cleaned_path)),
                                caption=f"✅ CreditCards CLEAN ({cleaned_count} valid card(s) after Luhn validation)",
                            )
                        except Exception as e:
                            debug_log(f"[CC] send error: {e}")
                            await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ Send error: {str(e)}")
                except Exception as e:
                    debug_log(f"[CC] file error: {e}")
                    await safe_edit(main_msg, f"🔄 Processing: Get CC...\n❌ Error: {str(e)}")

            elif action == "get_combo":
                if not keyword:
                    await safe_edit(main_msg, "⏭️ Skipping Get Combo (no keyword provided).")
                    continue
                
                await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n✅ Searching...")

                base_name = os.path.splitext(os.path.basename(dest_path))[0]
                timestamped_folder = get_timestamped_folder(base_name)
                result_path = os.path.join(timestamped_folder, f"{base_name}_combo_{keyword}.txt")

                loop = asyncio.get_running_loop()
                try:
                    debug_log(f"[COMBO] Searching in extracted folder for keyword: {keyword}")
                    combo_count = await loop.run_in_executor(
                        None, search_combos_in_extracted, extracted_dir, keyword, result_path
                    )
                    debug_log(f"[COMBO] Found {combo_count} combos")
                except Exception as e:
                    debug_log(f"[COMBO] Search error: {e}")
                    await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ Search error: {str(e)}")
                    continue

                if combo_count > 0 and os.path.exists(result_path):
                    await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n✅ Got Combo ({combo_count} found), sending...")
                    try:
                        with open(result_path, "rb") as f:
                            try:
                                await update.message.reply_document(
                                    document=InputFile(f, filename=os.path.basename(result_path)),
                                    caption=f"✅ Combos for '{keyword}' ({combo_count} found)",
                                )
                            except Exception as e:
                                debug_log(f"[COMBO] send error: {e}")
                                await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ Send error: {str(e)}")
                    except Exception as e:
                        debug_log(f"[COMBO] file error: {e}")
                        await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ Error: {str(e)}")
                else:
                    debug_log(f"[COMBO] No combos found")
                    await safe_edit(main_msg, f"🔄 Processing: Get Combo (keyword: {keyword})...\n❌ No combos found.")

            elif action == "cookies":
                await safe_edit(main_msg, "⏭️ Get Cookies (placeholder - coming soon).")

            elif action == "ulp":
                await safe_edit(main_msg, "⏭️ Get ULP (placeholder - coming soon).")
        
        except Exception as e:
            await safe_edit(main_msg, f"❌ Action error: {str(e)}")
    
    # Clean up extracted folder
    if extracted_dir and os.path.exists(extracted_dir):
        try:
            shutil.rmtree(extracted_dir, ignore_errors=True)
            debug_log(f"[MAIN] Cleaned up extracted folder")
        except Exception:
            pass

    if main_msg:
        await safe_edit(main_msg, "✅ All selected actions completed.")
    context.user_data.clear()


async def extract_combos_from_archive_async(url: str, keyword: str, password: str | None, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Download archive and extract combos using the combo extraction logic."""
    orig_url = context.user_data.get("url")
    context.user_data["url"] = url
    dest_path = await download_from_context_url(update, context)
    if not dest_path:
        if orig_url:
            context.user_data["url"] = orig_url
        return 0

    base_name = os.path.splitext(os.path.basename(dest_path))[0]
    timestamped_folder = get_timestamped_folder(base_name)
    result_path = os.path.join(timestamped_folder, f"{base_name}_combo_{keyword}.txt")

    # Check if stop was requested
    if context.user_data.get("stop_requested", False):
        await update.message.reply_text("Process stopped by user.")
        return

    loop = asyncio.get_running_loop()
    try:
        combo_count = await loop.run_in_executor(
            None, extract_combos_from_archive, dest_path, keyword, result_path, password
        )
    except Exception as e:
        await update.message.reply_text(f"Error extracting combos: {e}")
        if orig_url:
            context.user_data["url"] = orig_url
        return 0

    if combo_count > 0 and os.path.exists(result_path):
        try:
            with open(result_path, "rb") as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=os.path.basename(result_path)),
                    caption=f"Combos for keyword '{keyword}' ({combo_count} found)",
                )
        except Exception:
            await update.message.reply_text(f"Failed to send combo file. Found {combo_count} combos.")
    else:
        await update.message.reply_text(f"No combos found for keyword: {keyword}")

    if orig_url:
        context.user_data["url"] = orig_url
    return combo_count


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    
    # Handle /skip command
    if text.lower() == "/skip":
        state = context.user_data.get("awaiting")
        if state == "archive_password":
            await update.message.delete()
            context.user_data["archive_password"] = None
            selected = context.user_data.get("selected_actions", set())
            main_msg = context.user_data.get("main_message")
            if "get_combo" in selected:
                context.user_data["awaiting"] = "combo_keyword"
                if main_msg:
                    await main_msg.edit_text("🔍 Send the keyword to search for combos:")
                else:
                    await update.message.reply_text("🔍 Send the keyword to search for combos:")
            else:
                context.user_data["awaiting"] = None
                await process_selected_actions(update, context)
            return
    
    state = context.user_data.get("awaiting")
    
    # Handle archive password input
    if state == "archive_password":
        await update.message.delete()
        context.user_data["archive_password"] = text
        selected = context.user_data.get("selected_actions", set())
        main_msg = context.user_data.get("main_message")
        if "get_combo" in selected:
            context.user_data["awaiting"] = "combo_keyword"
            if main_msg:
                await main_msg.edit_text("🔍 Send the keyword to search for combos:")
            else:
                await update.message.reply_text("🔍 Send the keyword to search for combos:")
        else:
            context.user_data["awaiting"] = None
            await process_selected_actions(update, context)
        return
    
    # Handle keyword for get_combo (unified handler)
    if state == "combo_keyword":
        await update.message.delete()
        context.user_data["combo_keyword"] = text
        context.user_data["awaiting"] = None
        await process_selected_actions(update, context)
        return

    # Detect URLs in message
    urls_in_text = extract_urls(text)
    if len(urls_in_text) > 1:
        # Multiple links detected
        context.user_data.clear()
        context.user_data["url_list"] = urls_in_text
        context.user_data["awaiting"] = "action_choice"
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Process All - Get CC (With Password) 🔐", callback_data="process_all_cc_with_pwd")],
                [InlineKeyboardButton("Process All - Get CC (Without Password) 🔓", callback_data="process_all_cc_without_pwd")],
                [InlineKeyboardButton("Process All - Get Combo 🔎", callback_data="process_all_combo")],
            ]
        )
        display = "\n".join([os.path.basename(u.split("?")[0]) or u for u in urls_in_text])
        await update.message.reply_text(f"Multiple links received and queued:\n{display}\n\nChoose how to process them:", reply_markup=keyboard)
        return

    # If the user sends a single URL directly, store it and ask what to do
    if len(urls_in_text) == 1 or text.startswith("http://") or text.startswith("https://"):
        url = urls_in_text[0] if urls_in_text else text
        context.user_data.clear()
        context.user_data["url"] = url
        context.user_data["awaiting"] = "initial_choice"
        keyboard = show_initial_options_keyboard()
        msg = await update.message.reply_text(
            "Link received ✅\nChoose an action:", reply_markup=keyboard
        )
        context.user_data["main_message"] = msg
        return

    # Default
    await update.message.reply_text("Kripya `/start` type karo shuru karne ke liye.")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("about", about_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("clear_confirm", clear_confirm))
    app.add_handler(CommandHandler("clean", clean_cmd))
    app.add_handler(CallbackQueryHandler(callback_query_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot running... Press Ctrl+C to stop.")
    # Use the synchronous run_polling() helper which manages the event loop internally.
    app.run_polling()


if __name__ == "__main__":
    if BOT_TOKEN == "PUT_YOUR_TELEGRAM_BOT_TOKEN_HERE":
        raise SystemExit("Please set your BOT_TOKEN in the script.")
    main()
