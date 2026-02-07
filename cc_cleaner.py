# cc_cleaner.py - Advanced Credit Card Extractor & Cleaner
# Outputs: 
#   - _MERGED.txt = All CCs found (before cleaning)
#   - _CLEAN.txt = Valid CCs only (after Luhn validation)

import re
from typing import List, Tuple


def luhn_checksum(card_num: str) -> bool:
    """Validate credit card using Luhn algorithm."""
    try:
        digits = [int(d) for d in card_num if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        
        for d in even_digits:
            checksum += sum([int(x) for x in str(d * 2)])
        
        return checksum % 10 == 0
    except Exception:
        return False


def normalize_month_year(month_str: str, year_str: str) -> Tuple[str, str] | None:
    """Convert month/year to normalized MM|YYYY format.
    
    Returns: (month_MM, year_YYYY) or None if invalid
    """
    try:
        month = int(month_str.strip())
        year_val = int(year_str.strip())
        
        # Validate month
        if not (1 <= month <= 12):
            return None
        
        # Normalize month to 2 digits
        month_str = f"{month:02d}"
        
        # Normalize year
        if year_val < 100:
            # 2-digit year: assume 20XX
            year_str = f"20{year_val:02d}"
        else:
            # 4-digit year
            year_str = f"{year_val:04d}"
        
        # Validate year is reasonable (2020-2040)
        year_int = int(year_str)
        if year_int < 2020 or year_int > 2040:
            return None
        
        return (month_str, year_str)
    except Exception:
        return None


def extract_creditcards(raw_text: str) -> Tuple[List[str], List[str]]:
    """
    Extract credit cards from messy text with multiple format support.
    
    Returns:
    - List of valid CCs: "number|MM|YYYY|CVV"
    - List of all extracted CCs (valid + invalid mixed): "number|MM|YYYY|CVV"
    """
    valid_ccs = set()
    all_ccs = set()  # For merged file
    seen_cc_numbers = set()
    
    # Pattern 1: NAME | CC | MM/YYYY | CVV
    # Matches: anything | 13-19 digits | digit/digit | 3-4 digits
    pattern1 = re.compile(
        r'(?:[^|]*\|\s*)?(\d{13,19})\s*\|\s*(\d{1,2})/(\d{2,4})\s*\|\s*(\d{3,4})',
        re.MULTILINE
    )
    
    # Pattern 2: CN: CC, DATE: MM/YYYY, CVV: XXX (block format)
    pattern2 = re.compile(
        r'CN:\s*(\d{13,19}).*?DATE:\s*(\d{1,2})/(\d{2,4}).*?(?:CVV|CVC):\s*(\d{3,4})',
        re.IGNORECASE | re.DOTALL
    )
    
    # Pattern 3: Card Holder: / Card Number: / Expiration: / CVC:
    pattern3 = re.compile(
        r'Card\s+(?:Holder|Number):\s*([^\n]*\n)?'
        r'Card\s+Number:\s*(\d{13,19}).*?'
        r'Expiration:\s*(\d{1,2})/(\d{2,4}).*?'
        r'(?:CVC|CVV):\s*(\d{3,4})',
        re.IGNORECASE | re.DOTALL
    )
    
    # Pattern 4: CardNumber: / NameOnCard: / ExpirationDate:
    pattern4 = re.compile(
        r'CardNumber:\s*(\d{13,19}).*?'
        r'(?:NameOnCard|Name\s+On\s+Card):[^\n]*\n.*?'
        r'(?:ExpirationDate|Exp):\s*(\d{1,2})/(\d{2,4}).*?'
        r'(?:CVC|CVV):\s*(\d{3,4})',
        re.IGNORECASE | re.DOTALL
    )
    
    # Pattern 5: Browser format - CardNumber: / ExpirationDate: / CVC:
    pattern5 = re.compile(
        r'CardNumber:\s*(\d{13,19}).*?'
        r'ExpirationDate:\s*(\d{1,2})/(\d{2,4}).*?'
        r'(?:CVC|CVV):\s*(\d{3,4})',
        re.IGNORECASE | re.DOTALL
    )
    
    # Pattern 6: Flexible: CC | MM/YYYY | CVV (with optional name before)
    pattern6 = re.compile(
        r'\|?\s*(\d{13,19})\s*\|\s*(\d{1,2})/(\d{2,4})\s*\|?\s*(\d{3,4})?',
        re.MULTILINE
    )
    
    patterns = [
        (pattern1, "Pipe-separated: NAME|CC|MM/YY|CVV", 4),
        (pattern2, "Block format: CN:/DATE:/CVV:", 4),
        (pattern3, "Card Holder block", 5),
        (pattern4, "CardNumber block", 4),
        (pattern5, "Browser credit card", 4),
    ]
    
    # Extract from all patterns
    for pattern, pattern_name, groups_count in patterns:
        for match in pattern.finditer(raw_text):
            groups = match.groups()
            
            if pattern == pattern3:
                cc, exp_m, exp_y, cvv = groups[1], groups[2], groups[3], groups[4]
            elif pattern == pattern1 or pattern == pattern6:
                if len(groups) < 3:
                    continue
                cc, exp_m, exp_y, cvv = groups[0], groups[1], groups[2], groups[3] if len(groups) > 3 else None
            else:
                if len(groups) < 3:
                    continue
                cc, exp_m, exp_y, cvv = groups[0], groups[1], groups[2], groups[3]
            
            if not cc or not exp_m or not exp_y or not cvv:
                continue
            
            cc = cc.strip()
            cvv = cvv.strip()
            
            # Check length
            if not (13 <= len(cc) <= 19):
                continue
            
            # Check not all zeros
            if set(cc) == {"0"}:
                continue
            
            # Normalize month/year
            normalized = normalize_month_year(exp_m, exp_y)
            if not normalized:
                continue
            
            month_str, year_str = normalized
            result = f"{cc}|{month_str}|{year_str}|{cvv}"
            
            # Add to merged (all found)
            all_ccs.add(result)
            
            # Add to valid only if Luhn passes
            if luhn_checksum(cc) and cc not in seen_cc_numbers:
                seen_cc_numbers.add(cc)
                valid_ccs.add(result)
    
    return sorted(list(valid_ccs)), sorted(list(all_ccs))


def main():
    # Input file
    in_path = r"C:\Users\anshu\Downloads\Telegram Desktop\AgADB523633_CreditCards_merged.txt"
    
    # Output files
    out_merged = in_path.replace(".txt", "_MERGED.txt")
    out_clean = in_path.replace(".txt", "_CLEAN.txt")
    
    try:
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"❌ Input file not found: {in_path}")
        return
    
    print(f"📥 Processing {len(raw)} bytes...")
    
    # Extract and clean
    cleaned_ccs, all_ccs = extract_creditcards(raw)
    
    # Write merged (all found CCs - before cleaning)
    with open(out_merged, "w", encoding="utf-8") as f:
        for cc in all_ccs:
            f.write(cc + "\n")
    
    # Write cleaned (only valid CCs - after Luhn validation)
    with open(out_clean, "w", encoding="utf-8") as f:
        for cc in cleaned_ccs:
            f.write(cc + "\n")
    
    # Print summary
    print(f"\n✅ RESULTS:")
    print(f"  📋 All Found: {len(all_ccs)} CCs")
    print(f"  ✓ Valid (Luhn): {len(cleaned_ccs)} CCs")
    print(f"  ✗ Invalid: {len(all_ccs) - len(cleaned_ccs)} CCs")
    print(f"\n📄 Files Created:")
    print(f"  1. {out_merged} → All CCs found (before cleaning)")
    print(f"  2. {out_clean} → Valid CCs only (after Luhn check)")
    print(f"\n💡 Send both files to bot to process!")


if __name__ == "__main__":
    main()
