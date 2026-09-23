#!/usr/bin/env python3
"""
Patch tu dong cho pickleball.py:
  1. Chi ve khung cho "nguoi" (qua chip imx500), bo qua moi vat the khac
     (xe, ghe, cho, TV...) - giam so lan goi cv2.rectangle/putText moi frame.
  2. Xoa dong legend "Vat the (imx500)" trong trang HTML vi khong con dung nua.

Face detection (YuNet) va pickleball detection (YOLO) khong doi - 2 model do
von da chi nhan dien dung 1 thu cua no roi.

Cach dung:
  1. Dat file nay CUNG THU MUC voi pickleball.py tren Raspberry Pi.
  2. Chay:  python3 optimize_pickleball.py
  3. Kiem tra log in ra, roi thu:  python3 pickleball.py

An toan: script CHI ghi de pickleball.py neu tim thay dung vi tri can sua.
Neu khong tim thay (vi du code ban dang co da khac ban goc), no se BAO LOI
va KHONG doi gi ca - khong lam hong file cua ban. Chay lai nhieu lan cung
khong sao, no tu nhan biet neu da patch roi thi bo qua.
"""

PATH = "pickleball.py"

with open(PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

changed = False

# --- Patch 1: chi ve khung cho "nguoi", bo qua vat the khac ---
anchor1 = 'name = labels[cls_idx] if labels and cls_idx < len(labels) else f"class{cls_idx}"'
idx1 = next((i for i, l in enumerate(lines) if anchor1 in l), None)

if idx1 is None:
    print("[BO QUA - Patch 1] Khong tim thay dong tinh bien 'name' trong vong lap")
    print("  object detection cua imx500. Co the code ban dang co da khac ban goc")
    print("  ma minh doc duoc. Gui lai doan code quanh 'PERSON_LABEL_NAME' de minh")
    print("  chinh lai patch cho dung.")
else:
    already_patched = idx1 + 1 < len(lines) and "!= PERSON_LABEL_NAME" in lines[idx1 + 1]
    if already_patched:
        print("[BO QUA - Patch 1] Hinh nhu da patch roi (da co dieu kien loc nguoi).")
    else:
        raw = lines[idx1]
        indent = raw[: len(raw) - len(raw.lstrip())]
        new_lines = [
            f"{indent}if name.strip().lower() != PERSON_LABEL_NAME:\n",
            f"{indent}    continue  # chi quan tam \"nguoi\", bo qua moi vat the khac\n",
        ]
        lines[idx1 + 1 : idx1 + 1] = new_lines
        changed = True
        print("[OK - Patch 1] Da them dieu kien: chi ve khung cho 'nguoi', bo qua vat the khac.")

# --- Patch 2: xoa dong legend "Vat the (imx500)" khong con dung nua ---
idx2 = next((i for i, l in enumerate(lines) if "Vat the (imx500)" in l), None)
if idx2 is None:
    print("[BO QUA - Patch 2] Khong tim thay dong legend 'Vat the (imx500)'")
    print("  (co the da xoa roi, hoac noi dung file da khac ban goc).")
else:
    del lines[idx2]
    changed = True
    print("[OK - Patch 2] Da xoa dong legend 'Vat the (imx500)' khong con dung nua.")

if changed:
    with open(PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("\nDa luu pickleball.py. Chay thu: python3 pickleball.py")
else:
    print("\nKhong co thay doi nao duoc ghi vao file.")
