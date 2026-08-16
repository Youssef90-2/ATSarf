
import unicodedata


# ---------------------------------------------------------------------------
# 1. علامات التشكيل (Diacritics) — نفس اللي كان النظام القديم يشيلها
#    (من letters.h: fatha, damma, kasra, sukun, التنوين, shadda, madda...)
# ---------------------------------------------------------------------------
DIACRITICS = {
    "\u064B",  # fathatayn   ً
    "\u064C",  # dammatayn   ٌ
    "\u064D",  # kasratayn   ٍ
    "\u064E",  # fatha       َ
    "\u064F",  # damma       ُ
    "\u0650",  # kasra       ِ
    "\u0651",  # shadda      ّ
    "\u0652",  # sukun       ْ
    "\u0653",  # madda       ٓ
    "\u0654",  # hamza above ٔ
    "\u0655",  # hamza below ٕ
    "\u0670",  # superscript alef (alef khanjariyya) ٰ
}

TATWEEL = "\u0640"

ZERO_WIDTH_JOINER = "\u200D"


# ---------------------------------------------------------------------------
# 2. توحيد الحروف (Letter unification)
# ---------------------------------------------------------------------------
LETTER_MAP = {
    # كل أشكال الألف والهمزة → ا  (0x0627)
    "\u0623": "\u0627",  # أ  alef hamza above
    "\u0625": "\u0627",  # إ  alef hamza below
    "\u0622": "\u0627",  # آ  alef madda above
    "\u0671": "\u0627",  # ٱ  alef wasla
    "\u0649": "\u0627",  # ى  alef maksoura → ا  (النظام القديم كان يعتبرها شكل من الألف)

    # الهمزات المتوسطة/المنفصلة → ء  (توحيد بسيط، بيقلّل التنوّع)
    "\u0624": "\u0621",  # ؤ  waw hamza  → ء
    "\u0626": "\u0621",  # ئ  ya hamza   → ء

    # التاء المربوطة → هاء  (شائع في التطبيع العربي، والنظام القديم لمّح إلها)
    "\u0629": "\u0647",  # ة → ه
}


# ---------------------------------------------------------------------------
# 3. الرموز اللي بدنا نحوّلها لفراغ (من داتاك الحقيقية)
#    ملاحظة مهمة: منحوّلها لفراغ (مش نحذفها) تا نحافظ على طول النص واحد-لواحد
#    قدر الإمكان — بس حتى لو حذفنا، index_map بيمسك الموضوع.
# ---------------------------------------------------------------------------
SYMBOLS_TO_SPACE = set('"[](){}«»<>*')


def is_diacritic(ch: str) -> bool:
    """هل الحرف علامة تشكيل؟ (نفس فكرة isDiacritic في النظام القديم)."""
    return ch in DIACRITICS


def normalize(text: str):
   
    clean_chars = []
    index_map = []
    prev_was_space = False

    for original_pos, ch in enumerate(text):

        # (1) تشكيل / تطويل / وصل غير مرئي → بينشال تماماً
        if is_diacritic(ch) or ch == TATWEEL or ch == ZERO_WIDTH_JOINER:
            continue

        # (2) توحيد الحروف
        ch = LETTER_MAP.get(ch, ch)

        # (3) الرموز الدخيلة → فراغ
        if ch in SYMBOLS_TO_SPACE:
            ch = " "

        # (4) دمج الفراغات المتتالية بفراغ واحد
        if ch.isspace():
            if prev_was_space:
                continue          # تخطّي الفراغ الزائد
            ch = " "              # توحيد كل أنواع الفراغ (tab, newline...) لمسافة
            prev_was_space = True
        else:
            prev_was_space = False

        clean_chars.append(ch)
        index_map.append(original_pos)

    return "".join(clean_chars), index_map


def normalize_word(word: str) -> str:
    """
    نسخة مبسّطة لكلمة وحدة (بدون index_map).
    مفيدة لمقارنة كلمات المعجم (lexicons) والأسماء بسرعة.
    """
    clean, _ = normalize(word)
    return clean.strip()


# ---------------------------------------------------------------------------
# تجربة سريعة: شغّل الملف لحالو تشوف الفكرة شغّالة
#     python normalization.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "حَدَّثَنَا",                       # مع تشكيل كامل
        "أحمد",                             # ألف همزة
        "إسحاق",                            # ألف همزة تحت
        "آمنة",                             # ألف مدّة
        "معاوية",                           # تاء عادية
        "فاطمة",                            # تاء مربوطة → هاء
        'قال: "يا جابر"',                   # علامات اقتباس دخيلة
        "المتوفى سنة 328 ه‍",               # رمز التاريخ ه‍ مع وصل غير مرئي
        "[ تعالى ]",                        # أقواس مربعة
    ]

    print("الأصل".rjust(25), "→", "المُنظّف")
    print("-" * 55)
    for s in samples:
        clean, idx = normalize(s)
        print(s.rjust(25), "→", repr(clean))

    # إثبات إنو index_map صحيح: نرجّع أول كلمة منظّفة لموقعها الأصلي
    print("\n--- إثبات تتبّع المواقع ---")
    original = 'قال: "يا جابر"'
    clean, idx = normalize(original)
    print("النص الأصلي   :", repr(original))
    print("النص المنظّف  :", repr(clean))
    # ناخد كلمة "يا" بالنص المنظّف ونرجّعها للأصلي
    start_clean = clean.index("يا")
    start_original = idx[start_clean]
    print(f"'يا' بالمنظّف موقعها {start_clean}، وبالأصلي موقعها {start_original}")
    print("الحرف بالنص الأصلي عند هالموقع:", repr(original[start_original]))

    # ملاحظة: منقرا الملفات دايماً بـ text-mode utf-8 (مش newline='')، لأنه
    # Python بيوحّد CRLF -> LF لحالو، وهيك المواقع بتطابق ملفات الـ gold
    # يلي انعملت على نسخة LF بالنظام القديم.
    result = normalize_word("حَدَّثَنَا")
    print("normalize_word('حَدَّثَنَا') ==", repr(result),
          "->", result == "حدثنا")