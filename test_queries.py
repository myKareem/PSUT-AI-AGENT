# -*- coding: utf-8 -*-
"""
Test harness for the PSUT Voice Agent chatbot.
Runs test queries covering every KB area and pipeline path,
then prints results in a structured format.
"""
import os, sys
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from chatbot import generate_response, session_history

# ─── Test Queries ────────────────────────────────────────────
# Each tuple: (category, query, expected_ground_truth_keywords, correct_answer_note)
TEST_QUERIES = [
    # ══════════════════════════════════════════════════════════
    #  1. GENERAL FAQ / CAG  (general_faq.md — in-memory)
    # ══════════════════════════════════════════════════════════
    ("CAG: المرشد الأكاديمي",
     "كيف أعرف مرشدي الأكاديمي؟",
     ["بوابة الطالب", "بيانات الطالب", "ملف الطالب", "أكاديمية"],
     "KB Answer: عبر بوابة الطالب → بيانات الطالب → ملف الطالب → بيانات أكاديمية"),

    ("CAG: الهوية الجامعية",
     "متى بستلم الهوية الجامعية؟",
     ["بداية التدريس", "عمادة شؤون الطلبة", "الوصل المالي"],
     "KB Answer: عند بداية التدريس، مراجعة عمادة شؤون الطلبة"),

    ("CAG: نسيت كلمة السر",
     "نسيت كلمة السر لبوابة الطالب",
     ["نسيت كلمة السر", "الرقم الجامعي", "البريد الالكتروني", "ارسال"],
     "KB Answer: اضغط على 'هل نسيت كلمة السر' → أدخل الرقم الجامعي → سيتم ارسال رسالة"),

    ("CAG: كيف أقدم للجامعة",
     "كيف أقدم طلب التحاق للجامعة؟",
     ["بوابة الطالب", "التقديم", "online"],
     "KB Answer: التقديم online من خلال بوابة الطالب الإلكترونية"),

    ("CAG: هل امتحان القبول لجميع المتقدمين",
     "هل امتحان القبول الزامي؟",
     ["جميع المتقدمين", "معدلاتهم"],
     "KB Answer: نعم لجميع المتقدمين مهما كانت معدلاتهم"),

    ("CAG: إذا لم أقبل في الدفعة",
     "إذا لم أقبل في الدفعة اللي قدمت لها شو بصير؟",
     ["قائمة الانتظار", "الدفعة", "منافسة"],
     "KB Answer: اسمك يبقى في قائمة الانتظار وتدخل المنافسة مع الدفعة التالية"),

    ("CAG: براءة ذمة",
     "كيف أعمل براءة ذمة بعد التخرج؟",
     ["خدمات إلكترونية", "براءة ذمة", "المكتبة", "المالية", "شؤون الطلبة"],
     "KB Answer: بوابة الطالب → الخدمات الإلكترونية → طلب براءة ذمة → مراجعة عدة جهات"),

    # ══════════════════════════════════════════════════════════
    #  2. MAJOR OVERVIEW / RAG  (major_overview collection)
    # ══════════════════════════════════════════════════════════
    ("RAG: سعر ساعة الأمن السيبراني",
     "كم سعر الساعة في تخصص الأمن السيبراني؟",
     ["130", "مئة وثلاثين"],
     "KB Answer: 130 دينار / مئة وثلاثين دينار"),

    ("RAG: كلية هندسة البرمجيات",
     "هندسة البرمجيات تحت أي كلية؟",
     ["كلية الملك الحسين لعلوم الحوسبة"],
     "KB Answer: كلية الملك الحسين لعلوم الحوسبة"),

    ("RAG: ساعات تخرج إدارة الأعمال",
     "كم ساعة بدي أخلص عشان أتخرج من إدارة الأعمال؟",
     ["132", "مئة واثنين وثلاثين", "81"],
     "KB Answer: 132 ساعة معتمدة total, 81 ساعة تخصص"),

    ("RAG: تخصص علم البيانات والذكاء الاصطناعي",
     "شو تخصص علم البيانات والذكاء الاصطناعي؟",
     ["الأول من نوعه", "الأردن", "الحوسبة", "132", "بيانات", "ذكاء", "الحسين", "مئة"],
     "KB Answer: الأول من نوعه في الأردن، كلية الملك الحسين، 132 ساعة، 130 دينار"),

    ("RAG: ماجستير أمن سيبراني شروط",
     "شو شروط القبول في ماجستير الأمن السيبراني؟",
     ["بكالوريوس", "جيد"],
     "KB Answer: بكالوريوس في تخصص ذي صلة بتقدير جيد"),

    ("RAG: سعر ساعة المحاسبة",
     "كم سعر ساعة المحاسبة؟",
     ["120", "مئة وعشرين"],
     "KB Answer: 120 دينار"),

    # ══════════════════════════════════════════════════════════
    #  3. STAFF DIRECTORY  (graph search)
    # ══════════════════════════════════════════════════════════
    ("STAFF: رئيس الجامعة",
     "مين رئيس الجامعة؟",
     ["وجدان", "أبو الهيجاء", "الهيجاء"],
     "KB Answer: أ.د. وجدان أبو الهيجاء, elhaija@psut.edu.jo"),

    ("STAFF: عميد كلية الحوسبة",
     "مين عميد كلية الحوسبة؟",
     ["أنس", "أبو طالب", "ابو طالب"],
     "KB Answer: أ.د أنس ابو طالب, a.abutaleb@psut.edu.jo"),

    ("STAFF: إيميل قسم المحاسبة",
     "شو إيميل قسم المحاسبة؟",
     ["srouji", "عنان", "سروجي"],
     "KB Answer: a.srouji@psut.edu.jo (د. عنان سروجي, رئيس قسم المحاسبة)"),

    ("STAFF: عميد كلية الأعمال",
     "مين عميد كلية الأعمال؟",
     ["عدي", "الطويسي", "طويسي"],
     "KB Answer: د. عدي الطويسي, a.tweissi@psut.edu.jo"),

    # ══════════════════════════════════════════════════════════
    #  5. OUT-OF-SCOPE / Grounding Test
    # ══════════════════════════════════════════════════════════
    ("GROUNDING: سؤال خارج النطاق",
     "شو عاصمة فرنسا؟",
     ["آسف", "ما عندي"],
     "Expected: rejection — this is not about PSUT"),

    # ══════════════════════════════════════════════════════════
    #  6. TTS QUALITY CHECKS
    # ══════════════════════════════════════════════════════════
    ("TTS-CHECK: طالب مستجد بوابة الطالب",
     "كيف أدخل على بوابة الطالب الإلكترونية كطالب مستجد؟",
     ["الرقم الجامعي", "كلمة السر"],
     "KB Answer: أدخل الرقم الجامعي وكلمة السر ثم وافق على الشروط"),

    # ══════════════════════════════════════════════════════════
    #  7. ROBUSTNESS STRESS TESTS
    #  These reproduce exact hallucination scenarios from live sessions
    # ══════════════════════════════════════════════════════════

    # Live session hallucination: router sent to student_guide (skipped), answer is in CAG
    ("ROBUST: امتحان القبول إلزامي",
     "هل امتحان القبول الزامي؟",
     ["نعم", "جميع", "المتقدمين"],
     "KB Answer (general_faq): نعم لجميع المتقدمين مهما كانت معدلاتهم"),

    # Live session hallucination: vague follow-up with no context
    ("ROBUST: سعر ساعة الأمن السيبراني بالعامية",
     "كم سعر ساعه الامن السيبراني؟",
     ["مئة وثلاثين", "130", "الحوسبة", "الحسين"],
     "KB Answer (major_overview): 130 دينار، كلية الملك الحسين لعلوم الحوسبة"),

    # Live session hallucination: عميد routed to major_overview instead of staff
    ("ROBUST: عميد كلية الأعمال بالعامية",
     "طيب مين عميد كليه الاعمال؟",
     ["عدي", "الطويسي", "طويسي"],
     "KB Answer (staff_directory): د. عدي الطويسي"),

    # Live session hallucination: STT spelled رءيس not رئيس
    ("ROBUST: رئيس الجامعة بتهجئة مختلفة",
     "مين رءيس الجامعه؟",
     ["وجدان", "الهيجاء", "هيجاء"],
     "KB Answer (staff_directory): أ.د. وجدان أبو الهيجاء"),

    # Test: out-of-scope question should be refused (not hallucinated)
    ("ROBUST: سؤال خارج النطاق عن أسعار",
     "كم سعر البنزين اليوم؟",
     ["آسف", "ما عندي"],
     "Expected: rejection — not about PSUT"),

    # Test: ساعات تخرج should include the number from context
    ("ROBUST: ساعات تخرج هندسة الحاسوب",
     "كم ساعة لازم اقطع عشان اتخرج من هندسة الحاسوب؟",
     ["مئة وستين", "160", "مئة واثنين وثلاثين", "132", "الهندسة", "الحوسبة"],
     "KB Answer (major_overview): 160 ساعة معتمدة"),

    # Test: رئيس قسم specific department
    ("ROBUST: رئيس قسم هندسة البرمجيات",
     "مين رئيس قسم هندسة البرمجيات؟",
     ["عبدالله", "قصف", "Abdullah", "Qussef", "qusef"],
     "KB Answer (staff_directory): أ.د. عبدالله قصف"),

    # Test: براءة ذمة — should hit CAG
    ("ROBUST: براءة ذمة بالعامية",
     "كيف اعمل براءه ذمه؟",
     ["تخرج", "شهاد"],
     "KB Answer (general_faq): بعد التخرج تعمل براءة ذمة وتستلم شهاداتك"),

    # Test: إيميل specific doctor
    ("ROBUST: إيميل دكتور محمد العزة",
     "شو ايميل دكتور محمد العزه؟",
     ["azzeh", "m.azzeh"],
     "KB Answer (staff_directory): m.azzeh@psut.edu.jo"),

    # Test: كلية الهندسة عميد
    ("ROBUST: عميد كلية الهندسة",
     "مين عميد كلية الهندسة؟",
     ["اسامة", "أبو شرخ", "ابو شرخ", "اسامه"],
     "KB Answer (staff_directory): د. اسامة أبو شرخ"),

    # Test: الخطوة بعد تقديم الطلب — CAG
    ("ROBUST: الخطوة بعد تقديم الطلب",
     "شو الخطوة بعد ما قدمت الطلب؟",
     ["انتظر", "اتصال", "امتحان القبول", "اختبار", "رسالة", "موعد"],
     "KB Answer (general_faq): انتظر اتصال هاتفي أو email يبلغك عن موعد امتحان القبول"),
]


def run_all_tests():
    """Run each query through generate_response and collect results."""
    results = []
    total = len(TEST_QUERIES)

    for idx, (category, query, expected_kw, note) in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"  TEST {idx}/{total}: {category}")
        print(f"  QUERY: {query}")
        print(f"  EXPECTED: {note}")
        print(f"{'='*70}")

        # Clear history between tests so they're independent
        session_history.clear()

        # Collect full response from generator
        full_response = ""
        try:
            for raw, cleaned in generate_response(query):
                full_response += raw + " "
        except Exception as e:
            full_response = f"[ERROR] {e}"

        full_response = full_response.strip()

        # Quality checks
        has_url      = "http" in full_response or "www." in full_response
        has_dots     = ". . ." in full_response or "...." in full_response
        has_markdown = any(m in full_response for m in ["###", "**", "---", "* "])
        has_emoji    = any(ord(c) > 0x1F000 for c in full_response)

        # Keyword match check
        matched_kw   = [kw for kw in expected_kw if kw in full_response]
        kw_hit       = len(matched_kw) > 0

        # TTS quality flag
        tts_clean = not has_url and not has_dots and not has_markdown and not has_emoji

        results.append({
            "idx": idx,
            "category": category,
            "query": query,
            "response": full_response,
            "note": note,
            "kw_hit": kw_hit,
            "matched_kw": matched_kw,
            "expected_kw": expected_kw,
            "tts_clean": tts_clean,
            "has_url": has_url,
            "has_dots": has_dots,
            "has_markdown": has_markdown,
        })

        # Live output
        print(f"\n  RESPONSE: {full_response[:400]}{'...' if len(full_response) > 400 else ''}")
        print(f"  KW_MATCH: {'PASS' if kw_hit else 'FAIL'}  matched={matched_kw}")
        print(f"  TTS_CLEAN: {'PASS' if tts_clean else 'FAIL'}  url={has_url} dots={has_dots} md={has_markdown}")

    return results


def print_summary(results):
    """Print a final summary table."""
    print("\n\n" + "=" * 100)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'#':<4} {'Category':<50} {'KW':>4} {'TTS':>5} {'Issues'}")
    print("-" * 100)

    kw_pass = 0
    tts_pass = 0

    for r in results:
        issues = []
        if r["has_url"]:      issues.append("URL")
        if r["has_dots"]:     issues.append("DOTS")
        if r["has_markdown"]: issues.append("MARKDOWN")
        if not r["kw_hit"]:   issues.append("NO_KW_MATCH")

        kw_sym  = "PASS" if r["kw_hit"] else "FAIL"
        tts_sym = "PASS" if r["tts_clean"] else "FAIL"
        issue_str = ", ".join(issues) if issues else "—"

        print(f"{r['idx']:<4} {r['category']:<50} {kw_sym:>4} {tts_sym:>5} {issue_str}")

        if r["kw_hit"]:   kw_pass += 1
        if r["tts_clean"]: tts_pass += 1

    total = len(results)
    print("-" * 100)
    print(f"  KW Match: {kw_pass}/{total} ({kw_pass/total*100:.0f}%)")
    print(f"  TTS Clean: {tts_pass}/{total} ({tts_pass/total*100:.0f}%)")
    print("=" * 100)

    # Write full details to file
    report_path = os.path.join(os.path.dirname(__file__), "test_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PSUT Voice Agent — Test Results (Run 2 — After Fixes)\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"Test {r['idx']}: {r['category']}\n")
            f.write(f"  Query:    {r['query']}\n")
            f.write(f"  Expected: {r['note']}\n")
            f.write(f"  Response: {r['response']}\n")
            f.write(f"  KW Match: {'PASS' if r['kw_hit'] else 'FAIL'} — matched: {r['matched_kw']} / expected: {r['expected_kw']}\n")
            f.write(f"  TTS Clean: {'PASS' if r['tts_clean'] else 'FAIL'} — url={r['has_url']}, dots={r['has_dots']}, md={r['has_markdown']}\n")
            f.write("-" * 80 + "\n")
        f.write(f"\nSummary: KW={kw_pass}/{total}, TTS={tts_pass}/{total}\n")
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    results = run_all_tests()
    print_summary(results)
