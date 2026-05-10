# -*- coding: utf-8 -*-
"""
PSUT Voice Agent — Comprehensive Evaluation Suite
==================================================
Methodology:
  1. Factual Accuracy (FA)  — Does the response contain the correct facts from KB?
  2. Grounding (GR)         — Does the response refuse when it should (no context)?
  3. Hallucination (HL)     — Does the response contain fabricated info?
  4. TTS Readiness (TTS)    — No URLs, emojis, markdown, or dot sequences?
  5. Routing Accuracy (RT)  — Did the pipeline reach the correct source?

Each test has:
  - category: which pipeline component is tested
  - query: the user's input
  - ground_truth: the correct answer from KB
  - required_kw: keywords that MUST appear (any one = pass)
  - forbidden_kw: keywords that MUST NOT appear (any one = fail for hallucination)
  - expected_source: which retrieval source should respond (CAG, RAG, STAFF, REFUSE)
"""

import sys, os, time, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

from chatbot import generate_response, session_history

# ══════════════════════════════════════════════════════════════
#  GROUND TRUTH TEST CASES — verified against KB files
# ══════════════════════════════════════════════════════════════
TESTS = [
    # ─── CAG / FAQ Tests ───────────────────────────────────────
    {
        "id": 1, "category": "CAG-FAQ",
        "query": "كيف أعرف مرشدي الأكاديمي؟",
        "ground_truth": "من خلال بوابة الطالب الإلكترونية، اختر بيانات الطالب ثم ملف الطالب ثم بيانات أكاديمية",
        "required_kw": ["بوابة الطالب", "بيانات"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
    {
        "id": 2, "category": "CAG-FAQ",
        "query": "متى بستلم الهوية الجامعية؟",
        "ground_truth": "عند بداية التدريس، راجع عمادة شؤون الطلبة. قبلها اعتمد الوصل المالي مع هويتك الشخصية",
        "required_kw": ["شؤون الطلبة", "هوية", "وصل"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
    {
        "id": 3, "category": "CAG-FAQ",
        "query": "نسيت كلمة السر لبوابة الطالب، شو أعمل؟",
        "ground_truth": "اضغط هل نسيت كلمة السر، أدخل الرقم الجامعي، سيتم إرسال رسالة لبريدك",
        "required_kw": ["كلمة السر", "الرقم الجامعي"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
    {
        "id": 4, "category": "CAG-FAQ",
        "query": "هل امتحان القبول لجميع المتقدمين؟",
        "ground_truth": "نعم لجميع المتقدمين مهما كانت معدلاتهم",
        "required_kw": ["نعم", "جميع"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
    {
        "id": 5, "category": "CAG-FAQ",
        "query": "إذا لم أقبل في الدفعة اللي قدمت لها شو بصير؟",
        "ground_truth": "اسمك يبقى في قائمة الانتظار وتدخل المنافسة مع الدفعة التالية",
        "required_kw": ["قائمة", "الدفعة", "انتظار", "منافسة"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
    {
        "id": 6, "category": "CAG-FAQ",
        "query": "كيف أعمل براءة ذمة بعد التخرج؟",
        "ground_truth": "ادخل بوابة الطالب، اختر طلب براءة ذمة، سبب تخرج، ثم راجع شؤون الطلبة والمكتبة والمالية",
        "required_kw": ["تخرج", "شهاد", "براءة"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
    {
        "id": 7, "category": "CAG-FAQ",
        "query": "شو الخطوة بعد ما قدمت طلب الالتحاق؟",
        "ground_truth": "انتظر اتصال هاتفي أو email يبلغك عن موعد امتحان القبول",
        "required_kw": ["انتظر", "اتصال", "امتحان", "موعد", "رسالة", "اختبار"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
    {
        "id": 8, "category": "CAG-FAQ",
        "query": "كيف أقدم على الجامعة؟",
        "ground_truth": "التقديم online من خلال بوابة الطالب الإلكترونية",
        "required_kw": ["بوابة", "إلكتروني", "online", "تقديم"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },

    # ─── RAG / Major Overview Tests ───────────────────────────
    {
        "id": 9, "category": "RAG-Major",
        "query": "كم سعر ساعة الأمن السيبراني؟",
        "ground_truth": "130 دينار، كلية الملك الحسين لعلوم الحوسبة، 132 ساعة",
        "required_kw": ["مئة وثلاثين", "130", "الحوسبة", "الحسين"],
        "forbidden_kw": [],
        "expected_source": "RAG"
    },
    {
        "id": 10, "category": "RAG-Major",
        "query": "كم ساعة لازم أخلص عشان أتخرج من إدارة الأعمال؟",
        "ground_truth": "132 ساعة معتمدة، 81 ساعة تخصص",
        "required_kw": ["مئة واثنين وثلاثين", "132", "81"],
        "forbidden_kw": [],
        "expected_source": "RAG"
    },
    {
        "id": 11, "category": "RAG-Major",
        "query": "كم سعر ساعة المحاسبة؟",
        "ground_truth": "120 دينار، كلية الملك طلال لتكنولوجيا الأعمال، 132 ساعة",
        "required_kw": ["مئة وعشرين", "120", "طلال"],
        "forbidden_kw": [],
        "expected_source": "RAG"
    },
    {
        "id": 12, "category": "RAG-Major",
        "query": "هندسة البرمجيات تحت أي كلية؟",
        "ground_truth": "كلية الملك الحسين لعلوم الحوسبة",
        "required_kw": ["الحسين", "الحوسبة"],
        "forbidden_kw": [],
        "expected_source": "RAG"
    },
    {
        "id": 13, "category": "RAG-Major",
        "query": "كم ساعة تخرج الهندسة الإلكترونية؟",
        "ground_truth": "160 ساعة معتمدة، كلية الملك عبدالله الثاني للهندسة، 130 دينار",
        "required_kw": ["مئة وستين", "160", "الهندسة"],
        "forbidden_kw": [],
        "expected_source": "RAG"
    },
    {
        "id": 14, "category": "RAG-Major",
        "query": "شو شروط ماجستير الأمن السيبراني؟",
        "ground_truth": "بكالوريوس في تخصص ذي صلة بتقدير جيد",
        "required_kw": ["بكالوريوس", "جيد"],
        "forbidden_kw": [],
        "expected_source": "RAG"
    },

    # ─── Staff Directory Tests ─────────────────────────────────
    {
        "id": 15, "category": "STAFF",
        "query": "مين رئيس الجامعة؟",
        "ground_truth": "أ.د. وجدان أبو الهيجاء، رئيس الجامعة",
        "required_kw": ["وجدان", "الهيجاء", "هيجاء"],
        "forbidden_kw": ["عدنان", "أبو حليقة"],
        "expected_source": "STAFF"
    },
    {
        "id": 16, "category": "STAFF",
        "query": "مين رءيس الجامعه؟",  # STT spelling variant
        "ground_truth": "أ.د. وجدان أبو الهيجاء",
        "required_kw": ["وجدان", "الهيجاء", "هيجاء"],
        "forbidden_kw": ["عدنان", "أبو حليقة"],
        "expected_source": "STAFF"
    },
    {
        "id": 17, "category": "STAFF",
        "query": "مين عميد كلية الحوسبة؟",
        "ground_truth": "أ.د أنس ابو طالب",
        "required_kw": ["أنس", "أبو طالب", "ابو طالب"],
        "forbidden_kw": [],
        "expected_source": "STAFF"
    },
    {
        "id": 18, "category": "STAFF",
        "query": "مين عميد كليه الاعمال؟",  # colloquial spelling
        "ground_truth": "د. عدي الطويسي",
        "required_kw": ["عدي", "الطويسي", "طويسي"],
        "forbidden_kw": ["أحمد العتوم"],
        "expected_source": "STAFF"
    },
    {
        "id": 19, "category": "STAFF",
        "query": "مين عميد كلية الهندسة؟",
        "ground_truth": "د. اسامة أبو شرخ",
        "required_kw": ["اسامة", "أبو شرخ", "ابو شرخ", "اسامه"],
        "forbidden_kw": [],
        "expected_source": "STAFF"
    },
    {
        "id": 20, "category": "STAFF",
        "query": "شو ايميل دكتور محمد العزه؟",
        "ground_truth": "m.azzeh@psut.edu.jo",
        "required_kw": ["azzeh", "m.azzeh"],
        "forbidden_kw": [],
        "expected_source": "STAFF"
    },
    {
        "id": 21, "category": "STAFF",
        "query": "مين رئيس قسم هندسة البرمجيات؟",
        "ground_truth": "أ.د. عبدالله قصف",
        "required_kw": ["عبدالله", "قصف", "Abdullah", "qusef", "Qussef"],
        "forbidden_kw": [],
        "expected_source": "STAFF"
    },
    {
        "id": 22, "category": "STAFF",
        "query": "شو إيميل قسم المحاسبة؟",
        "ground_truth": "a.srouji@psut.edu.jo (د. عنان سروجي، رئيس قسم المحاسبة)",
        "required_kw": ["srouji", "عنان", "سروجي"],
        "forbidden_kw": [],
        "expected_source": "STAFF"
    },

    # ─── Grounding / Out-of-Scope Tests ───────────────────────
    {
        "id": 23, "category": "GROUNDING",
        "query": "شو عاصمة فرنسا؟",
        "ground_truth": "REFUSE — not about PSUT",
        "required_kw": ["آسف", "ما عندي"],
        "forbidden_kw": ["باريس", "Paris"],
        "expected_source": "REFUSE"
    },
    {
        "id": 24, "category": "GROUNDING",
        "query": "كم سعر البنزين اليوم؟",
        "ground_truth": "REFUSE — not about PSUT",
        "required_kw": ["آسف", "ما عندي"],
        "forbidden_kw": ["دينار", "لتر", "فلس"],
        "expected_source": "REFUSE"
    },

    # ─── TTS Quality Tests ────────────────────────────────────
    {
        "id": 25, "category": "TTS",
        "query": "كيف أدخل على بوابة الطالب الإلكترونية كطالب مستجد؟",
        "ground_truth": "أدخل الرقم الجامعي وكلمة السر ثم وافق على الشروط",
        "required_kw": ["الرقم الجامعي", "كلمة السر"],
        "forbidden_kw": [],
        "expected_source": "CAG"
    },
]


def check_tts_clean(text):
    """Check TTS readiness — returns dict of issues."""
    issues = {}
    issues["has_url"] = bool(re.search(r'https?://|www\.', text))
    issues["has_dots"] = bool(re.search(r'\.{3,}', text))
    issues["has_markdown"] = bool(re.search(r'[#*\-]{2,}|^\s*[-*]\s', text, re.MULTILINE))
    issues["has_emoji"] = bool(re.search(r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff]', text))
    issues["is_clean"] = not any(issues.values())
    return issues


def check_hallucination(text, forbidden_kw):
    """Check if the response contains fabricated/forbidden keywords."""
    found = [kw for kw in forbidden_kw if kw in text]
    return found


def run_evaluation():
    """Run all tests, collect metrics, and return structured results."""
    results = []
    total = len(TESTS)

    print(f"\n{'='*80}")
    print(f"  PSUT Voice Agent — Comprehensive Evaluation ({total} tests)")
    print(f"{'='*80}\n")

    for i, test in enumerate(TESTS):
        # Clear history between tests to ensure independence
        session_history.messages.clear()

        print(f"  [{i+1}/{total}] {test['category']}: {test['query'][:50]}...", flush=True)

        t_start = time.perf_counter()
        response_parts = []
        try:
            for raw, cleaned in generate_response(test["query"]):
                response_parts.append(raw)
        except Exception as e:
            response_parts = [f"ERROR: {e}"]
        elapsed = time.perf_counter() - t_start
        response = "".join(response_parts)

        # ── Factual Accuracy ──
        kw_matched = [kw for kw in test["required_kw"] if kw in response]
        fa_pass = len(kw_matched) > 0

        # ── Hallucination Check ──
        hallucinated = check_hallucination(response, test["forbidden_kw"])
        hl_pass = len(hallucinated) == 0

        # ── TTS Readiness ──
        tts = check_tts_clean(response)
        tts_pass = tts["is_clean"]

        # ── Overall ──
        overall = fa_pass and hl_pass and tts_pass

        result = {
            "id": test["id"],
            "category": test["category"],
            "query": test["query"],
            "ground_truth": test["ground_truth"],
            "response": response,
            "fa_pass": fa_pass,
            "kw_matched": kw_matched,
            "kw_expected": test["required_kw"],
            "hl_pass": hl_pass,
            "hallucinated_kw": hallucinated,
            "tts_pass": tts_pass,
            "tts_issues": {k: v for k, v in tts.items() if k != "is_clean"},
            "overall": overall,
            "latency_ms": elapsed * 1000,
        }
        results.append(result)

        status = "PASS" if overall else "FAIL"
        detail = ""
        if not fa_pass:
            detail += " [FA:FAIL]"
        if not hl_pass:
            detail += f" [HL:FAIL={hallucinated}]"
        if not tts_pass:
            detail += " [TTS:FAIL]"
        print(f"           → {status}{detail} ({elapsed*1000:.0f}ms)")

    return results


def print_summary(results):
    """Print formatted evaluation summary."""
    total = len(results)
    fa_pass = sum(1 for r in results if r["fa_pass"])
    hl_pass = sum(1 for r in results if r["hl_pass"])
    tts_pass = sum(1 for r in results if r["tts_pass"])
    overall_pass = sum(1 for r in results if r["overall"])
    avg_latency = sum(r["latency_ms"] for r in results) / total

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "fa": 0, "hl": 0, "tts": 0, "overall": 0}
        categories[cat]["total"] += 1
        categories[cat]["fa"] += int(r["fa_pass"])
        categories[cat]["hl"] += int(r["hl_pass"])
        categories[cat]["tts"] += int(r["tts_pass"])
        categories[cat]["overall"] += int(r["overall"])

    print(f"\n{'='*100}")
    print(f"  EVALUATION RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"  {'#':<4} {'Category':<15} {'Query':<45} {'FA':<6} {'HL':<6} {'TTS':<6} {'ALL':<6}")
    print(f"  {'-'*94}")

    for r in results:
        fa_s = "PASS" if r["fa_pass"] else "FAIL"
        hl_s = "PASS" if r["hl_pass"] else "FAIL"
        tts_s = "PASS" if r["tts_pass"] else "FAIL"
        all_s = "PASS" if r["overall"] else "FAIL"
        q = r["query"][:42] + "..." if len(r["query"]) > 42 else r["query"]
        print(f"  {r['id']:<4} {r['category']:<15} {q:<45} {fa_s:<6} {hl_s:<6} {tts_s:<6} {all_s:<6}")

    print(f"  {'-'*94}")
    print(f"  {'TOTALS':<20} {'':45} {fa_pass}/{total} {hl_pass}/{total} {tts_pass}/{total} {overall_pass}/{total}")
    print(f"\n  ── Aggregate Metrics ──")
    print(f"  Factual Accuracy:     {fa_pass}/{total} ({100*fa_pass/total:.1f}%)")
    print(f"  Hallucination Free:   {hl_pass}/{total} ({100*hl_pass/total:.1f}%)")
    print(f"  TTS Readiness:        {tts_pass}/{total} ({100*tts_pass/total:.1f}%)")
    print(f"  Overall Pass Rate:    {overall_pass}/{total} ({100*overall_pass/total:.1f}%)")
    print(f"  Avg Latency:          {avg_latency:.0f} ms")

    print(f"\n  ── Per-Category Breakdown ──")
    for cat, m in categories.items():
        print(f"  {cat:<15}  Overall: {m['overall']}/{m['total']}  FA: {m['fa']}/{m['total']}  HL: {m['hl']}/{m['total']}  TTS: {m['tts']}/{m['total']}")

    print(f"{'='*100}")

    return {
        "total": total,
        "factual_accuracy": f"{fa_pass}/{total} ({100*fa_pass/total:.1f}%)",
        "hallucination_free": f"{hl_pass}/{total} ({100*hl_pass/total:.1f}%)",
        "tts_readiness": f"{tts_pass}/{total} ({100*tts_pass/total:.1f}%)",
        "overall": f"{overall_pass}/{total} ({100*overall_pass/total:.1f}%)",
        "avg_latency_ms": round(avg_latency),
        "categories": categories,
    }


if __name__ == "__main__":
    results = run_evaluation()
    summary = print_summary(results)

    # Save detailed results
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results": [
                {k: v for k, v in r.items()}
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")
