import io
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

FORM_VIEW_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLScvqClkNtfCMKNgiF-b48W-yHBr0DA8F5yv6H2RHD1cOnpouQ/viewform"
)
FORM_RESPONSE_URL = FORM_VIEW_URL.replace("/viewform", "/formResponse")

MAIN_FORM_ROW_IDS = [
    76144874,
    1813949879,
    1119411314,
    1362691222,
    1527003955,
    573270383,
]
LOCAL_FORM_ROW_IDS = {
    "B": 1773698472,
    "R": 822263556,
    "H": 1974064118,
    "E": 1373343653,
}


class HiddenInputParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "input" and attributes.get("type") == "hidden":
            name = attributes.get("name")
            if name:
                self.values[name] = attributes.get("value", "")


def comparison_form_choice(rating):
    if np.isclose(rating, 1.0):
        return "متساويان 1"
    if rating > 1:
        return f"أهمية الطرف الأيسر {round(rating)}"
    return f"أهمية الطرف الأيمن {round(1.0 / rating)}"


def submit_to_google_form(
    expert_code,
    expert_spec,
    expert_exp,
    expert_org,
    expert_domains,
    consent,
    main_ratings,
    local_ratings,
):
    with urlopen(FORM_VIEW_URL, timeout=20) as response:
        parser = HiddenInputParser()
        parser.feed(response.read().decode("utf-8", errors="replace"))

    form_data = {
        "entry.1343181339": expert_spec.strip(),
        "entry.2146602075": expert_exp,
        "entry.951579804": expert_org,
        "entry.1726462633": expert_domains,
        "entry.1737297635": consent,
    }
    for pair, entry_id in zip(MAIN_PAIRS, MAIN_FORM_ROW_IDS):
        form_data[f"entry.{entry_id}"] = comparison_form_choice(
            main_ratings[pair]
        )
    for group_code, group in LOCAL_GROUPS.items():
        pair = group["pairs"][0]
        form_data[f"entry.{LOCAL_FORM_ROW_IDS[group_code]}"] = (
            comparison_form_choice(local_ratings[group_code][pair])
        )

    request = Request(
        FORM_RESPONSE_URL,
        data=urlencode(
            {**parser.values, **form_data}, doseq=True
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": FORM_VIEW_URL,
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        return response.status

RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}
STRENGTHS = [3, 5, 7, 9]

MAIN = {
    "S1": "أضرار المباني والركام",
    "S2": "أضرار الطرق والوصولية",
    "S3": "التعرض والحرمان الصحي",
    "S4": "التعرض والحرمان التعليمي",
}
MAIN_PAIRS = [
    ("S1", "S2"),
    ("S1", "S3"),
    ("S1", "S4"),
    ("S2", "S3"),
    ("S2", "S4"),
    ("S3", "S4"),
]

LOCAL_GROUPS = {
    "B": {
        "title": "مؤشرات أضرار المباني والركام",
        "criteria": {"B1": "نسبة التدمير الكلي", "B2": "كثافة الركام"},
        "pairs": [("B1", "B2")],
        "parent": "S1",
    },
    "R": {
        "title": "مؤشرات أضرار الطرق والوصولية",
        "criteria": {
            "R1": "نسبة أضرار الطرق",
            "R2": "متوسط المسافة لأقرب طريق",
        },
        "pairs": [("R1", "R2")],
        "parent": "S2",
    },
    "H": {
        "title": "مؤشرات التعرض والحرمان الصحي",
        "criteria": {
            "H1": "تعرض المنشآت الصحية",
            "H2": "المسافة لأقرب مستشفى/مركز",
        },
        "pairs": [("H1", "H2")],
        "parent": "S3",
    },
    "E": {
        "title": "مؤشرات التعرض والحرمان التعليمي",
        "criteria": {
            "E1": "تعرض المدارس",
            "E2": "السعة الطلابية المتأثرة",
        },
        "pairs": [("E1", "E2")],
        "parent": "S4",
    },
}


def ahp_weights(matrix: np.ndarray):
    """حساب أوزان AHP بالمتوسط الهندسي وحساب CR بالطريقة القياسية."""
    n = matrix.shape[0]
    geometric = np.prod(matrix, axis=1) ** (1.0 / n)
    weights = geometric / geometric.sum()
    if n <= 2:
        lambda_max, ci, cr = float(n), 0.0, 0.0
    else:
        aw = matrix @ weights
        lambda_max = float(np.mean(aw / weights))
        ci = (lambda_max - n) / (n - 1)
        ri = RI_TABLE.get(n, 1.12)
        cr = ci / ri if ri else 0.0
    return weights, float(lambda_max), float(ci), float(cr)


def build_matrix(codes, ratings):
    n = len(codes)
    matrix = np.ones((n, n), dtype=float)
    pos = {code: i for i, code in enumerate(codes)}
    for (left, right), value in ratings.items():
        value = float(value)
        i, j = pos[left], pos[right]
        matrix[i, j] = value
        matrix[j, i] = 1.0 / value
    return matrix


def comparison_widget(pair, labels, key_prefix):
    left, right = pair
    left_label, right_label = labels[left], labels[right]
    st.markdown(f"**{left_label}**  ← مقارنة →  **{right_label}**")
    direction = st.radio(
        "أي المعيارين أهم؟",
        [left_label, "متساويان", right_label],
        key=f"{key_prefix}_direction",
        horizontal=True,
    )
    if direction == "متساويان":
        st.caption("تم تثبيت درجة التساوي على 1 وفق مقياس ساتي.")
        return 1.0, "متساويان: 1"

    strength = st.select_slider(
        "درجة الأفضلية",
        options=STRENGTHS,
        value=5,
        key=f"{key_prefix}_strength",
        help="3 أفضلية طفيفة، 5 قوية، 7 قوية جداً، 9 مطلقة.",
    )
    if direction == left_label:
        return float(strength), f"{left_label} أهم بدرجة {strength}"
    return 1.0 / float(strength), f"{right_label} أهم بدرجة {strength}"


def collect_comparisons(labels, pairs, key_prefix):
    ratings, displayed = {}, {}
    for idx, pair in enumerate(pairs, start=1):
        st.markdown(f"#### المقارنة {idx} من {len(pairs)}")
        value, text = comparison_widget(pair, labels, f"{key_prefix}_{idx}")
        ratings[pair] = value
        displayed[f"{key_prefix}_{idx}"] = text
    return ratings, displayed


def main():
    st.set_page_config(page_title="استبانة AHP – خان يونس", layout="centered")
    st.title("استبانة تحليل التسلسل الهرمي (AHP)")
    st.subheader("تحديد أولويات إعادة الإعمار في مدينة خان يونس")
    st.info(
        "قارن كل زوج من المعايير. اختر المعيار الأهم ثم درجة الأفضلية. "
        "إذا كانا متساويين اختر «متساويان»، وستحسب الدرجة 1 تلقائياً. "
        "المشاركة طوعية وتستخدم الإجابات لأغراض البحث العلمي فقط."
    )

    st.markdown("### أولاً: بيانات الخبير")
    col1, col2 = st.columns(2)
    with col1:
        expert_code = st.text_input("رمز الخبير أو الأحرف الأولى (اختياري)")
        expert_spec = st.text_input("التخصص والخبرة ذات الصلة *")
        expert_exp = st.selectbox(
            "سنوات الخبرة ذات الصلة *",
            [
                "اختر...",
                "أقل من 5 سنوات",
                "من 5 إلى أقل من 10 سنوات",
                "من 10 إلى أقل من 15 سنة",
                "15 سنة فأكثر",
            ],
        )
    with col2:
        expert_org = st.selectbox(
            "قطاع العمل أو الجهة العامة *",
            [
                "اختر...",
                "جامعة أو مؤسسة أكاديمية",
                "وزارة أو بلدية أو جهة حكومية",
                "مكتب استشاري أو شركة خاصة",
                "منظمة إنسانية أو دولية",
                "مؤسسة صحية",
                "مؤسسة تعليمية",
                "أخرى",
            ],
        )
        expert_domains = st.multiselect(
            "مجال الخبرة المرتبط بالدراسة *",
            [
                "الهندسة المدنية أو الإنشائية",
                "نظم المعلومات الجغرافية أو الاستشعار عن بعد",
                "التخطيط الحضري أو البنية التحتية",
                "إدارة الكوارث أو إعادة الإعمار",
                "الطرق وشبكات النقل",
                "الخدمات الصحية",
                "الخدمات التعليمية",
            ],
        )
        consent = st.radio(
            "الموافقة على المشاركة البحثية *",
            [
                "أوافق على استخدام إجابتي لأغراض البحث العلمي مع الحفاظ على السرية",
                "لا أوافق",
            ],
            index=None,
        )

    st.markdown("### ثانياً: مقارنة المعايير الرئيسية")
    st.caption("مقياس ساتي: 3 = طفيفة، 5 = قوية، 7 = قوية جداً، 9 = مطلقة.")
    main_ratings, main_displayed = collect_comparisons(
        MAIN, MAIN_PAIRS, "main"
    )

    local_ratings, local_displayed = {}, {}
    for group_code, group in LOCAL_GROUPS.items():
        st.markdown(f"### {group['title']}")
        ratings, displayed = collect_comparisons(
            group["criteria"], group["pairs"], group_code
        )
        local_ratings[group_code] = ratings
        local_displayed.update(displayed)

    if st.button("إرسال الإجابة إلى Google Forms وحساب الأوزان", type="primary"):
        errors = []
        if not expert_spec.strip():
            errors.append("أدخل التخصص والخبرة ذات الصلة.")
        if expert_exp == "اختر...":
            errors.append("اختر سنوات الخبرة.")
        if expert_org == "اختر...":
            errors.append("اختر قطاع العمل.")
        if not expert_domains:
            errors.append("اختر مجال خبرة واحداً على الأقل.")
        if expert_code.strip():
            errors.append(
                "حقل رمز الخبير في Google Forms لا يقبل نصاً حراً حالياً. "
                "اتركه فارغاً أو غيّر نوع السؤال في النموذج إلى إجابة قصيرة."
            )
        if consent is None or consent.startswith("لا"):
            errors.append("اختر الموافقة على المشاركة البحثية لإرسال الإجابة.")
        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        main_codes = list(MAIN.keys())
        main_matrix = build_matrix(main_codes, main_ratings)
        main_weights_array, main_lambda, main_ci, main_cr = ahp_weights(
            main_matrix
        )
        main_weights = dict(zip(main_codes, main_weights_array))

        output = []
        consistency = [
            {
                "level": "main",
                "n": 4,
                "lambda_max": main_lambda,
                "CI": main_ci,
                "CR": main_cr,
            }
        ]
        for group_code, group in LOCAL_GROUPS.items():
            codes = list(group["criteria"].keys())
            matrix = build_matrix(codes, local_ratings[group_code])
            weights, lam, ci, cr = ahp_weights(matrix)
            local_weights = dict(zip(codes, weights))
            consistency.append(
                {
                    "level": group_code,
                    "n": len(codes),
                    "lambda_max": lam,
                    "CI": ci,
                    "CR": cr,
                }
            )
            for code, label in group["criteria"].items():
                output.append(
                    {
                        "parent_code": group["parent"],
                        "parent_label": MAIN[group["parent"]],
                        "sub_code": code,
                        "sub_label": label,
                        "local_weight": local_weights[code],
                        "global_weight": (
                            main_weights[group["parent"]] * local_weights[code]
                        ),
                    }
                )

        profile = {
            "expert_code": expert_code.strip(),
            "specialization": expert_spec.strip(),
            "experience": expert_exp,
            "sector": expert_org,
            "domains": "; ".join(expert_domains),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        consistency_df = pd.DataFrame(consistency)
        weights_df = pd.DataFrame(output)
        ratings_df = pd.DataFrame(
            [
                {
                    "expert_code": expert_code.strip(),
                    "specialization": expert_spec.strip(),
                    "experience": expert_exp,
                    "sector": expert_org,
                    "domains": "; ".join(expert_domains),
                    **main_displayed,
                    **local_displayed,
                }
            ]
        )

        try:
            response_status = submit_to_google_form(
                expert_code,
                expert_spec,
                expert_exp,
                expert_org,
                expert_domains,
                consent,
                main_ratings,
                local_ratings,
            )
            if response_status == 200:
                st.success("تم إرسال الإجابة إلى Google Forms بنجاح.")
            else:
                st.error(
                    "لم يؤكد Google Forms استلام الإجابة. يمكنك تنزيل الملفات "
                    "وإرسالها يدويًا."
                )
        except HTTPError as error:
            st.error(
                f"رفض Google Forms الإرسال (HTTP {error.code}). "
                "يمكنك تنزيل الملفات وإرسالها يدويًا."
            )
        except URLError:
            st.error(
                "تعذر الاتصال بـ Google Forms. يمكنك تنزيل الملفات وإرسالها "
                "يدويًا."
            )
        except Exception as error:
            st.error(
                "حدث خطأ أثناء إرسال الإجابة: "
                f"{type(error).__name__}. يمكنك تنزيل الملفات وإرسالها يدويًا."
            )

        st.success("تم حساب الأوزان بنجاح.")
        st.metric(
            "CR للمعايير الرئيسية",
            f"{main_cr:.4f}",
            delta="مقبول" if main_cr <= 0.10 else "يحتاج مراجعة",
        )
        st.dataframe(weights_df, use_container_width=True)
        st.dataframe(consistency_df, use_container_width=True)
        if main_cr > 0.10:
            st.warning("CR أكبر من 0.10؛ يجب مراجعة المقارنات قبل اعتماد الإجابة.")
        else:
            st.info("CR للمعايير الرئيسية مقبول. المقارنات المحلية الثنائية لها CR = 0.")

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            weights_df.to_excel(writer, index=False, sheet_name="weights")
            consistency_df.to_excel(writer, index=False, sheet_name="consistency")
            ratings_df.to_excel(writer, index=False, sheet_name="ratings")
            pd.DataFrame([profile]).to_excel(
                writer, index=False, sheet_name="profile"
            )
        st.download_button(
            "تنزيل ملف إجابة الخبير Excel",
            data=buffer.getvalue(),
            file_name=f"AHP_expert_{expert_code or 'anonymous'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "تنزيل أوزان المؤشرات CSV",
            data=weights_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="AHP_weights_output.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()