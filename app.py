import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="استبانة AHP - أولويات إعادة الإعمار في خان يونس",
    layout="centered",
)

st.title("🌐 استبانة تحليل التسلسل الهرمي (AHP)")
st.subheader(
    "تقييم أولويات إعادة الإعمار وإدارة الركام في مدينة خان يونس"
)
st.write(
    "يرجى مقارنة المعايير والمؤشرات ثنائياً باستخدام مقياس ساتي (من 1 إلى 9)،"
    " حيث تعبر القيمة 1 عن التساوي، والقيم الكبرى عن رجحان أهمية أحد المعايير."
)

main_criteria = {
    "S1": "أضرار المباني والركام",
    "S2": "أضرار الطرق والوصولية",
    "S3": "التعرض والحرمان الصحي",
    "S4": "التعرض والحرمان التعليمي",
}

saaty_options = {
    "1 - تساوي الأهمية": 1,
    "2 - قيمة وسطية": 2,
    "3 - أهمية طفيفة": 3,
    "4 - قيمة وسطية": 4,
    "5 - أهمية قوية": 5,
    "6 - قيمة وسطية": 6,
    "7 - أهمية قوية جداً": 7,
    "8 - قيمة وسطية": 8,
    "9 - أهمية مطلقة": 9,
}

st.markdown("---")
st.markdown("### أولاً: مقارنة المعايير الرئيسية (Main Criteria)")

pairs_main = [
    ("S1", "S2"),
    ("S1", "S3"),
    ("S1", "S4"),
    ("S2", "S3"),
    ("S2", "S4"),
    ("S3", "S4"),
]

ratings_main = {}
for code1, code2 in pairs_main:
  label1 = main_criteria[code1]
  label2 = main_criteria[code2]

  col1, col2, col3 = st.columns([3, 2, 3])
  with col1:
    st.write(f"**{label1}**")
  with col2:
    direction = st.radio(
        f"الطرف الأرجح بين {code1} و {code2}",
        [label1, "متساويان", label2],
        key=f"dir_{code1}_{code2}",
    )
  with col3:
    choice = st.selectbox(
        f"درجة الأهمية",
        list(saaty_options.keys()),
        key=f"val_{code1}_{code2}",
    )

  val = saaty_options[choice]
  if direction == label1:
    ratings_main[(code1, code2)] = val
  elif direction == label2:
    ratings_main[(code1, code2)] = 1.0 / val
  else:
    ratings_main[(code1, code2)] = 1.0


def calculate_ahp(criteria_keys, ratings):
  n = len(criteria_keys)
  matrix = np.ones((n, n))

  keys_list = list(criteria_keys)
  for i in range(n):
    for j in range(n):
      if i == j:
        matrix[i, j] = 1.0
      elif i < j:
        c1, c2 = keys_list[i], keys_list[j]
        matrix[i, j] = ratings.get((c1, c2), 1.0)
        matrix[j, i] = 1.0 / matrix[i, j]

  col_sum = matrix.sum(axis=0)
  norm_matrix = matrix / col_sum
  weights = norm_matrix.mean(axis=1)

  lambda_max = np.dot(col_sum, weights)
  ci = (lambda_max - n) / (n - 1) if n > 1 else 0

  ri_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}
  ri = ri_dict.get(n, 1.12)
  cr = ci / ri if ri > 0 else 0

  return weights, cr, matrix


if st.button("حساب النتائج وتصدير الملف"):
  weights, cr, mat = calculate_ahp(main_criteria.keys(), ratings_main)

  st.success("تم حساب الأوزان بنجاح!")
  st.write(f"**نسبة الاتساق (CR):** {cr:.4f}")

  if cr <= 0.10:
    st.info("النتائج متسقة ومقبولة أكاديمياً (CR <= 0.10).")
  else:
    st.warning(
        "تنبيه: نسبة الاتساق مرتفعة (CR > 0.10)، يُفضل مراجعة المقارنات."
    )

  df_res = pd.DataFrame(
      {"المعيار": list(main_criteria.values()), "الوزن النسبي": weights}
  )
  st.dataframe(df_res)

  csv_data = df_res.to_csv(index=False, encoding="utf-8-sig")
  st.download_button(
      label="📥 تحميل النتائج والأوزان (CSV)",
      data=csv_data,
      file_name="AHP_weights_results.csv",
      mime="text/csv",
  )
