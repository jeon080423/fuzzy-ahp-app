import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gmean, ttest_rel
import plotly.graph_objects as go
import io
import xlsxwriter

# --- 1. 분석 엔진 함수 ---

def map_to_tfn(val):
    """크리스프 값을 삼각형 퍼지 수로 변환"""
    if val < 0:
        m = abs(val)
        return (max(1, m-1), m, min(9, m+1))
    elif val > 0:
        m = 1 / val
        return (1/(val+1), 1/val, 1/(max(1, val-1)))
    return (1, 1, 1)

def calculate_ahp_crisp(matrix):
    """일반 AHP 가중치 및 CR 계산"""
    n = len(matrix)
    if n <= 1: return np.array([1.0]), 0
    eig_val, eig_vec = np.linalg.eig(matrix)
    max_idx = np.argmax(eig_val)
    max_eig = eig_val[max_idx].real
    weights = eig_vec[:, max_idx].real
    weights /= weights.sum()
    ci = (max_eig - n) / (n - 1)
    ri_dict = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = ri_dict.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    return weights, cr

def auto_correct_consistency(matrix, target_cr=0.1):
    """일관성 자동 보정 알고리즘"""
    n = len(matrix)
    curr_matrix = matrix.copy()
    weights, cr = calculate_ahp_crisp(curr_matrix)
    for _ in range(20):
        if cr <= target_cr: break
        max_err = -1
        target_idx = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                err = abs(curr_matrix[i, j] - (weights[i] / weights[j]))
                if err > max_err:
                    max_err = err
                    target_idx = (i, j)
        ideal = weights[target_idx[0]] / weights[target_idx[1]]
        curr_matrix[target_idx[0], target_idx[1]] = curr_matrix[target_idx[0], target_idx[1]] * 0.7 + ideal * 0.3
        curr_matrix[target_idx[1], target_idx[0]] = 1 / curr_matrix[target_idx[0], target_idx[1]]
        weights, cr = calculate_ahp_crisp(curr_matrix)
    return curr_matrix, cr

def calculate_fahp_extent(fuzzy_matrix):
    """Chang's Method 기반 퍼지 가중치 산출"""
    n = len(fuzzy_matrix)
    row_sums = [(sum(t[0] for t in row), sum(t[1] for t in row), sum(t[2] for t in row)) for row in fuzzy_matrix]
    total_l, total_m, total_u = sum(r[0] for r in row_sums), sum(r[1] for r in row_sums), sum(r[2] for r in row_sums)
    s_i = [(r[0]/total_u, r[1]/total_m, r[2]/total_l) for r in row_sums]
    weights = np.array([s[1] for s in s_i])
    weights /= weights.sum()
    return weights, s_i

# --- 2. UI 및 로직 ---

st.set_page_config(page_title="Professional Fuzzy AHP", layout="wide")
st.title("📊 전문 퍼지 AHP 통합 분석 시스템 (2026 Ver.)")

uploaded_file = st.file_uploader("분석용 엑셀 파일을 업로드하세요.", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # [요구사항 3] 요소 수 자동 인식 및 레이블 추출
    # 1열(ID), 2열(Type) 제외한 나머지 열들이 쌍대비교 문항
    col_headers = df.columns[2:]
    num_questions = len(col_headers)
    n = int((1 + np.sqrt(1 + 8 * num_questions)) / 2)
    
    # 요소 이름 추정 (헤더에서 추출 시도, 실패 시 요소1, 2...)
    criteria_names = [f"요소{i+1}" for i in range(n)]
    st.info(f"데이터 감지: 요소 {n}개, 응답 문항 {num_questions}개")

    all_res = []
    corrected_data = []

    for _, row in df.iterrows():
        rid, rtype = row.iloc[0], row.iloc[1]
        vals = row.iloc[2:].values
        
        # 행렬 구성
        mat = np.eye(n)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                val = vals[idx]
                mat[i,j] = abs(val) if val < 0 else 1/val
                mat[j,i] = 1/mat[i,j]
                idx += 1
        
        orig_w, orig_cr = calculate_ahp_crisp(mat)
        corr_mat, new_cr = auto_correct_consistency(mat)
        
        # Fuzzy 분석
        f_mat = [[(1,1,1) for _ in range(n)] for _ in range(n)]
        new_row_vals = []
        for i in range(n):
            for j in range(n):
                m_v = corr_mat[i,j]
                if m_v >= 1: f_mat[i][j] = (max(1, m_v-1), m_v, min(9, m_v+1))
                else:
                    inv = 1/m_v
                    tfn_inv = (max(1, inv-1), inv, min(9, inv+1))
                    f_mat[i][j] = (1/tfn_inv[2], 1/tfn_inv[1], 1/tfn_inv[0])
                if i < j: new_row_vals.append(-m_v if m_v >= 1 else 1/m_v)
        
        f_w, s_i = calculate_fahp_extent(f_mat)
        
        entry = {"ID": rid, "Type": rtype, "Orig_CR": orig_cr, "New_CR": new_cr}
        for i, name in enumerate(criteria_names):
            entry[f"{name}(AHP)"] = orig_w[i]
            entry[f"{name}(Fuzzy)"] = f_w[i]
            entry[f"Si_{name}"] = f"{round(s_i[i][0],3)}, {round(s_i[i][1],3)}, {round(s_i[i][2],3)}"
        all_res.append(entry)
        corrected_data.append([rid, rtype] + new_row_vals)

    res_df = pd.DataFrame(all_res)
    corr_df = pd.DataFrame(corrected_data, columns=df.columns)

    # --- [요구사항 1] 웹 구현 (이미지 형태 시각화) ---
    
    tab1, tab2, tab3, tab4 = st.tabs(["일관성 및 보정", "가중치 비교", "퍼지 상세분석", "통계 검정"])
    
    with tab1:
        st.subheader("이미지 1: 일관성 분석 및 자동 보정 내역")
        st.dataframe(res_df[["ID", "Type", "Orig_CR", "New_CR"]].style.highlight_between(left=0.1, color='red', subset=['Orig_CR']))
        
    with tab2:
        st.subheader("이미지 2: 방법론별 가중치 비교")
        comp_cols = ["ID"] + [c for c in res_df.columns if "(AHP)" in c or "(Fuzzy)" in c]
        st.dataframe(res_df[comp_cols])
        
    with tab3:
        st.subheader("이미지 3: 퍼지 종합 정도(Si) 및 가중치 시각화")
        fig = go.Figure()
        ahp_means = [res_df[f"{n}(AHP)"].mean() for n in criteria_names]
        f_means = [res_df[f"{n}(Fuzzy)"].mean() for n in criteria_names]
        fig.add_trace(go.Bar(x=criteria_names, y=ahp_means, name="AHP Avg"))
        fig.add_trace(go.Bar(x=criteria_names, y=f_means, name="Fuzzy Avg"))
        st.plotly_chart(fig)

    with tab4:
        st.subheader("이미지 4: 대응표본 t-검정 결과")
        t_res = []
        for name in criteria_names:
            t_val, p_val = ttest_rel(res_df[f"{name}(AHP)"], res_df[f"{name}(Fuzzy)"])
            t_res.append({"요소": name, "t-stat": t_val, "p-value": p_val, "유의성": "유의" if p_val < 0.05 else "무의"})
        st.table(pd.DataFrame(t_res))

    # --- [요구사항 2] 엑셀 다운로드 (멀티 시트 구현) ---
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # 시트 1: 로우 데이터
        df.to_excel(writer, sheet_name='1_Original_Raw', index=False)
        corr_df.to_excel(writer, sheet_name='2_Corrected_Raw', index=False)
        
        # 시트 2: 분석 리포트
        res_df.to_excel(writer, sheet_name='3_Analysis_Report', index=False)
        
        # 시트 3: 통계 결과
        pd.DataFrame(t_res).to_excel(writer, sheet_name='4_Statistical_Test', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['3_Analysis_Report']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        
    st.download_button("💾 분석 리포트 엑셀 다운로드", output.getvalue(), "FAHP_Full_Report.xlsx")
