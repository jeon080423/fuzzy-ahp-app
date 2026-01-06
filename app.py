import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gmean

# --- 1. 유틸리티 및 수식 함수 ---

def map_to_tfn(val):
    """사용자 입력값(-9~9)을 삼각형 퍼지 수(L, M, U)로 변환"""
    if val < 0: # 왼쪽이 중요 (예: -3 -> 3)
        m = abs(val)
        l, u = max(1, m-1), min(9, m+1)
        return (l, m, u)
    elif val > 0: # 오른쪽이 중요 (예: 3 -> 1/3)
        m = 1 / val
        l, u = 1/(val+1), 1/(max(1, val-1))
        return (l, m, u)
    else: # 동등 (0 또는 1)
        return (1, 1, 1)

def calculate_fahp_core(matrix_list, n):
    """퍼지 행렬로부터 가중치와 CR을 계산하는 핵심 로직"""
    # 1. Fuzzy Synthetic Extent (Chang's Method)
    row_sums = []
    for row in matrix_list:
        l_sum = sum(t[0] for t in row)
        m_sum = sum(t[1] for t in row)
        u_sum = sum(t[2] for t in row)
        row_sums.append((l_sum, m_sum, u_sum))
        
    total_l = sum(r[0] for r in row_sums)
    total_m = sum(r[1] for r in row_sums)
    total_u = sum(r[2] for r in row_sums)
    
    s_i = [(r[0]/total_u, r[1]/total_m, r[2]/total_l) for r in row_sums]
    
    # 2. 가중치 산출 (M값 기준 정규화)
    weights = np.array([s[1] for s in s_i])
    weights /= weights.sum()
    
    # 3. 일관성 지수 (Crisp Matrix 기반)
    crisp_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            crisp_matrix[i, j] = matrix_list[i][j][1]
            
    eig_val, _ = np.linalg.eig(crisp_matrix)
    max_eig = max(eig_val).real
    ci = (max_eig - n) / (n - 1) if n > 1 else 0
    ri_dict = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
    ri = ri_dict.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    
    return weights, cr, crisp_matrix

def suggest_correction(matrix, weights, n, criteria_names):
    """일관성이 낮은 경우 가장 수정이 필요한 지점 추천"""
    # 에러율이 가장 높은 위치 찾기 (a_ij * w_j / w_i 가 1에서 가장 먼 곳)
    max_error = 0
    target_pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            error = abs(matrix[i, j] * (weights[j] / weights[i]) - 1)
            if error > max_error:
                max_error = error
                target_pair = (i, j)
    
    suggested_val = weights[target_pair[0]] / weights[target_pair[1]]
    return f"[{criteria_names[target_pair[0]]}] vs [{criteria_names[target_pair[1]]}] 문항의 응답을 약 {suggested_val:.2f} 정도로 조정해 보세요."

# --- 2. Streamlit UI ---

st.set_page_config(page_title="고도화 퍼지 AHP 분석기", layout="wide")
st.title("🚀 고도화 퍼지 AHP 분석 어플리케이션")

with st.sidebar:
    st.header("⚙️ 분석 설정")
    criteria_input = st.text_area("평가 요소명 입력 (쉼표로 구분)", "요소A, 요소B, 요소C")
    criteria_names = [x.strip() for x in criteria_input.split(",")]
    n = len(criteria_names)
    num_needed = int(n * (n - 1) / 2)
    st.info(f"선택된 요소: {n}개\n필요한 응답 문항: {num_needed}개")

uploaded_file = st.file_uploader("엑셀 파일 업로드 (1열:ID, 2열:Type, 3열~:응답값)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if len(df.columns) < num_needed + 2:
        st.error(f"엑셀 열 개수가 부족합니다. 최소 {num_needed + 2}열이 필요합니다.")
    else:
        all_individual_matrices = []
        valid_matrices = []
        results = []

        # --- 개별 분석 ---
        for _, row in df.iterrows():
            resp_id, resp_type = row.iloc[0], row.iloc[1]
            raw_data = row.iloc[2:2+num_needed].values
            
            # 행렬 구축
            matrix = [[(1, 1, 1) for _ in range(n)] for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    tfn = map_to_tfn(raw_data[idx])
                    matrix[i][j] = tfn
                    matrix[j][i] = (1/tfn[2], 1/tfn[1], 1/tfn[0])
                    idx += 1
            
            weights, cr, crisp_mat = calculate_fahp_core(matrix, n)
            
            correction_msg = "-"
            if cr >= 0.1:
                correction_msg = suggest_correction(crisp_mat, weights, n, criteria_names)
            else:
                valid_matrices.append(matrix)
            
            res_entry = {"ID": resp_id, "유형": resp_type, "CR": round(cr, 4), "판단": "적합" if cr < 0.1 else "보정필요"}
            for name, w in zip(criteria_names, weights):
                res_entry[name] = round(w, 4)
            res_entry["보정 제안"] = correction_msg
            results.append(res_entry)
            all_individual_matrices.append(matrix)

        res_df = pd.DataFrame(results)

        # --- 결과 출력 ---
        st.subheader("1. 개별 응답 분석 결과")
        st.dataframe(res_df.style.applymap(lambda x: 'background-color: #ffcccc' if x == "보정필요" else '', subset=['판단']))

        # --- 그룹 종합 분석 (기하평균 활용) ---
        st.divider()
        st.subheader("2. 그룹 종합 분석 결과 (Group Decision Making)")
        
        if not valid_matrices:
            st.warning("일관성 기준(CR < 0.1)을 만족하는 응답이 없어 그룹 분석을 진행할 수 없습니다.")
        else:
            # 기하평균을 이용한 퍼지 행렬 통합
            group_matrix = [[None for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    l_vals = [m[i][j][0] for m in valid_matrices]
                    m_vals = [m[i][j][1] for m in valid_matrices]
                    u_vals = [m[i][j][2] for m in valid_matrices]
                    group_matrix[i][j] = (gmean(l_vals), gmean(m_vals), gmean(u_vals))
            
            group_weights, group_cr, _ = calculate_fahp_core(group_matrix, n)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**[종합 가중치 결과]**")
                group_res_df = pd.DataFrame({"요소": criteria_names, "가중치": group_weights})
                st.table(group_res_df.set_index("요소"))
                st.metric("Group Consistency Ratio (CR)", round(group_cr, 4))
            
            with col2:
                st.write("**[가중치 시각화]**")
                st.bar_chart(group_res_df.set_index("요소"))

        # --- 데이터 다운로드 ---
        csv = res_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("분석 결과 전체 다운로드 (CSV)", csv, "fahp_full_report.csv", "text/csv")