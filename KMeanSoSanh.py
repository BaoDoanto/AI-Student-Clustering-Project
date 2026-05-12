import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import io

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH GIAO DIỆN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="So sánh K-Means & Hierarchical", layout="wide")

st.markdown("""
<style>
    .main-title { background: linear-gradient(135deg,#1e40af 0%,#3b82f6 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2rem; font-weight:800; }
    .compare-card { background:#f8fafc; border-radius:10px; padding:15px; border-left:5px solid #3b82f6; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧪 Thực nghiệm & So sánh thuật toán Gom cụm</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        # Ưu tiên đọc file CSV bạn đã cung cấp
        df = pd.read_csv("DuLieuSinhVien.csv")
        # Chuẩn hóa tên cột nếu có khoảng trắng hoặc ký tự lạ
        df.columns = [c.strip() for c in df.columns]
        return df
    except:
        st.error("Không tìm thấy file 'DuLieuSinhVien.csv'. Hãy đảm bảo file nằm cùng thư mục với code.")
        return None

df = load_data()

if df is not None:
    # Lấy dữ liệu số (GPA và Điểm RL)
    # Lưu ý: Cột Điểm RL trong file của bạn có thể là "Điểm RL" hoặc "Diem_RL"
    col_gpa = 'GPA'
    col_drl = 'Điểm RL' if 'Điểm RL' in df.columns else 'Diem_RL'
    
    X = df[[col_gpa, col_drl]].values
    
    # 1. CHUẨN HÓA DỮ LIỆU (Bắt buộc cho K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ─────────────────────────────────────────────────────────────────────────────
    # THỰC THI THUẬT TOÁN
    # ─────────────────────────────────────────────────────────────────────────────
    k = 4 # Số cụm cố định để so sánh
    
    # K-Means
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    df['KMeans'] = km.fit_predict(X_scaled)
    
    # Hierarchical (Ward)
    Z = linkage(X_scaled, method='ward')
    df['Hierarchical'] = fcluster(Z, k, criterion='maxclust')

    # ─────────────────────────────────────────────────────────────────────────────
    # TABS SO SÁNH
    # ─────────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Trực quan hóa so sánh", "📈 Chỉ số thực nghiệm", "📝 Đánh giá tổng quan"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🎯 Kết quả K-Means")
            fig1, ax1 = plt.subplots()
            sns.scatterplot(data=df, x=col_drl, y=col_gpa, hue='KMeans', palette='viridis', ax=ax1)
            st.pyplot(fig1)
            st.caption("K-Means phân hoạch không gian thành các vùng dựa trên tâm cụm.")
            
        with c2:
            st.subheader("🌳 Kết quả Hierarchical")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x=col_drl, y=col_gpa, hue='Hierarchical', palette='magma', ax=ax2)
            st.pyplot(fig2)
            st.caption("Hierarchical gom cụm dựa trên cấu trúc liên kết phân cấp.")

    with tab2:
        st.subheader("Chỉ số đo lường chất lượng")
        
        # Tính Silhouette Score
        sil_km = silhouette_score(X_scaled, df['KMeans'])
        sil_hc = silhouette_score(X_scaled, df['Hierarchical'])
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Silhouette (K-Means)", f"{sil_km:.4f}")
        col_m2.metric("Silhouette (Hierarchical)", f"{sil_hc:.4f}")
        
        st.markdown("""
        > **Giải thích:** Chỉ số Silhouette (từ -1 đến 1) càng cao chứng tỏ các cụm càng tách biệt và đặc khít. 
        """)
        
        # Bảng so sánh đặc điểm
        compare_data = {
            "Tiêu chí": ["Thời gian thực thi", "Độ ổn định", "Khả năng xử lý nhiễu", "Hình dạng cụm"],
            "K-Means": ["Nhanh (Tuyến tính)", "Phụ thuộc tâm cụm đầu", "Trung bình", "Hình tròn/cầu"],
            "Hierarchical": ["Chậm (Bình phương)", "Rất ổn định", "Tốt", "Linh hoạt"]
        }
        st.table(pd.DataFrame(compare_data))

    with tab3:
        st.markdown("""
        ### 🧐 Đánh giá thực nghiệm trên bộ dữ liệu 60 SV
        
        1. **Về ranh giới phân cụm:** - **K-Means** chia các nhóm khá "cứng nhắc" theo khoảng cách hình học. Nếu một sinh viên nằm ở biên, nó sẽ bị kéo về phía tâm gần nhất.
           - **Hierarchical** lại quan tâm đến việc sinh viên đó "gần giống" ai nhất trong quá khứ, nên ranh giới có phần mềm mại hơn.
        
        2. **Về độ ưu tiên:**
           - Với dữ liệu nhỏ (60 SV), **Hierarchical** cho kết quả trực quan và đáng tin cậy hơn vì ta có thể quan sát được Dendrogram.
           - Nếu dữ liệu tăng lên 60.000 SV, **K-Means** sẽ là lựa chọn duy nhất vì Hierarchical sẽ cực kỳ tốn tài nguyên máy tính.
           
        3. **Kết luận:**
           Thuật toán K-Means phù hợp để triển khai vào hệ thống quản lý tự động, trong khi Hierarchical Clustering phù hợp để giáo viên nghiên cứu sâu về cấu trúc lớp học.
        """)