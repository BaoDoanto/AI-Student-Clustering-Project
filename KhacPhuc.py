import streamlit as st
import pandas as pd
import numpy as np

# Thêm phần này để code của bạn có dữ liệu để chạy
# (Đây là dữ liệu mẫu để khớp với logic của bạn)
df_raw = pd.read_csv("du_lieu_sinh_vien.csv")

# --- ĐÂY LÀ CODE CỦA BẠN (GIỮ NGUYÊN) ---
def phan_tich_va_khuyen_nghi(row):
    gpa = row['GPA']
    drl = row['Điểm RL']
    
    # Định nghĩa logic khắc phục
    if gpa >= 3.2 and drl < 80:
        nhan = "Nhóm Học Thuật (Mọt sách)"
        giai_phap = "⚠️ HÀNH ĐỘNG: Bắt buộc tham gia ít nhất 1 CLB kỹ năng mềm. Gợi ý làm Ban tổ chức các giải đấu chuyên môn để lấy điểm rèn luyện mà vẫn đúng sở thích."
        
    elif gpa < 2.5 and drl >= 80:
        nhan = "Nhóm Năng Nổ (Hoạt động bề nổi)"
        giai_phap = "⚠️ HÀNH ĐỘNG: Gửi cảnh báo học vụ sớm. Yêu cầu giảm tham gia Đoàn/Hội. Bắt buộc tham gia các nhóm học tập (Study Group) hoặc đăng ký gia sư đồng hành."
        
    elif gpa >= 3.2 and drl >= 80:
        nhan = "Nhóm Toàn Diện (Hạt giống)"
        giai_phap = "⭐ ĐỀ XUẤT: Xét duyệt học bổng. Giao vai trò Mentor (người hướng dẫn) cho Nhóm Năng Nổ để kéo thành tích chung của lớp lên."
        
    else:
        nhan = "Nhóm Cần Chú Ý (Yếu cả 2)"
        giai_phap = "🚨 BÁO ĐỘNG: Cố vấn học tập cần gặp mặt trực tiếp để tìm hiểu hoàn cảnh (tâm lý, gia đình, tài chính) và định hướng lại ngay lập tức."
        
    return pd.Series([nhan, giai_phap])

# Áp dụng logic vào DataFrame kết quả
df_raw[['Đặc trưng Nhóm', 'Giải pháp Quản lý đào tạo']] = df_raw.apply(phan_tich_va_khuyen_nghi, axis=1)

# hãy dùng st.write hoặc st.markdown de show ket qua
st.subheader("📊 BÁO CÁO GIẢI PHÁP QUẢN LÝ ĐÀO TẠO:")
st.markdown("---")

# Dùng vòng lặp để hiển thị lên web thay vì Terminal
for index, row in df_raw.sort_values(by='GPA', ascending=False).iterrows():
    # Tạo một container nhỏ cho mỗi sinh viên để nhìn rõ hơn
    with st.container(border=True):
        st.write(f"**[{row['Sinh viên']}]** - GPA: {row['GPA']} | ĐRL: {row['Điểm RL']} -> Thuộc {row['Nhóm (Cluster)']}")
        st.write(f"**Phân loại:** {row['Đặc trưng Nhóm']}")
        st.write(f"**=>** {row['Giải pháp Quản lý đào tạo']}")
# --- HẾT CODE CỦA BẠN ---