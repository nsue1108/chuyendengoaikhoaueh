# OKKIO Customer Care – Website v2

## Điểm mới
- Tra cứu hội viên bằng số điện thoại đăng ký (chuẩn hóa số dạng 84... về 0...).
- Đọc sheet `Danh_ba_hoi_vien` để liên kết số điện thoại và tên với mã hội viên nội bộ.
- Hiển thị cửa hàng và thời điểm giao dịch gần nhất dựa trên giao dịch mới nhất của hội viên.
- Cột `Xử lý`: đề xuất mức ưu tiên chăm sóc và bàn giao cho cửa hàng gần nhất.
- Cột `Marketing`: gợi ý kênh liên hệ Zalo/SMS/điện thoại khi có số điện thoại hợp lệ.
- Danh sách chăm sóc có thể lọc theo mức ưu tiên, cửa hàng và tình trạng có SĐT; xuất CSV để phân công.
- Bổ sung trang Marketing & Business: kết quả theo cửa hàng, xu hướng tháng, nhóm một lần ghé, nhóm quay lại, sử dụng mã giảm giá và gợi ý hoạt động.
- Ẩn các trang chỉ trình bày chỉ số/chi tiết kỹ thuật mô hình; website tập trung vào ứng dụng vận hành.

## Chạy local
```bash
pip install -r requirements.txt
export OKKIO_APP_PASSWORD='tao-mat-khau-rieng'
streamlit run app.py
```
Trên Windows PowerShell: `$env:OKKIO_APP_PASSWORD="tao-mat-khau-rieng"` trước khi chạy Streamlit.

## Bảo vệ truy cập
Ứng dụng yêu cầu mật khẩu qua biến môi trường `OKKIO_APP_PASSWORD` vì có hiển thị số điện thoại hội viên. Không bỏ lớp bảo vệ này khi dùng dữ liệu thật.

## Chạy trên Colab
Dùng notebook `OKKIO_Loyalty_Website_Colab.ipynb` hiện có:
1. Restart runtime để dừng website phiên cũ.
2. Upload ZIP v2 ở cell upload source.
3. Chạy lại lần lượt các cell.
4. Khi website mở, upload workbook loyalty trong sidebar của website.

## Cấu trúc workbook cần có
- `Giao_dich`: gồm mã hội viên, thời gian, ID giao dịch, chi nhánh, mã cửa hàng và số tiền.
- `Doi_diem`
- `TK_can_xem_xet`
- `Danh_ba_hoi_vien`: `Mã hội viên`, `SĐT`, `Tên hội viên`.

## Quy tắc xử lý và bảo mật
- Cửa hàng gần nhất là cửa hàng trên giao dịch mới nhất trong dữ liệu, không phải cửa hàng thường ghé nhất.
- Website chỉ tạo danh sách bàn giao; chưa tự động gửi Zalo/SMS, gọi điện, hoặc đẩy task vào hệ thống iPOS.
- Trước khi Marketing liên hệ, phải xác nhận quyền liên hệ/đồng ý nhận tin của hội viên.
- SĐT là dữ liệu cá nhân. Chỉ triển khai cho nhân sự được phân quyền; không chia sẻ tunnel URL công khai và không commit workbook lên GitHub.
- Điểm rủi ro là tín hiệu ưu tiên để rà soát/chăm sóc, không phải kết luận khách chắc chắn rời bỏ.
