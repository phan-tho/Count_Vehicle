## Đếm số xe mỗi làn
Làm theo plan trong file [vehicle_counting_plan.md](vehicle_counting_plan.md)

#### Init prompt:
Tôi muốn đếm số lượng phương tiện mỗi làn cho dự án IoT, đèn giao thông thích ứng, file 
`sa_ban_traffic.mp4` là video tôi đã quay màn hình bằng cam của chúng tôi trước đó, file `sample.png` là frame mà tôi chụp màn hình để test. 

Bạn hãy đọc kỹ plan. Tôi không biết gì cả, tôi sẽ vào từng bước một, và bạn sẽ cung cấp prompt tiếng anh để tôi mở cửa sổ agent mới và paste prompt của bạn vào (lưu ý là prompt cần hướng dẫn tôi manual test), sau đó tôi sẽ copy response của agent về đưa bạn xem, nếu ok thì bước tiếp theo


chúng tôi làm dự án đèn giao thông thích ứng, để demo cả phần mềm và phần cứng (iot), tôi đã code demo giả lập 4 ngã tư đường (hiện tại logic đèn đỏ đang để thời gian đếm ngược cố định hard code). Chúng tôi bật giả lập này, giả lập hiện trên màn hình máy tính, tôi dùng 1 camera để quay màn hình máy tính này, tôi đã viết phần mềm đếm số xe mỗi làn đường và xuất ra csv format "Frame,TimeStamp,Lane_0_Cars,Lane_0_Bikes,Lane_1_Cars,...,Lane_15_Bikes
0,11-51-06,0,0,5,...". Khi lắp full pipeline, tôi sẽ bật giả lập, quay màn hình bằng cam (kết nối với pi5), pi5 sau đó xử lý ảnh, đếm số xe, sau đó dữ liệu số xe này sẽ được ném lên server, server sẽ có thuật toán để chỉnh đèn đỏ, sau đó sẽ gửi tín hiệu đèn đỏ về cho phần mềm chạy giả lập. Giờ tôi muốn bạn viết 1 file markdown docs về giao thức kết nối, format gói tin pi5 truyền server, server truyền máy chạy giả lập. file này của bạn tôi sẽ dán vào các folder dự án và prompt AI làm theo (tôi không biết code). Giờ bạn hãy viết file md này cho tôi. Còn nữa, pi5 gửi lên server csv kiểu vậy có hợp lý không?

có 1 vài điểm cần sửa. Đầu tiên, AI agent cũng rất khôn, bạn chỉ cần viết format chung chung, không viết qúa chi tiết như vậy gây cứng ngắc. Thứ 2 là có 1 số parameter của tôi bạn cần sửa lại cho match với giả lập và phần mềm đếm xe. ngã tư 0 (top left) có lane 10, 3, 9, 0 (chiều kim 12h -> 3 6 9). ngã tư 1 (top right) có lane 14, 2, 13, 1. ngã tư 2 (bottom lèft) có lane 11, 7, 8, 4. ngã tư 3 (bottom right) có lane 15, 6, 12, 5. Nếu bạn thấy đánh số linh tinh này bất tiện thì bạn có thể viết 1 đoạn mapping sang cách đánh chuẩn của bạn. Bạn bỏ đoạn giải thích "Tại sao không dùng CSV để truyền real-time:" này đi, tôi chốt là dùng json, bạn hãy viết lại format file json. Cuối cùng là ở mỗi ngã tư, có 4 làn cho xe. Với mỗi làn có 2 cái đèn đỏ (1 cái đèn đỏ đi thẳng, 1 cái rẽ trái)