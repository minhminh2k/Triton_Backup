import os
import shutil
import random

# Đường dẫn đến thư mục chứa 15000 ảnh
thu_muc_goc = './test_v2'

# Đường dẫn đến thư mục đích
os.makedirs("./input_test_v2_8000/", exist_ok=True)

thu_muc_dich = './input_test_v2_8000/'

# Số lượng ảnh bạn muốn sao chép
so_luong_anh = 8000

# Lấy danh sách tất cả các file ảnh từ thư mục gốc
tat_ca_anh = os.listdir(thu_muc_goc)

# Ngẫu nhiên chọn 3000 ảnh từ danh sách tất cả ảnh
anh_can_copy = random.sample(tat_ca_anh, so_luong_anh)

# Sao chép các ảnh được chọn sang thư mục đích
for anh in anh_can_copy:
    duong_dan_goc = os.path.join(thu_muc_goc, anh)
    duong_dan_dich = os.path.join(thu_muc_dich, anh)
    shutil.copy(duong_dan_goc, duong_dan_dich)

print(f'{so_luong_anh} ảnh đã được sao chép từ {thu_muc_goc} đến {thu_muc_dich}.')


'''
thu_muc_goc = './input_test_v2_3000'

# Đường dẫn đến thư mục đích
os.makedirs("./no_ship/", exist_ok=True)

thu_muc_dich = './no_ship/'

# Số lượng ảnh bạn muốn sao chép
so_luong_anh = len(image_files_ship)

for anh in image_files_ship:
    duong_dan_goc = os.path.join(thu_muc_goc, anh)
    duong_dan_dich = os.path.join(thu_muc_dich, anh)
    shutil.copy(duong_dan_goc, duong_dan_dich)

print(f'{so_luong_anh} ảnh đã được sao chép từ {thu_muc_goc} đến {thu_muc_dich}.')
'''