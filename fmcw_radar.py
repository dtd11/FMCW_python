import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    # -------------------------------
    # 1. 사용자 파라미터 설정
    # -------------------------------
    csv_file_path = "3m_rignt_20_1.csv"   # CSV 파일 경로
    skip_header_rows = 1                 # 첫 행을 무시
    remove_first_col = True              # 첫 열 제거 여부

    chirpperframe = 120 + 1  # 기존 코드에서 사용하던 값 (121)
    start_idx = 25           # 추가로 앞쪽 (frame) 몇 줄을 더 날리는 로직
    idle_point = 20          # 앞쪽 샘플 20개 버림
    conv_factor = 2.0 / 4096.0

    # CFAR 파라미터 (예시)
    num_train_range = 4
    num_guard_range = 2
    num_train_doppler = 4
    num_guard_doppler = 2
    threshold_scale = 1.25
    cfar_start = 20
    cfar_end = 50

    # -------------------------------
    # 2. CSV 읽기
    # -------------------------------
    # 예: 첫 행 skip, delimiter="," → data_raw.shape 파악
    data_raw = np.loadtxt(csv_file_path, delimiter=",", skiprows=skip_header_rows)

    # 첫 열 제거 (필요 시)
    if remove_first_col:
        Data = data_raw[:, 1:]
    else:
        Data = data_raw

    print("[DEBUG] CSV loaded shape =", Data.shape)

    # -------------------------------
    # 3. Rx별 데이터 분리 (4 Rx 예시)
    # -------------------------------
    # 실제로 Rx가 4개면,
    #   Rx1: Data[0:chirpperframe, :]
    #   Rx2: Data[chirpperframe:chirpperframe*2, :]
    #   Rx3: Data[chirpperframe*2:chirpperframe*3, :]
    #   Rx4: Data[chirpperframe*3:chirpperframe*4, :]
    #
    # 만약 8 Rx라면 8배(= chirpperframe*8)까지 슬라이싱해야 합니다.
    # 여기서는 4 Rx 예시.

    Rx1_data = Data[0 : chirpperframe, :]
    Rx2_data = Data[chirpperframe*1 : chirpperframe*2, :]
    Rx3_data = Data[chirpperframe*2 : chirpperframe*3, :]
    Rx4_data = Data[chirpperframe*3 : chirpperframe*4, :]

    # -------------------------------
    # 4. 1 CHIRP 제거 & idle_point 열 제거
    # -------------------------------
    def remove_chirp_and_idle(rx_data, idle_pt):
        """
        rx_data: shape = (chirpperframe, cols)
        1) 첫 행(1 CHIRP) 날림 => (chirpperframe-1, cols)
        2) idle_point만큼 열 제거 => [:, idle_pt:]
        3) Voltage 변환 => * conv_factor
        """
        if rx_data.shape[0] <= 1:
            raise ValueError("Not enough rows to remove 1 chirp.")

        rx_1chirp = rx_data[1:, :]             # (chirpperframe-1, ?)
        if rx_1chirp.shape[1] <= idle_pt:
            raise ValueError("Not enough columns to remove idle_point.")

        rx_idle = rx_1chirp[:, idle_pt:]       # 열 버리기
        rx_vol = rx_idle * conv_factor
        return rx_vol

    RAW_Rx1_data_Vol = remove_chirp_and_idle(Rx1_data, idle_point)
    RAW_Rx2_data_Vol = remove_chirp_and_idle(Rx2_data, idle_point)
    RAW_Rx3_data_Vol = remove_chirp_and_idle(Rx3_data, idle_point)
    RAW_Rx4_data_Vol = remove_chirp_and_idle(Rx4_data, idle_point)

    print("[DEBUG] RAW_Rx1_data_Vol shape =", RAW_Rx1_data_Vol.shape)
    # 예: (120, 3980) -> (119, 3980) -> (119, 3960) ... 실제 CSV마다 다름

    # -------------------------------
    # 5. Nd / total_pt 자동 조정
    # -------------------------------
    # "start_idx"만큼 더 row(프레임) 버리고, 남은 크기를 Nd로 삼음.
    # 열 개수 = total_pt.
    def shape_adjust(rx_vol, start_idx):
        """
        rx_vol: shape=(rows, cols)
        -> Nd = rows - start_idx
        -> total_pt = cols
        -> range_data.shape = (Nd, total_pt)
           range_data[:] = rx_vol[start_idx : start_idx+Nd, : total_pt]
        """
        rows, cols = rx_vol.shape
        Nd_ = rows - start_idx
        if Nd_ < 0:
            raise ValueError(f"start_idx({start_idx}) is too large for data row({rows}).")

        total_pt_ = cols
        # 실제로 slicing 시 min()을 사용해 인덱스 초과 방지
        valid_rows = min(Nd_, rows - start_idx)
        valid_cols = cols  # min(total_pt_, cols) -> 같을 것

        # range_data 초기화
        range_data = np.zeros((valid_rows, valid_cols), dtype=float)
        range_data[:valid_rows, :valid_cols] = rx_vol[start_idx:start_idx+valid_rows, :valid_cols]
        return range_data, valid_rows, valid_cols

    # Rx1만 예시로 range_rx1를 만든 뒤 FFT. (실제로는 Rx2, Rx3, Rx4도 마찬가지)
    range_rx1, Nd, total_pt = shape_adjust(RAW_Rx1_data_Vol, start_idx)
    print(f"[DEBUG] range_rx1 shape=({Nd},{total_pt}) after start_idx={start_idx}")

    # -------------------------------
    # 6. Range-Doppler FFT
    # -------------------------------
    def fft_2d(data_2d):
        """
        2D FFT: (Nd, total_pt)
         1) range FFT(axis=1)
         2) doppler FFT(axis=0)
         3) shift+scaling
        """
        # range FFT
        fft_1d = 2*np.fft.fft(data_2d, n=total_pt, axis=1)/total_pt
        # doppler FFT
        fft_2d_res = 2*np.fft.fftshift(np.fft.fft(fft_1d, n=Nd, axis=0), axes=0)/Nd
        return fft_2d_res

    start_fft_time = time.time()
    fft_rx1_2d = fft_2d(range_rx1)
    end_fft_time = time.time()
    print(f"[INFO] FFT 실행 시간: {end_fft_time - start_fft_time:.4f} 초")

    # dB 변환
    def to_dB(x):
        return 20*np.log10(np.abs(x)/10 + 1e-16) + 30

    rd1 = to_dB(fft_rx1_2d)

    # (만약 Rx2, Rx3, Rx4도 동일 로직 -> range_rx2_2d, range_rx3_2d, ...)
    # 여기서는 예시로 Rx1만 진행

    # Range축: 0~(total_pt/2)
    rd1_half = rd1[:, :total_pt//2]

    # -------------------------------
    # 7. 2D-CFAR
    # -------------------------------
    start_cfar_time = time.time()
    min_val = np.min(rd1_half)
    rd1_shifted = rd1_half - min_val  # 음수 없애기

    # cfar_start ~ cfar_end
    if cfar_end > (total_pt//2):
        cfar_end = (total_pt//2)  # 범위 제한

    cfar_region = rd1_shifted[:, cfar_start:cfar_end]
    nr, ndp = cfar_region.shape
    cfar_result = np.zeros((nr, ndp), dtype=int)

    for r in range(num_train_range+num_guard_range, nr-(num_train_range+num_guard_range)):
        for d in range(num_train_doppler+num_guard_doppler, ndp-(num_train_doppler+num_guard_doppler)):
            noise_level = 0.0
            num_training_cells = 0
            for rr in range(r-(num_train_range+num_guard_range), r+(num_train_range+num_guard_range)+1):
                for dd in range(d-(num_train_doppler+num_guard_doppler), d+(num_train_doppler+num_guard_doppler)+1):
                    if abs(rr-r)<=num_guard_range and abs(dd-d)<=num_guard_doppler:
                        continue
                    noise_level += cfar_region[rr, dd]
                    num_training_cells += 1
            avg_ = noise_level/(num_training_cells+1e-16)
            thresh_ = avg_*threshold_scale
            if cfar_region[r, d] > thresh_:
                cfar_result[r, d] = 1

    # 인접포인트 제거
    cfar_peak = np.argwhere(cfar_result==1)
    for i in range(len(cfar_peak)):
        for j in range(i+1, len(cfar_peak)):
            if abs(cfar_peak[i,0]-cfar_peak[j,0])<=1 and abs(cfar_peak[i,1]-cfar_peak[j,1])<=1:
                val_i = rd1_half[ cfar_peak[i,0], cfar_peak[i,1]+cfar_start ]
                val_j = rd1_half[ cfar_peak[j,0], cfar_peak[j,1]+cfar_start ]
                if val_i > val_j:
                    cfar_result[ cfar_peak[j,0], cfar_peak[j,1] ] = 0
                else:
                    cfar_result[ cfar_peak[i,0], cfar_peak[i,1] ] = 0

    final_peak = np.argwhere(cfar_result==1)
    # col 보정
    final_peak_col = final_peak[:,1]+cfar_start
    final_peak_row = final_peak[:,0]

    end_cfar_time = time.time()
    print(f"[INFO] CFAR 실행 시간: {end_cfar_time - start_cfar_time:.4f} 초")

    # -------------------------------
    # 8. 결과 시각화 & 저장
    # -------------------------------
    # Range axis: 0 ~ max_range
    # Velocity axis: 예시로 -vmax~+vmax
    # (실제로 lamda, doppler_resolution 등 계산해서 plot에 반영 가능)
    # 여기서는 간단히 rd1_half만 plot
    plt.figure(figsize=(8,6))
    plt.imshow(rd1_half, aspect='auto', origin='upper')
    plt.scatter(final_peak_col, final_peak_row, c='red', marker='x')
    plt.colorbar()
    plt.title("Range-Doppler (Rx1)")
    plt.xlabel("Range bin")
    plt.ylabel("Doppler bin")
    plt.show()

    np.savetxt("rd1_half.csv", rd1_half, delimiter=",")
    print("[INFO] rd1_half.csv saved.")
    detect_ranges = final_peak_col  # 실제론 range_axis에 매핑해야 함
    detect_vels = final_peak_row    # 실제론 velocity_axis에 매핑
    detect_arr = np.column_stack((detect_ranges, detect_vels))
    np.savetxt("cfar_detections.csv", detect_arr, delimiter=",", header="range_bin,doppler_bin", comments="")
    print("[INFO] cfar_detections.csv saved.")

if __name__=="__main__":
    main()

