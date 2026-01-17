% ====================================================================
% THUẬT TOÁN ĐOM ĐÓM CẢI TIẾN (IMPROVED FIREFLY ALGORITHM)
% Tính năng:
%   1. Smart Initialization (Khởi tạo dựa trên nghiệm Capon/ILS)
%   2. Manual Tuning Weights (Trọng số chỉnh tay kiểm soát hình dáng)
%   3. Hybrid Optimization (Kết hợp tìm kiếm pha và giải tích biên độ)
% ====================================================================
function [W_opt] = firefly_improve(instr, Aq, PdM, alpha_idx, M, eqDir, W0)
    % 1. CỐ ĐỊNH KẾT QUẢ (Reproducibility)
    rng(100); 
    
    if nargin < 1, instr = [100 800]; end
    n = instr(1);               % Kích thước quần thể
    MaxGeneration = instr(2);   % Số vòng lặp tối đa
    
    % 2. THAM SỐ FIREFLY
    alpha = 0.5;                % Hệ số ngẫu nhiên khởi điểm
    alpha_end = 1e-6;           % Hệ số ngẫu nhiên kết thúc
    delta = (alpha_end / alpha)^(1/MaxGeneration);
    beta0 = 1.0;
    gamma_val = 1.0;
    
    % 3. CẤU HÌNH TRỌNG SỐ (Chiến thuật "Sharp & Suppressed")
    cfg.w_main = 15000;       % Ưu tiên hình dạng búp chính
    cfg.w_null_force = 10000; % Ưu tiên độ dốc (ép hẹp búp sóng)
    cfg.w_sll = 2000;         % Ưu tiên nén búp sóng phụ
    cfg.w_sym = 500;          % Ưu tiên đối xứng
    cfg.target_sll = -25;     % Ngưỡng mục tiêu búp phụ (-25dB)
    
    % 4. KHỞI TẠO VÙNG TÌM KIẾM
    % Xác định vùng búp chính và điểm neo (Null Anchors)
    main_idx = find(PdM > 0.01);
    dim_v = length(main_idx);
    
    left_anchor = main_idx(1) - 1;
    right_anchor = main_idx(end) + 1;
    
    % Tinh chỉnh điểm neo để đảm bảo tính hợp lệ
    if left_anchor > 0 && PdM(left_anchor) > 0.001, left_anchor = left_anchor - 1; end
    if right_anchor <= length(PdM) && PdM(right_anchor) > 0.001, right_anchor = right_anchor + 1; end
    null_anchors = [left_anchor, right_anchor];
    null_anchors = null_anchors(null_anchors > 0 & null_anchors <= length(PdM));
    
    [~, zero_idx] = min(abs(eqDir)); % Chỉ số hướng 0 độ
    Lb = -pi * ones(1, dim_v); 
    Ub = pi * ones(1, dim_v);
    
    % --- KHỞI TẠO THÔNG MINH (SMART INITIALIZATION) ---
    if nargin >= 7 && ~isempty(W0)
        % Nếu có W0 (từ Capon/ILS), dùng làm mồi dẫn hướng
        initial_phases = angle(W0' * Aq(:, main_idx))';  
        % Tạo quần thể tập trung quanh nghiệm W0 với độ phân tán nhỏ
        ns = repmat(initial_phases', n, 1) + 0.1 * (rand(n, dim_v) - 0.5) * 2 * pi;
    else
        % Nếu không, khởi tạo ngẫu nhiên toàn cục (Standard FA)
        ns = Lb + (Ub - Lb) .* rand(n, dim_v);
    end
    
    Lightn = zeros(n, 1);
    
    % Đánh giá chất lượng ban đầu
    for i = 1:n
        Lightn(i) = cost_func_manual(ns(i, :), Aq, PdM, main_idx, alpha_idx, zero_idx, null_anchors, cfg);
    end
    
    [~, best_idx] = max(Lightn);
    global_best_sol = ns(best_idx, :);
    global_best_fit = Lightn(best_idx);
    
    % 5. VÒNG LẶP CHÍNH (MAIN LOOP)
    for k = 1:MaxGeneration
        nso = ns; Lighto = Lightn;
        
        % Sắp xếp đom đóm (Sáng nhất lên đầu)
        [Lightn, sort_idx] = sort(Lightn, 'descend');
        ns = ns(sort_idx, :);
        
        for i = 1:n
            moved = false;
            for j = 1:n
                % Nếu con j sáng hơn con i -> Di chuyển i về phía j
                if Lightn(i) < Lighto(j)
                    r = sqrt(sum((ns(i,:) - nso(j,:)).^2));
                    beta = beta0 * exp(-gamma_val * r^2);
                    
                    % Nhiễu ngẫu nhiên giảm dần theo thời gian (alpha)
                    noise = alpha * (rand(1, dim_v) - 0.5) .* (Ub-Lb);
                    
                    % Cập nhật vị trí
                    ns(i, :) = ns(i, :) .* (1 - beta) + nso(j, :) .* beta + noise;
                    % Ràng buộc biên
                    ns(i, :) = max(min(ns(i, :), Ub), Lb);
                    moved = true;
                end
            end
            
            % Nếu có di chuyển, đánh giá lại độ sáng
            if moved
                fit_val = cost_func_manual(ns(i, :), Aq, PdM, main_idx, alpha_idx, zero_idx, null_anchors, cfg);
                Lightn(i) = fit_val;
                
                % Cập nhật nghiệm toàn cục tốt nhất
                if fit_val > global_best_fit
                    global_best_fit = fit_val;
                    global_best_sol = ns(i, :);
                end
            end
        end
        
        % Giảm độ ngẫu nhiên
        alpha = alpha * delta;
    end
    
    % 6. TÍNH TOÁN TRỌNG SỐ TỐI ƯU (HYBRID STEP)
    best_phases = global_best_sol;
    total_len = length(PdM);
    
    % Tái tạo vector mục tiêu v từ pha tốt nhất
    v = zeros(total_len, 1);
    v(main_idx) = PdM(main_idx).' .* exp(1i * best_phases.');
    V = Aq(:, alpha_idx);
    v_target = v(alpha_idx);
    
    % Tính biên độ tối ưu bằng Nghịch đảo giả (Regularized Pinv)
    % Tham số 2e-8 giúp cân bằng độ sắc nét và tính ổn định
    W_opt = pinv(V', 2e-8) * v_target;
    
    % Chuẩn hóa pha tại đỉnh 0 độ
    a0 = Aq(:, zero_idx);
    phase_shift = angle(W_opt' * a0);
    W_opt = W_opt * exp(-1i * phase_shift);
end

% ====================================================================
% HÀM MỤC TIÊU ĐA TRỌNG SỐ (MULTI-OBJECTIVE COST FUNCTION)
% ====================================================================
function Fitness = cost_func_manual(phases, Aq, PdM, main_idx, alpha_idx, zero_idx, null_anchors, cfg)
    
    % 1. Tái tạo trọng số W từ vector pha ứng viên
    total_len = length(PdM);
    v = zeros(total_len, 1);
    v(main_idx) = PdM(main_idx).' .* exp(1i * phases.');
    V = Aq(:, alpha_idx);
    w = pinv(V', 2e-8) * v(alpha_idx); % Hybrid calculation
    
    % 2. Tính đáp ứng mảng
    P_complex = w' * Aq;
    P_abs = abs(P_complex);
    
    % Chuẩn hóa theo đỉnh
    val_at_zero = P_abs(zero_idx);
    if val_at_zero > 0, P_norm = P_abs / val_at_zero; else, P_norm = P_abs; end
    
    PdM_norm = PdM; 
    if max(PdM) > 0, PdM_norm = PdM / max(PdM); end
    
    % 3. Tính toán các thành phần lỗi
    % Lỗi búp chính
    err_main = sum((P_norm(main_idx) - PdM_norm(main_idx)).^2);
    
    % Lỗi tại điểm neo (ép độ dốc)
    if ~isempty(null_anchors)
        vals_at_anchor = P_norm(null_anchors);
        err_null = sum(vals_at_anchor.^2);
    else
        err_null = 0;
    end
    
    % Lỗi búp sóng phụ (chỉ phạt nếu vượt ngưỡng mục tiêu)
    mask_sll = true(size(P_norm));
    mask_sll(main_idx) = false;
    mask_sll(null_anchors) = false;
    sll_vals = P_norm(mask_sll);
    
    viol = max(0, 20*log10(sll_vals + 1e-9) - cfg.target_sll);
    err_sll = sum(viol.^2);
    
    % Lỗi đối xứng
    L = length(P_norm); half_L = floor(L/2);
    left_part = P_norm(1:half_L);
    right_part = P_norm(end:-1:end-half_L+1);
    err_sym = sum(abs(left_part - right_part).^2);
    
    % 4. Tổng hợp chi phí có trọng số
    Cost = cfg.w_main * err_main + ...
           cfg.w_null_force * err_null + ...
           cfg.w_sll * err_sll + ...
           cfg.w_sym * err_sym;
       
    Fitness = 1 / (Cost + 1e-12);
end