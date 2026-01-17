function [W_opt] = firefly_standard(instr, Aq, PdM, alpha_idx, M)
    rng(150);
    % instr: [Số đom đóm, Số vòng lặp]
    
    if nargin < 1, instr = [40 100]; end
    n = instr(1);
    MaxGeneration = instr(2);
    
    % THAM SỐ CHUẨN (CỐ ĐỊNH)
    alpha = 0.2;   
    gamma = 1.0;
    beta0 = 1.0;
    
    Lb = -pi * ones(1, M);
    Ub = pi * ones(1, M);
    
    % 1. KHỞI TẠO NGẪU NHIÊN (Random Uniform)
    ns = Lb + (Ub - Lb) .* rand(n, M);
    Lightn = zeros(n, 1);
    for i = 1:n
        Lightn(i) = cost_function_basic(ns(i, :), Aq, PdM, alpha_idx);
    end
    
    % 2. VÒNG LẶP
    for k = 1:MaxGeneration
        nso = ns;
        Lighto = Lightn;
        
        % Di chuyển
        for i = 1:n
            for j = 1:n
                if Lightn(i) < Lighto(j)
                    r = sqrt(sum((ns(i,:) - nso(j,:)).^2));
                    beta = beta0 * exp(-gamma * r^2);
                    
                    % Công thức chuẩn: alpha * (rand - 0.5)
                    noise = alpha .* (rand(1, M) - 0.5) .* (Ub(1)-Lb(1));
                    ns(i, :) = ns(i, :) .* (1 - beta) + nso(j, :) .* beta + noise;
                    
                    % Kiểm tra biên
                    ns(i, :) = max(ns(i, :), Lb);
                    ns(i, :) = min(ns(i, :), Ub);
                    
                    % Cập nhật độ sáng
                    Lightn(i) = cost_function_basic(ns(i,:), Aq, PdM, alpha_idx);
                end
            end
        end
    end
    
    % Kết quả
    [~, best_idx] = max(Lightn);
    best_phases = ns(best_idx, :);
    W_opt = exp(1i * best_phases).'; 
end

% Hàm Cost cơ bản
function I = cost_function_basic(phases, Aq, PdM, alpha_idx)
    W = exp(1i * phases).';
    P_actual = abs(W' * Aq);
    if max(P_actual) > 0, P_actual = P_actual / max(P_actual); end
    if max(PdM) > 0, PdM = PdM / max(PdM); end
    
    diff = sum(abs(P_actual(alpha_idx) - PdM(alpha_idx)));
    I = 1 / (diff + 1e-10);
end