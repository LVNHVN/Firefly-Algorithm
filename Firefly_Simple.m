% ====================================================================
% THUẬT TOÁN ĐOM ĐÓM (FIREFLY ALGORITHM - FA)
% Tác giả gốc: Xin-She Yang
% Tài liệu tham khảo: Nature-Inspired Metaheuristic Algorithms, 2nd Edition
% ====================================================================

function [best] = firefly_simple(instr)
    if nargin < 1, instr = [25 50]; end
    
    n = instr(1);               % Số lượng đom đóm (Population size)
    MaxGeneration = instr(2);   % Số vòng lặp tối đa (Max generations)
    
    % --- 1. ĐỊNH NGHĨA HÀM MỤC TIÊU (Objective Function) ---
    % Hàm 4 đỉnh (Four-peak function) để kiểm tra khả năng tìm cực đại
    str1 = 'exp(-(x-4).^2-(y-4).^2) + exp(-(x+4).^2-(y-4).^2)';
    str2 = '+ 2*exp(-x.^2-(y+4).^2) + 2*exp(-x.^2-y.^2)';
    funstr = strcat(str1, str2);
    f = vectorize(inline(funstr)); % Chuyển chuỗi thành hàm nội tuyến
    
    % Phạm vi tìm kiếm [-5, 5] cho cả x và y
    range = [-5 5 -5 5]; 
    
    % --- 2. THIẾT LẬP THAM SỐ (Parameters) ---
    alpha = 0.2;      % Tham số ngẫu nhiên hóa (Randomness)
    gamma = 1.0;      % Hệ số hấp thụ ánh sáng (Absorption coefficient)
    beta0 = 1.0;      % Độ hấp dẫn tại r=0
    
    % --- 3. KHỞI TẠO (Initialization) ---
    [xn, yn, Lightn] = init_ffa(n, range);
    
    % Vẽ đồ thị hàm mục tiêu để theo dõi
    figure(1);
    Ngrid = 100;
    dx = (range(2)-range(1))/Ngrid;
    dy = (range(4)-range(3))/Ngrid;
    [x,y] = meshgrid(range(1):dx:range(2), range(3):dy:range(4));
    z = f(x,y);
    surfc(x,y,z); title('Hàm mục tiêu 4 đỉnh');
    
    % --- 4. VÒNG LẶP CHÍNH (Main Loop) ---
    figure(2); % Cửa sổ hiển thị quá trình di chuyển
    
    for i = 1:MaxGeneration
        
        % Đánh giá cường độ sáng tại vị trí hiện tại
        zn = f(xn, yn);
        
        % Xếp hạng đom đóm (Sắp xếp theo độ sáng)
        [Lightn, Index] = sort(zn);
        xn = xn(Index); 
        yn = yn(Index);
        
        % Lưu vị trí cũ để so sánh
        xo = xn; 
        yo = yn; 
        Lighto = Lightn;
        
        % Vẽ vị trí đom đóm hiện tại
        contour(x,y,z, 15); hold on;
        plot(xn, yn, '.', 'markersize', 10, 'markerfacecolor', 'g');
        title(['Thế hệ: ', num2str(i)]);
        axis(range); 
        hold off; 
        drawnow;
        
        % Di chuyển (Movement)
        [xn, yn] = ffa_move(xn, yn, Lightn, xo, yo, Lighto, alpha, gamma, range, beta0);
        
    end % Kết thúc vòng lặp
    
    % Kết quả tốt nhất
    best(:,1) = xo'; 
    best(:,2) = yo'; 
    best(:,3) = Lighto';
    
    disp('Vị trí tối ưu (x, y) và Giá trị hàm (z):');
    disp(best(end, :)); % Hiển thị con tốt nhất (cuối danh sách sau khi sort)
end

% ====================================================================
% CÁC HÀM CON (SUB-FUNCTIONS)
% ====================================================================

% --- Hàm 1: Khởi tạo vị trí ngẫu nhiên ---
function [xn, yn, Lightn] = init_ffa(n, range)
    xrange = range(2) - range(1);
    yrange = range(4) - range(3);
    
    % Tạo n vị trí ngẫu nhiên trong phạm vi cho phép
    xn = rand(1, n) * xrange + range(1);
    yn = rand(1, n) * yrange + range(3);
    Lightn = zeros(size(yn)); % Khởi tạo mảng độ sáng
end

% --- Hàm 2: Xử lý Di chuyển ---
function [xn, yn] = ffa_move(xn, yn, Lightn, xo, yo, Lighto, alpha, gamma, range, beta0)
    ni = size(yn, 2); % Số lượng đom đóm i
    nj = size(yo, 2); % Số lượng đom đóm j
    
    % Vòng lặp lồng nhau (Nested Loops)
    for i = 1:ni
        for j = 1:nj
            
            % Tính khoảng cách Euclidean (r_ij)
            r = sqrt((xn(i) - xo(j))^2 + (yn(i) - yo(j))^2);
            
            % SO SÁNH: Nếu con j sáng hơn con i
            if Lightn(i) < Lighto(j)
                
                % Tính độ hấp dẫn: 
                beta = beta0 * exp(-gamma * r^2);
                
                % Cập nhật vị trí x và y
                xn(i) = xn(i).*(1 - beta) + xo(j).*beta + alpha.*(rand - 0.5);
                yn(i) = yn(i).*(1 - beta) + yo(j).*beta + alpha.*(rand - 0.5);
                
            end
        end 
    end 
    
    % Kiểm tra biên (nếu bay ra ngoài thì kéo lại)
    [xn, yn] = findrange(xn, yn, range);
end

% --- Hàm 3: Kiểm tra biên (Boundary Constraint) ---
function [xn, yn] = findrange(xn, yn, range)
    for i = 1:length(yn)
        if xn(i) <= range(1), xn(i) = range(1); end
        if xn(i) >= range(2), xn(i) = range(2); end
        if yn(i) <= range(3), yn(i) = range(3); end
        if yn(i) >= range(4), yn(i) = range(4); end
    end
end