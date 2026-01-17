close all;
clear;
clc;

%% 1. KHOI TAO THAM SO (Setup)
theta = (-90:0.1:90-0.1)*pi/180; 
lambda = 1; 
M = 12; 

% Tao vector lai tia
A = generateSteeringVector(theta, M, lambda);
desDirs_c = 0.0;
W_ref = zeros(M, size(desDirs_c, 2));

%% 2. TAO MOI TRUONG (Quantized Response)
Q = 160; 
phi = 1;
eqDir = -1:phi/Q:1-phi/Q;
Aq = generateQuantizedArrResponse(M, eqDir);

%% 3. TAO MAU BUP SONG (Conventional & Desired)
[PdM, P_refGen, W0] = generateDesPattern(eqDir, sin(desDirs_c), Aq);
P_init = ones(size(eqDir));
PM = P_init;
alpha = sort([find(ismember(eqDir, eqDir(1:4:end))), find(PdM)]);

%% 4. CHAY CAC THUAT TOAN (Running Algorithms)
fprintf('Dang chay ILS...\n');
W_ILS = twoStepILS(50, alpha, Aq, W0, PM, PdM);

fprintf('Dang chay Standard FA...\n');
instr_std = [200, 500]; % [So dom dom, So vong lap]
W_Std = firefly_standard(instr_std, Aq, PdM, alpha, M);

fprintf('Dang chay Improved FA...\n');
instr_imp = [100, 800]; 
W_Imp = firefly_improve(instr_imp, Aq, PdM, alpha, M, eqDir, W0);

%% 5. TINH TOAN DAP UNG (Calculate Response in dB)
% Conventional
Resp_Conv = P_refGen;
dB_Conv = 10*log10(Resp_Conv/max(Resp_Conv));

% ILS
Resp_ILS = abs(W_ILS' * Aq);
dB_ILS = 10*log10(Resp_ILS/max(Resp_ILS));

% Standard FA
Resp_Std = abs(W_Std' * Aq);
dB_Std = 10*log10(Resp_Std/max(Resp_Std));

% Improved FA
Resp_Imp = abs(W_Imp' * Aq);
dB_Imp = 10*log10(Resp_Imp/max(Resp_Imp));

% Desired (chi lay mau de ve dau *)
dB_Des = 10*log10(PdM/max(PdM));

%% 6. VE DO THI (Plotting)
figure('Color', 'w', 'Position', [100, 100, 800, 600]);
hold on; box on; grid on;

% 1. Zero Reference Line (Duong den dam ngang so 0)
yline(0, 'k-', 'LineWidth', 1.5); 

% 2. Conventional ULA (Net dut mau den)
plot(eqDir, dB_Conv, 'k--', 'LineWidth', 1.2);

% 3. Standard FA (Mau xanh duong - Blue)
plot(eqDir, dB_Std, 'b-', 'LineWidth', 1.2);

% 4. Desired (Dau sao mau tim - Magenta)
% Ve thua ra mot chut de do roi mat
plot(eqDir(1:5:end), dB_Des(1:5:end), 'm*', 'MarkerSize', 4);

% 5. ILS (Mau do - Red)
plot(eqDir, dB_ILS, 'r-', 'LineWidth', 1.5);

% 6. Improved FA (Mau xanh la - Green - Ve day hon de noi bat)
plot(eqDir, dB_Imp, 'g-', 'LineWidth', 2.0);

% --- TRANG TRI ---
xlim([-1 1]);
ylim([-40 1]); % De du ra 1 xiu phia tren cho dep

xlabel('Equivalent directions', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('|A|, dB', 'FontSize', 12, 'FontWeight', 'bold');
title('Algorithm Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% Tao Legend giong hinh mau
legend({'Zero Ref', 'Conventional ULA', 'Standard FA', 'Desired', 'ILS', 'Improved FA'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'NumColumns', 3, ...
    'FontSize', 10);

hold off;