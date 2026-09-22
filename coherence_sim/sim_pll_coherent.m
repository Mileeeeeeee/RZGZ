%% sim_pll_coherent.m — 相干测距测振 FPGA 鉴相+锁相 方案仿真验证 (1550 nm)
% 系统参数: λ=1550nm | 基频40MHz | ADC 1GHz | 帧长100us(1e5点) | 帧率10kHz | Kp=10Hz/度(可调)
%
% 信号链路: AD0(偏差1+振动) × AD1(偏差2+40MHz) → FPGA 数字正交混频(I/Q 两路)
%           差频 IF = (偏差1-偏差2) + 振动 - 40MHz = IF0 + f_d(t)
%           注: 必须 I/Q 复混频, 否则 ±多普勒符号折叠、负频无法分辨
%
% 鉴相锁相(每帧 100us 执行):
%   ① FFT 找峰 → 带符号粗频 f_coarse (分辨率 10kHz)
%   ② IQ 解调(参考=上一帧估计 f_demod) → 相位 ψ_iq, 恢复参考无关相位 ψ_corr=ψ_iq+π·f_demod·T
%      (demod 时间原点=帧起点, Dirichlet 相位对应帧中点)
%   ③ 帧间相位差解缠 → 精频 f_est (双分支):
%      · 连续跟踪: f_est = f_prev + wrap(Δψ_corr-2π·f_prev·T)/(2πT)  半bin±π模糊由上一帧估计解除
%      · 频率阶跃: |f_coarse-f_prev|>10kHz 时 粗频重捕获 f_est = f_coarse + wrap(Δψ_corr-2π·f_coarse·T)/(2πT)
%   ④ 环路: 残差 f_res = f_est - f_track
%      · 原稿字面  Δf = Kp·ψ_w(包裹相位) → 临界稳定(等幅振荡/失锁), 见测试2
%      · 解缠修正  Δf = Kp·360°·f_res·T → 一阶稳定环, 环增益 a = Kp·360·T = 0.36
%      · 完善版    II 型数字锁相环: NCO 带相位累加器
%                  e = wrap(ψ_PD - φ_NCO(t_mid)),  ψ_PD = 在 f_track 处 IQ 解调相位
%                  φ_NCO ← φ_NCO + 2π·f_track·T + Kp_pll·e   (相位锁)
%                  f_track ← f_track + Kp·360°·f_res·T + Kf_pll·e  (频率跟踪)
%   ⑤ 测量输出 = f_est (10kHz 采样, 帧平均多普勒)
%   宽频振动(PM 指数 β=4πA/λ≫π, 或帧内 FM>5kHz)时 f_est 帧率估计器失效(见测试4),
%   改用直接相位解调: 混频至 IF0 → 块平均降采样(~1MHz) → atan2 → 解缠 → x=λφ/4π
%
% 运行: 在已打开的 MATLAB 中执行
%       cd('E:\matlab\coherence_sim'); sim_pll_coherent
%       约 30~60 秒, 输出 8 张 PNG 与结果表
clear; close all; clc; rng(0);

%% ───────────────────── 0. 参数 ─────────────────────
lambda = 1550e-9;           % 光波长 m
fs     = 1e9;               % ADC 采样率 Hz
T      = 100e-6;            % 帧长 s
N      = fs*T;              % 每帧点数 = 100000
df_bin = fs/N;              % FFT 分辨率 = 10 kHz
f_base = 40e6;              % DA 基频 Hz
IF0    = 500e3;             % 混频标称差频(工作点) Hz
Kp_hz  = 10;                % 比例系数 Hz/度 (可调)
a_FLL  = Kp_hz*360*T;       % FLL 环增益 = 0.36
Kp_pll = 0.2;               % PLL 相位累加增益 (完善版)
Kf_pll = 300;               % Hz/rad, PLL 频率跟踪增益 (完善版)
c0     = 3e8;               % 光速 m/s
phi0   = 0.7;               % 随机初相 rad
SNRdB  = 40;                % 默认信噪比 dB
wpi    = @(x) mod(x+pi,2*pi)-pi;
tfr    = (0:N-1)'/fs;       % 帧内时间轴
outdir = fileparts(mfilename('fullpath'));
if isempty(outdir), outdir = pwd; end
cfg0 = struct('lambda',lambda,'fs',fs,'T',T,'N',N,'IF0',IF0,...
              'Kp_hz',Kp_hz,'Kp_pll',Kp_pll,'Kf_pll',Kf_pll,'phi0',phi0);

%% ───────────── 测试1: 冷启动捕获 + 全范围阶跃 (10Hz~±1MHz) ─────────────
fprintf('\n===== 测试1: 恒定多普勒 捕获与稳态精度 (FLL+PLL, SNR=%g dB) =====\n',SNRdB);
fprintf('%-14s | %-10s | %-10s | %-10s | %s\n','f_d 真值(Hz)','f_est偏差','f_track偏差','f_est RMS','锁定帧数');
fd_list = [0 10 100 1e3 1e4 1e5 1e6 -1e6];
for i = 1:numel(fd_list)
    cfg = cfg0; cfg.fd_mode='const'; cfg.fd0=fd_list(i); cfg.Nf=120;
    cfg.mode='FLLPLL'; cfg.SNRdB=SNRdB;
    S = run_case(cfg);
    err = S.f_track(2:end) - S.f_true;                 % 每帧环路跟踪误差
    lock_n = find(abs(err)<1e3,1);                      % 首次进入±1kHz
    if isempty(lock_n) || any(abs(err(lock_n:end))>1e3), lock_n=nan; end
    ss = 30;                                            % 稳态取最后30帧
    be = mean(S.f_est(end-ss:end)-S.f_true(end-ss:end));
    bt = mean(S.f_track(end-ss+1:end)-S.f_true(end-ss:end));
    rmse = std(S.f_est(end-ss:end)-S.f_true(end-ss:end));
    fprintf('%-14.1f | %-10.1f | %-10.1f | %-10.1f | %s\n', ...
        fd_list(i), be, bt, rmse, num2str(lock_n));
    if fd_list(i)==1e6, S1 = S; end                     % 留作画图
end
figure('Color','w','Position',[60 60 900 380]);
plot((0:S1.Nf-1)*T*1e3, S1.f_est/1e6, '.-', 'DisplayName','f_{est} (开环测量)'); hold on;
plot((0:S1.Nf-1)*T*1e3, S1.f_true/1e6, 'k--', 'DisplayName','真值');
plot((0:S1.Nf)*T*1e3, S1.f_track/1e6, '-', 'LineWidth',1.2, 'DisplayName','f_{track} (NCO/环路)');
xlabel('时间 (ms)'); ylabel('差频 IF (MHz)'); title('测试1: 1 MHz 多普勒阶跃 捕获与锁定');
legend('Location','southeast'); grid on;
exportgraphics(gcf, fullfile(outdir,'fig1_step_track.png'),'Resolution',150);

%% ───────────── 测试2: 环路模式对比 (原稿字面 vs 解缠修正 vs 完善版) ─────────────
fprintf('\n===== 测试2: 环路模式对比 (恒定 f_d=100kHz, SNR=%g dB) =====\n',SNRdB);
modes = {'PLLwrap','FLL','FLLPLL'};
names = {'原稿字面 Δf=Kp·ψ_w(包裹相位)','解缠修正 Δf=Kp·360°·f_res·T','完善版 II型PLL(相位累加)'};
figure('Color','w','Position',[60 60 900 380]);
for i = 1:3
    cfg = cfg0; cfg.fd_mode='const'; cfg.fd0=1e5; cfg.Nf=300;
    cfg.mode=modes{i}; cfg.SNRdB=SNRdB;
    S = run_case(cfg);
    plot((0:S.Nf)*T*1e3, S.f_track/1e3, 'LineWidth',1.2, 'DisplayName',names{i}); hold on;
    err = S.f_track(100:end) - S.f_true(99:end);
    fprintf('  %-28s 稳态误差: 均值 %8.1f Hz, 峰峰 %8.1f Hz\n', ...
        names{i}, mean(err), max(err)-min(err));
end
plot([0 30],[100 100],'k--','DisplayName','真值 100 kHz');
xlabel('时间 (ms)'); ylabel('f_{track} (kHz)'); title('测试2: 三种环路模式对比');
legend('Location','southeast'); grid on;
exportgraphics(gcf, fullfile(outdir,'fig2_loop_modes.png'),'Resolution',150);

%% ───────────── 测试3: Kp 可调性与增益上限 ─────────────
fprintf('\n===== 测试3: Kp 可调性 (解缠修正环, 恒定 f_d=100kHz) =====\n');
fprintf('  增益 a = Kp[Hz/度]·360·T, 一阶环稳定要求 a < 2, 即 Kp < %.1f Hz/度\n', 2/(360*T));
figure('Color','w','Position',[60 60 900 380]);
for Kp_test = [10 30 55]
    cfg = cfg0; cfg.fd_mode='const'; cfg.fd0=1e5; cfg.Nf=200;
    cfg.mode='FLL'; cfg.Kp_hz=Kp_test; cfg.SNRdB=SNRdB;
    S = run_case(cfg);
    plot((0:S.Nf)*T*1e3, S.f_track/1e3,'LineWidth',1.2,...
        'DisplayName',sprintf('Kp = %d Hz/度 (a=%.2f)',Kp_test,Kp_test*360*T)); hold on;
    err = abs(S.f_track(2:end)-S.f_true);
    fprintf('  Kp=%2d Hz/度 (a=%.2f): 锁定帧数(±1kHz) = %d\n', ...
        Kp_test, Kp_test*360*T, find(err<1e3,1));
end
plot([0 20],[100 100],'k--','DisplayName','真值 100 kHz');
xlabel('时间 (ms)'); ylabel('f_{track} (kHz)'); title('测试3: Kp 可调性 (10/30/55 Hz/度)');
legend('Location','southeast'); grid on;
exportgraphics(gcf, fullfile(outdir,'fig3_kp_tuning.png'),'Resolution',150);

%% ───────────── 测试4: 正弦振动跟踪 (测振) ─────────────
A_vib = 10e-6;  f_vib = 1e3;              % 位移幅值 10um @ 1kHz
fd_max = 4*pi*f_vib*A_vib/lambda;         % 多普勒峰值 = 81.1 kHz
fprintf('\n===== 测试4: 正弦振动 位移%g um @ %g Hz → 多普勒峰值 %.1f kHz =====\n',...
    A_vib*1e6, f_vib, fd_max/1e3);
cfg = cfg0; cfg.fd_mode='sine'; cfg.A=A_vib; cfg.fv=f_vib; cfg.Nf=500;
cfg.mode='FLLPLL'; cfg.SNRdB=SNRdB;
S = run_case(cfg);
t_fr = (0:S.Nf-1)'*T;                     % 帧起始时刻 (列向量)
fd_meas = S.f_est.' - IF0;                % 多普勒测量 (10kHz 采样, 帧平均)
fd_avg  = fd_max*sinc(f_vib*T)*cos(2*pi*f_vib*t_fr);   % 帧平均真值 (S/H 等效)
rms_fd = sqrt(mean((fd_meas-fd_avg).^2));
fprintf('  多普勒测量 RMS 误差 = %.1f Hz (峰值 %.1f kHz, 相对 %.3f%%)\n',...
    rms_fd, fd_max/1e3, 100*rms_fd/fd_max);
fprintf('  (β=%.0f rad 宽频 PM: 帧内 FM=%.1f kHz 远超 5kHz 准静态极限, f_est 失效→用直接相位解调)\n',...
    4*pi*A_vib/lambda, 2*pi*f_vib*fd_max*T/1e3);

figure('Color','w','Position',[60 60 900 380]);
plot(t_fr*1e3, fd_avg/1e3,'k-','LineWidth',1.5,'DisplayName','多普勒真值(帧平均)'); hold on;
plot(t_fr*1e3, fd_meas/1e3,'.','MarkerSize',6,'DisplayName','f_{est} 测量 (10kHz 采样)');
plot(t_fr*1e3,(S.f_track(1:end-1).'-IF0)/1e3,'r-','LineWidth',1.2,'DisplayName','f_{track} (环路)');
xlabel('时间 (ms)'); ylabel('多普勒频偏 (kHz)');
title(sprintf('测试4: 正弦振动跟踪 (10um@1kHz, 多普勒峰值 %.1f kHz)',fd_max/1e3));
legend('Location','northeast'); grid on;
exportgraphics(gcf, fullfile(outdir,'fig4_vib_freq.png'),'Resolution',150);

% 速度/位移重构
v_true = lambda*fd_avg/2;
v_meas = lambda*fd_meas/2;
x_true = A_vib*sin(2*pi*f_vib*t_fr);
x_meas = lambda/2*cumtrapz(t_fr, fd_meas);
figure('Color','w','Position',[60 60 900 380]);
subplot(2,1,1);
plot(t_fr*1e3, v_true*1e3,'k-'); hold on; plot(t_fr*1e3, v_meas*1e3,'.');
xlabel('时间 (ms)'); ylabel('速度 (mm/s)'); title('速度重构 v = λ·f_d/2'); grid on;
subplot(2,1,2);
plot(t_fr*1e3, x_true*1e6,'k-'); hold on; plot(t_fr*1e3, x_meas*1e6,'.');
xlabel('时间 (ms)'); ylabel('位移 (um)'); title('位移重构 x = λ/2·∫f_d·dt'); grid on;
legend('真值','测量');
exportgraphics(gcf, fullfile(outdir,'fig5_vib_vel_disp.png'),'Resolution',150);

% ── 直接相位解调测振 (完善版方案): 宽频 PM 时 f_est 帧率估计器失效的解法 ──
% 混频至 IF0 → 1MHz 块平均降采样 → atan2 → 解缠 → x=λφ/4π (不依赖 FFT/帧长)
BLK=1000; NBLK=floor(N/BLK);              % 1GHz → 1MHz 降采样 (1us 块)
M=cfg.Nf*NBLK;
t_dd=zeros(1,M); x_dd=zeros(1,M);
unw=0; prev=nan;
for n=1:cfg.Nf
    tn=(n-1)*T; t=tn+tfr;
    pm=(4*pi*A_vib/lambda)*sin(2*pi*f_vib*t);
    z=exp(1j*(pm+phi0));
    N0=10^(-SNRdB/10);
    z=z+sqrt(N0/2)*(randn(N,1)+1j*randn(N,1));
    for b=1:NBLK
        ps=angle(sum(z((b-1)*BLK+1:b*BLK)));
        if isnan(prev), prev=ps; end
        unw=unw+wpi(ps-prev); prev=ps;
        idx=(n-1)*NBLK+b;
        t_dd(idx)=tn+(b-0.5)*BLK/fs;
        x_dd(idx)=lambda*unw/(4*pi);
    end
end
t0_dd=t_dd(1);                             % 相位解调只能测相对位移(AC), 与 AC 参考比较
x_ref=A_vib*(sin(2*pi*f_vib*t_dd)-sin(2*pi*f_vib*t0_dd));
rms_x=sqrt(mean((x_dd-x_ref).^2));
fprintf('  直接相位解调: x(t) 重构 RMS = %.2f nm (位移幅值 %g um)\n', rms_x*1e9, A_vib*1e6);
figure('Color','w','Position',[60 60 900 380]);
plot(t_dd*1e3, x_ref*1e6,'k-','LineWidth',1.5,'DisplayName','真值(AC)'); hold on;
plot(t_dd*1e3, x_dd*1e6,'.','MarkerSize',3,'DisplayName','直接相位解调 x=λφ/4π');
xlabel('时间 (ms)'); ylabel('位移 (um)');
title(sprintf('测试4b: 直接相位解调测振 (10um@1kHz, RMS %.2f nm)', rms_x*1e9));
legend('Location','northeast'); grid on;
exportgraphics(gcf, fullfile(outdir,'fig4b_direct_demod.png'),'Resolution',150);

%% ───────────── 测试5: 噪声性能 ─────────────
fprintf('\n===== 测试5: 噪声性能 (恒定 f_d=100kHz, FLL+PLL) =====\n');
fprintf('%-10s | %-12s | %-12s\n','SNR(dB)','σ_fest(Hz)','σ_track(Hz)');
SNR_list = [0 10 20 30 40];
sig_f = zeros(size(SNR_list));
figure('Color','w','Position',[60 60 700 400]);
for i = 1:numel(SNR_list)
    cfg = cfg0; cfg.fd_mode='const'; cfg.fd0=1e5; cfg.Nf=200;
    cfg.mode='FLLPLL'; cfg.SNRdB=SNR_list(i);
    S = run_case(cfg);
    ss = 100;
    sig_f(i) = std(S.f_est(end-ss:end)-S.f_true(end-ss:end));
    sig_t = std(S.f_track(end-ss+1:end)-S.f_true(end-ss:end));
    fprintf('%-10g | %-12.1f | %-12.1f\n', SNR_list(i), sig_f(i), sig_t);
end
loglog(SNR_list, sig_f, 'o-', 'LineWidth',1.2); grid on;
xlabel('SNR (dB)'); ylabel('频率估计标准差 (Hz)');
title('测试5: 开环频率估计精度 vs SNR');
exportgraphics(gcf, fullfile(outdir,'fig6_noise.png'),'Resolution',150);

%% ───────────── 测试6: 多普勒过零 (线性扫频 -600k→+600kHz, 30MHz/s) ─────────────
fd0=-600e3; fd1=600e3; Nf6=400;
mu=(fd1-fd0)/(Nf6*T);                     % 30 MHz/s (帧内变化 3 kHz < 5 kHz 准静态)
fprintf('\n===== 测试6: 多普勒过零 (线性扫频 %d→%d kHz, %.0f MHz/s, %d帧) =====\n',...
    fd0/1e3, fd1/1e3, mu/1e6, Nf6);
cfg = cfg0; cfg.fd_mode='sweep'; cfg.fd0=fd0; cfg.mu=mu; cfg.Nf=Nf6;
cfg.mode='FLLPLL'; cfg.SNRdB=SNRdB;
S = run_case(cfg);
t_fr = (0:Nf6-1)'*T;
fd_meas = S.f_est.' - IF0;
fd_avg = fd0 + mu*t_fr;                   % 帧平均频率 = 帧起始瞬时频率 (线性扫频)
fprintf('  过零(扫频)跟踪 RMS 误差 = %.1f Hz\n', sqrt(mean((fd_meas-fd_avg).^2)));
figure('Color','w','Position',[60 60 900 380]);
plot(t_fr*1e3, (IF0+fd_avg)/1e6,'k-','LineWidth',1.5,'DisplayName','真值'); hold on;
plot(t_fr*1e3, S.f_est.'/1e6,'.','MarkerSize',5,'DisplayName','f_{est}');
plot([0 40],[0 0],'r--','DisplayName','零频线');
xlabel('时间 (ms)'); ylabel('差频 IF (MHz)'); title('测试6: 多普勒过零跟踪 (复信号带符号, 线性扫频)');
legend('Location','southwest'); grid on;
exportgraphics(gcf, fullfile(outdir,'fig7_zero_cross.png'),'Resolution',150);

%% ───────────── 测试7: 测距 (锁相相位-频率斜率法) ─────────────
fprintf('\n===== 测试7: 测距 相位-频率斜率法 (目标 75 m, DA0 扫频 1 MHz) =====\n');
R_true = 75; dF = 1e6; Nfr = 300;
cfg = cfg0; cfg.fd_mode='const'; cfg.fd0=0; cfg.Nf=Nfr; cfg.mode='FLLPLL'; cfg.SNRdB=SNRdB;
[Rhat, f_DA0, phiR] = run_range(cfg, R_true, dF, f_base);
fprintf('  R_true = %g m,  R_hat = %.4f m,  误差 = %.3f m\n', R_true, Rhat, Rhat-R_true);
figure('Color','w','Position',[60 60 900 380]);
plot((f_DA0-f_base)/1e6, phiR/pi*180, '.-');
xlabel('DA0 扫频偏移 (MHz)'); ylabel('测距相位 (度)');
title(sprintf('测试7: 测距相位-频率斜率 (R=75 m, 拟合 R=%.4f m)', Rhat)); grid on;
exportgraphics(gcf, fullfile(outdir,'fig8_ranging.png'),'Resolution',150);

fprintf('\n仿真完成, 图已保存至 %s\n', outdir);

%% ═══════════════════════ 局部函数 ═══════════════════════

function S = run_case(cfg)
% 单次锁相环路仿真: 每帧 FFT 粗频 → IQ 解调相位 → 帧间相位差解缠 → 环路更新
lambda=cfg.lambda; fs=cfg.fs; T=cfg.T; N=cfg.N; df_bin=fs/N;
IF0=cfg.IF0; Kp_hz=cfg.Kp_hz; Kp_pll=cfg.Kp_pll; Kf_pll=cfg.Kf_pll; Nf=cfg.Nf;
mode=cfg.mode; SNR=cfg.SNRdB; phi0=cfg.phi0;
tfr=(0:N-1)'/fs; wpi=@(x) mod(x+pi,2*pi)-pi;
f_track=zeros(1,Nf+1); f_est=zeros(1,Nf); f_true=zeros(1,Nf); psi_w=zeros(1,Nf);
psi_corr_prev=0; f_est_prev=0; phi_nco=0;
for n=1:Nf
    tn=(n-1)*T; t=tn+tfr; t_mid=tn+T/2;
    % ── 信号模型: 数字正交混频后的复差频 ──
    switch cfg.fd_mode
        case 'const'
            z=exp(1j*(2*pi*(IF0+cfg.fd0)*t+phi0));
            fd=cfg.fd0;
        case 'sine'
            pm=(4*pi*cfg.A/lambda)*sin(2*pi*cfg.fv*t);      % 位移引起的相位调制
            z=exp(1j*(2*pi*IF0*t+pm+phi0));
            fd=(4*pi*cfg.fv*cfg.A/lambda)*cos(2*pi*cfg.fv*t_mid);
        case 'sweep'
            z=exp(1j*(2*pi*(IF0*t+cfg.fd0*t+0.5*cfg.mu*t.^2)+phi0));
            fd=cfg.fd0+cfg.mu*tn;                           % 帧平均频率=帧起始瞬时频率
    end
    if isfinite(SNR)
        N0=10^(-SNR/10);
        z=z+sqrt(N0/2)*(randn(N,1)+1j*randn(N,1));
    end
    % ── ① FFT 找峰 → 带符号粗频 (10kHz 分辨率) ──
    X=fft(z); [~,k]=max(abs(X));
    f_coarse=(k-1)*df_bin;
    if f_coarse>fs/2, f_coarse=f_coarse-fs; end
    % ── ② IQ 解调(参考=上一帧估计) → 参考无关相位 ψ_corr ──
    if n==1, f_demod=f_coarse; else, f_demod=f_est_prev; end
    psi_iq1=angle(sum(z.*exp(-1j*2*pi*f_demod*tfr)));
    psi_corr=psi_iq1+pi*f_demod*T;           % demod 时间原点=帧起点, 帧中点在本地时间 T/2
    % ── ③ 帧间相位差解缠 → 精频 (双分支) ──
    if n==1
        f_est(n)=f_coarse;
    else
        dpsi_c=wpi(psi_corr-psi_corr_prev-2*pi*f_est_prev*T);
        fe_c=f_est_prev+dpsi_c/(2*pi*T);
        if abs(f_coarse-f_est_prev)<=1e4
            f_est(n)=fe_c;                  % 连续跟踪: 半bin±π模糊用上一帧估计解除
        else                                % 频率阶跃(>1 bin) → 粗频重捕获
            dpsi_a=wpi(psi_corr-psi_corr_prev-2*pi*f_coarse*T);
            f_est(n)=f_coarse+dpsi_a/(2*pi*T);
        end
    end
    psi_corr_prev=psi_corr;
    f_est_prev=f_est(n);
    % ── ④ 环路 ──
    f_res=f_est(n)-f_track(n);
    switch mode
        case 'PLLwrap'
            % 原稿字面: 包裹相位×Kp, 绝对时间基准、无相位累加
            psi_iq2=angle(sum(z.*exp(-1j*2*pi*f_est(n)*tfr)));
            psi_w(n)=wpi(psi_iq2+pi*f_est(n)*T-2*pi*f_track(n)*t_mid);  % 恢复信号相位(参考无关)
            df=Kp_hz*(180/pi)*psi_w(n);
        case 'FLL'
            % 解缠修正: ψ_deg = 360·f_res·T (帧间累积相位)
            df=Kp_hz*360*f_res*T;
        case 'FLLPLL'
            % 完善版: II 型锁相环 — NCO 频率处 IQ 解调鉴相 + NCO 相位累加
            psi_pd=angle(sum(z.*exp(-1j*2*pi*f_track(n)*tfr)));
            e_pll=wpi(psi_pd-(phi_nco+2*pi*f_track(n)*T/2));   % 帧中点相位误差
            psi_w(n)=e_pll;
            phi_nco=phi_nco+2*pi*f_track(n)*T+Kp_pll*e_pll;    % 相位锁
            df=Kp_hz*360*f_res*T;                              % FLL 主环
            if abs(f_res)<1e4, df=df+Kf_pll*e_pll; end         % PLL 频率跟踪(捕获带内)
    end
    % ── ⑤ NCO 频率字更新 ──
    f_track(n+1)=f_track(n)+df;
    f_true(n)=IF0+fd;
end
S.f_track=f_track; S.f_est=f_est; S.f_true=f_true; S.psi_w=psi_w; S.Nf=Nf;
end

function [Rhat, f_DA0, phiR] = run_range(cfg, R, dF, f_base)
% 测距: DA0 扫频 dF, 每帧扣除已知扫频相位 2π·IF·t, 剩余相位斜率 → R = c·Δφ/(4π·ΔF)
lambda=cfg.lambda; fs=cfg.fs; T=cfg.T; N=cfg.N; df_bin=fs/N;
Nf=cfg.Nf; SNR=cfg.SNRdB; phi0=cfg.phi0; c0=3e8;
tfr=(0:N-1)'/fs; wpi=@(x) mod(x+pi,2*pi)-pi;
f_DA0=f_base+(0:Nf-1)*dF/(Nf-1);          % DA0 扫频 (信号路径)
rem=zeros(1,Nf);
for n=1:Nf
    tn=(n-1)*T; t=tn+tfr;
    IFn=f_DA0(n)-f_base;                   % 当前差频(已知, DA 命令给定)
    z=exp(1j*(2*pi*IFn*t+4*pi*R*f_DA0(n)/c0+phi0));
    if isfinite(SNR)
        N0=10^(-SNR/10);
        z=z+sqrt(N0/2)*(randn(N,1)+1j*randn(N,1));
    end
    % 在已知扫频 IF 处解调(残差为零, demod 时间原点=帧起点 → 解调相位=帧起始相位)
    psi_iq2=angle(sum(z.*exp(-1j*2*pi*IFn*tfr)));
    rem(n)=wpi(psi_iq2-2*pi*IFn*tn);       % 扣除已知载波相位, 仅剩测距相位 4πR·f_DA0/c + φ0
end
phiR=zeros(1,Nf); phiR(1)=rem(1);
for n=2:Nf
    phiR(n)=phiR(n-1)+wpi(rem(n)-rem(n-1));
end
p=polyfit(f_DA0(:)-f_base, phiR(:), 1);
Rhat=p(1)*c0/(4*pi);
end
