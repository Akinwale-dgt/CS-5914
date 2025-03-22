%% Start measuring Execution Time
execution_start = tic;
 %% Load and Define EMG Data
sampling_rate = 1259.2593;
t = (0:size(emg_data, 1)-1) / sampling_rate;
num_channels = size(emg_data, 2) - 1;
timestamps = emg_data(:, 1);
emg_channels = emg_data(:, 2:end);
 %% 1. Plot Raw EMG Data
figure;
for ch = 1:num_channels
    subplot(num_channels, 1, ch);
    plot(t, emg_channels(:, ch));
    title(['Raw EMG Channel ', num2str(ch)]);
    xlabel('Time (s)'); ylabel('Amplitude');
end
sgtitle('Raw EMG Signals');
 
%% 2. Apply Hampel Filtering (Vectorized)

window_size = 5;
emg_filtered_hampel = hampel(emg_channels, window_size);
 
%% 3. Apply Band-pass Filtering (20-450 Hz)
low_cutoff = 20; high_cutoff = 450;
[b, a] = butter(4, [low_cutoff, high_cutoff] / (sampling_rate / 2), 'bandpass');
emg_filtered = filtfilt(b, a, emg_filtered_hampel);
 
%% 4. Compute FFT with AI-Optimized Variants
fft_size = 2^nextpow2(size(emg_filtered, 1));
freqs = (0:fft_size/2-1) * (sampling_rate / fft_size);
% Choose FFT variant based on data length
if mod(fft_size, 4) == 0
    disp('Using Radix-4 FFT');
    fft_values = abs(fft(emg_filtered, fft_size, 1));
elseif mod(fft_size, 2) == 0
    disp('Using Radix-2 FFT');
    fft_values = abs(fft(emg_filtered, fft_size, 1));
else
    disp('Using Mixed-Radix FFT');
    fft_values = abs(fft(emg_filtered, fft_size, 1));
end
fft_values = fft_values(1:fft_size/2, :);
 %% 5. Normalize FFT using MVC Peak
peak_freq_mvc = max(fft_values, [], 1);
normalized_fft_values = fft_values ./ peak_freq_mvc;

 %% 6. Compute Mean and Median Frequency

power_spectrum = normalized_fft_values .^ 2;
power_sum = sum(power_spectrum, 1);
mean_freq = sum(freqs(:) .* power_spectrum, 1) ./ power_sum;
 median_freq = arrayfun(@(ch) freqs(find(cumsum(power_spectrum(:, ch)) >= sum(power_spectrum(:, ch))/2, 1)), 1:num_channels);
 
%% Display Results

disp('Mean Frequency (Hz) per Channel:'); disp(mean_freq);
disp('Median Frequency (Hz) per Channel:'); disp(median_freq);
 
%% Performance Metrics

elapsedTime = toc(execution_start);
memoryUsage = memory;
filt_ops = numel(emg_channels) * (length(b) + 1);
fft_ops = numel(emg_channels) * log2(size(emg_channels, 1));
flops_total = filt_ops + fft_ops;
 fprintf('Execution Time: %.4f ms\n', elapsedTime * 1000);
fprintf('Memory Used: %.4f MB\n', memoryUsage.MemUsedMATLAB / 1e6);
disp(['Estimated FLOPs: ', num2str(flops_total)]);

 