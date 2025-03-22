%% Start measuring Execution Time
execution_start = tic;

%% Load EMG Data
timestamps = gpuArray(emg_data(:, 1));
emg_channels = gpuArray(emg_data(:, 2:end));

%% Define Sampling Rate
sampling_rate = 1259.2593;
t = (0:length(timestamps)-1) / sampling_rate;
num_channels = size(emg_channels, 2);
num_samples = size(emg_channels, 1);

%% Define Optimal Thread Block Size
block_size = 256; % Start with 256, adjust based on profiling
num_blocks = ceil(num_samples / block_size);

%% Hampel Filtering (Optimized)
window_size = 5;
emg_filtered_hampel = gpuArray.zeros(size(emg_channels));
parfor i = 1:num_channels
    emg_filtered_hampel(:, i) = hampel(emg_channels(:, i), window_size);
end

%% Band-pass Filtering (Optimized)
low_cutoff = 20;
high_cutoff = 450;
if high_cutoff >= (sampling_rate / 2)
    error('High cutoff frequency must be less than half the Nyquist frequency.');
end
[b, a] = butter(4, [low_cutoff, high_cutoff] / (sampling_rate / 2), 'bandpass');
emg_filtered = gpuArray.zeros(size(emg_filtered_hampel));
parfor i = 1:num_channels
    emg_filtered(:, i) = filtfilt(b, a, emg_filtered_hampel(:, i));
end

%% FFT Computation (Optimized)
fft_size = 2^nextpow2(num_samples);
freqs = (0:fft_size/2-1) * (sampling_rate / fft_size);
fft_values = abs(fft(emg_filtered, fft_size, 1)); 
fft_values = fft_values(1:fft_size/2, :);

%% Normalize Using MVC Peak in Frequency Domain
peak_freq_mvc = max(fft_values, [], 1);
normalized_fft_values = fft_values ./ peak_freq_mvc;

%% Compute Mean Frequency
normalized_power_spectrum = normalized_fft_values .^ 2;
power_sum = sum(normalized_power_spectrum, 1);
freqs = freqs(:);
mean_freq_normalized = sum(freqs .* normalized_power_spectrum, 1) ./ power_sum;

%% Compute Median Frequency
median_freq_normalized = zeros(1, num_channels);
parfor i = 1:num_channels
    cum_power = cumsum(normalized_power_spectrum(:, i));
    median_freq_normalized(i) = freqs(find(cum_power >= cum_power(end) / 2, 1, 'first'));
end

%% Performance Metrics
elapsedTime = toc(execution_start);
memoryUsage = memory;
filt_operations = num_samples * num_channels * (length(b) + 1);
fft_operations = num_samples * log2(num_samples) * num_channels;
flops_total = filt_operations + fft_operations;

%% Print Performance Metrics
fprintf('Execution Time: %.4f ms\n', elapsedTime * 1000);
fprintf('Total Memory Used: %.4f MB\n', memoryUsage.MemUsedMATLAB / 1e6);
disp(['Estimated FLOPs: ', num2str(flops_total)]);
%%
gpuDevice;
g = gpuDevice;
fprintf('Max threads per block: %d\n', g.MaxThreadsPerBlock);