
%% Start measuring Execution Time
execution_start = tic;


%%
% EMG signal preprocessing

% Denoting channels

% A1A6datasample(:,1) = Channel 1 = Latissimus dorsi-1
% A1A6datasample(:,1) = Channel 2 = Latissimus dorsi-2
% A1A6datasample(:,1) = Channel 3 = Erector Spinae-1
% A1A6datasample(:,1) = Channel 4 = Erector Spinae-2
% A1A6datasample(:,1) = Channel 5 = Rectus Abdominal-1
% A1A6datasample(:,1) = Channel 6 = Rectus Abdominal-2
% A1A6datasample(:,1) = Channel 7 = Rectus femoris-1
% A1A6datasample(:,1) = Channel 8 = Rectus femoris-2


%% EMG signal preprocessing

% Load EMG data
timestamps = emg_data(:, 1);
emg_channels = emg_data(:, 2:end);

% Define sampling rate
sampling_rate = 1259.2593;
t = (0:length(timestamps)-1) / sampling_rate;

% Number of channels
num_channels = size(emg_channels, 2);

%% 1. Plot Raw EMG Data
figure;
for channel = 1:num_channels
    subplot(num_channels, 1, channel);
    plot(t, emg_channels(:, channel));
    title(['Raw EMG Channel ', num2str(channel)]);
    xlabel('Time (s)');
    ylabel('Amplitude');
end
sgtitle('Raw EMG Signals');

%% 2. Apply Hampel Filtering (Outlier Removal)
window_size = 5;
emg_filtered_hampel = zeros(size(emg_channels));
for i = 1:num_channels
    emg_filtered_hampel(:, i) = hampel(emg_channels(:, i), window_size);
end

% Plot before and after Hampel filtering
figure;
for i = 1:num_channels
    subplot(2, num_channels, i);
    plot(t, emg_channels(:, i));
    title(['Before Hampel - Channel ', num2str(i)]);
    
    subplot(2, num_channels, num_channels + i);
    plot(t, emg_filtered_hampel(:, i));
    title(['After Hampel - Channel ', num2str(i)]);
end
sgtitle('Hampel Filtering Effect');

%% 3. Apply Band-pass Filtering (20-450 Hz)
low_cutoff = 20;
high_cutoff = 450;
if high_cutoff >= (sampling_rate / 2)
    error('High cutoff frequency must be less than half the Nyquist frequency.');
end

[b, a] = butter(4, [low_cutoff, high_cutoff] / (sampling_rate / 2), 'bandpass');
emg_filtered = filtfilt(b, a, emg_filtered_hampel);

% Plot before and after filtering
figure;
for i = 1:num_channels
    subplot(2, num_channels, i);
    plot(t, emg_filtered_hampel(:, i));
    title(['Before Filter - Channel ', num2str(i)]);
    
    subplot(2, num_channels, num_channels + i);
    plot(t, emg_filtered(:, i));
    title(['After Filter - Channel ', num2str(i)]);
end
sgtitle('Band-pass Filtering Effect');

%% 4. Apply FFT (Frequency Analysis)
fft_size = 2^nextpow2(size(emg_filtered, 1));
freqs = (0:fft_size/2-1) * (sampling_rate / fft_size);
fft_values = abs(fft(emg_filtered, fft_size, 1)); 
fft_values = fft_values(1:fft_size/2, :); 

% Plot FFT
figure;
for i = 1:num_channels
    subplot(num_channels, 1, i);
    plot(freqs, fft_values(:, i));
    title(['FFT Spectrum - Channel ', num2str(i)]);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
end
sgtitle('FFT Analysis');

%% 5. Normalize Using MVC Peak in Frequency Domain
% Compute peak MVC in frequency domain (maximum FFT magnitude)
peak_freq_mvc = max(fft_values, [], 1); % MVC peak in the frequency domain for each channel

% Normalize FFT values by MVC peak
normalized_fft_values = fft_values ./ peak_freq_mvc;

% Plot Normalized FFT in the frequency domain
figure;
for i = 1:num_channels
    subplot(num_channels, 1, i);
    plot(freqs, normalized_fft_values(:, i));
    title(['Normalized FFT - Channel ', num2str(i)]);
    xlabel('Frequency (Hz)');
    ylabel('Normalized Magnitude');
end
sgtitle('Frequency Domain MVC Normalization');

%% 6. Compute Mean Frequency (Using Normalized FFT)
% Calculate power spectrum of the normalized FFT
normalized_power_spectrum = normalized_fft_values .^ 2;

% Check if the sum of the power spectrum is non-zero to avoid division by zero
power_sum = sum(normalized_power_spectrum, 1);
if any(power_sum == 0)
    warning('Some channels have zero power spectrum, resulting in undefined mean frequency.');
end

% Ensure freqs is a column vector for element-wise multiplication
freqs = freqs(:);

% Compute Mean Frequency for each channel
mean_freq_normalized = sum(freqs .* normalized_power_spectrum, 1) ./ power_sum;

%% 7. Compute Median Frequency (Using Normalized FFT)
% Calculate median frequency using the normalized power spectrum
median_freq_normalized = zeros(1, num_channels);
for i = 1:num_channels
    cum_power = cumsum(normalized_power_spectrum(:, i));
    % Find the frequency corresponding to 50% of the cumulative power
    median_freq_normalized(i) = freqs(find(cum_power >= cum_power(end) / 2, 1, 'first'));
end

%% Display Mean and Median Frequency Results (Normalized)
disp('Mean Frequency (Hz) per Channel (Normalized):');
disp(mean_freq_normalized);

disp('Median Frequency (Hz) per Channel (Normalized):');
disp(median_freq_normalized);



%% Measure Execution Time
elapsedTime = toc(execution_start);  % End the timer

%% Measure Memory Usage
memoryUsage = memory;  % Get memory usage after running the script

%% Compute FLOPs
num_samples = size(emg_channels, 1);
filt_operations = num_samples * num_channels * (length(b) + 1); % Filtering operations
fft_operations = num_samples * log2(num_samples) * num_channels; % FFT operations
flops_total = filt_operations + fft_operations;


%% Print Performance Metrics
fprintf('Execution Time: %.4f seconds\n', elapsedTime);
fprintf('Total Memory Used: %.4f MB\n', memoryUsage.MemUsedMATLAB / 1e6);
disp(['Estimated FLOPs: ', num2str(flops_total)]);
