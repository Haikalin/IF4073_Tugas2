function [outputImage, imageWithMotionBlur] = motionBlur(imagePath, len, angle, noiseVar)
    % MOTION BLUR RGB + WIENER RESTORATION TANPA FUNGSI FILT APAPUN
    % -------------------------------------------------------------
    % imagePath : path file gambar
    % len       : panjang motion blur
    % angle     : sudut blur (0 = horizontal)
    % noiseVar  : variansi noise (misal 0.0001)
    %
    % outputImage         : hasil restorasi (Wiener)
    % imageWithMotionBlur : gambar blur dengan noise

    % ==== BACA GAMBAR ====
    img = imread(imagePath);
    img = double(img);
    [M, N, C] = size(img);

    % ==== BUAT PSF (tanpa fungsi bawaan) ====
    PSF = createMotionBlurPSF(len, angle);
    [psfM, psfN] = size(PSF);

    % PAD PSF KE UKURAN GAMBAR
    PSF_padded = zeros(M, N);
    startM = floor((M - psfM) / 2) + 1;
    startN = floor((N - psfN) / 2) + 1;
    PSF_padded(startM:startM+psfM-1, startN:startN+psfN-1) = PSF;
    H = fft2(fftshift(PSF_padded));

    % ==== BLUR + NOISE + RESTORE PER CHANNEL ====
    blurred = zeros(M, N, C);
    restored = zeros(M, N, C);

    for ch = 1:C
        F = fft2(img(:,:,ch));
        G = F .* H;
        b = real(ifft2(G));

        % Tambah noise Gaussian (jika ada)
        if noiseVar > 0
            noise = sqrt(noiseVar) * randn(M, N) * 255;
            b = b + noise;
        end

        blurred(:,:,ch) = b;

        % Wiener restoration
        restoredFFT = wienerFilter(fft2(b), H, noiseVar, mean(img(:,:,ch), 'all'));
        restored(:,:,ch) = real(ifft2(restoredFFT));
    end

    % ==== NORMALISASI DAN KONVERSI KE UINT8 ====
    blurred = mat2gray(blurred) * 255;
    restored = mat2gray(restored) * 255;

    imageWithMotionBlur = uint8(blurred);
    outputImage = uint8(restored);
end

% ======================================================
function PSF = createMotionBlurPSF(len, theta)
    % Membuat motion blur PSF manual tanpa fungsi filt apapun
    len = max(1, round(len));
    theta = mod(theta, 180);
    theta_rad = theta * pi / 180;

    psfSize = 2 * len + 1;
    PSF = zeros(psfSize, psfSize);
    center = ceil(psfSize / 2);

    % Gambar garis lurus sesuai sudut
    for i = -len:len
        x = round(center + i * cos(theta_rad));
        y = round(center - i * sin(theta_rad));
        if x >= 1 && x <= psfSize && y >= 1 && y <= psfSize
            PSF(y, x) = 1;
        end
    end

    % Lakukan sedikit smoothing manual (gaussian kecil buatan)
    % agar tidak terlalu tajam -> mencegah ringing
    PSF = smooth2D(PSF);
    PSF = PSF / sum(PSF(:));
end

% ======================================================
function F_restored = wienerFilter(G, H, noiseVar, signalPower)
    K = 0.01 + noiseVar / (signalPower + eps);
    H_conj = conj(H);
    H_abs_sq = abs(H).^2;
    denominator = H_abs_sq + K;
    denominator(abs(denominator) < eps) = eps;
    F_restored = (H_conj ./ denominator) .* G;
end

% ======================================================
function A_smooth = smooth2D(A)
    % Smooth manual pakai kernel Gaussian kecil tanpa fungsi filt
    kernel = [1 2 1; 2 4 2; 1 2 1];
    kernel = kernel / sum(kernel(:));
    [m, n] = size(A);
    A_smooth = zeros(m, n);

    for i = 2:m-1
        for j = 2:n-1
            window = A(i-1:i+1, j-1:j+1);
            A_smooth(i,j) = sum(sum(window .* kernel));
        end
    end
end
