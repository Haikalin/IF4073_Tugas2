function [output, filterMask, filterImage] = edgeDetection(imagePath, cutOff, order, type)
    % Edge Detection menggunakan High-Pass Filter di frequency domain
    % imagePath : path ke file gambar
    % cutOff    : cutoff frequency D0
    % order     : order untuk Butterworth (n)
    % type      : 'IHPF', 'GHPF', atau 'BHPF'
    
    % Step 1: Baca dan konversi gambar
    f = imread(imagePath);
    
    % Konversi ke grayscale jika color
    if size(f, 3) == 3
        f = rgb2gray(f);
    end
    
    f = im2double(f);
    [M, N] = size(f);
    
    % Step 2: Tentukan parameter padding
    P = 2 * M;
    Q = 2 * N;
    
    % Step 3: Bentuk citra padding
    fp = zeros(P, Q);
    fp(1:M, 1:N) = f;
    
    % Step 4: Transformasi Fourier
    F = fftshift(fft2(fp));
    
    % Step 5: Bangkitkan High-Pass Filter H
    H = createHighPassFilter(P, Q, cutOff, order, type);
    
    % Step 6: Kalikan F dengan H
    G = H .* F;
    G1 = ifftshift(G);
    
    % Step 7: Inverse FFT
    G2 = real(ifft2(G1));
    
    % Step 8: Potong kembali ke ukuran semula
    output = G2(1:M, 1:N);
    
    % Normalisasi output untuk display
    output = normalizeEdgeImage(output);
    
    % Generate filter mask untuk display
    filterMask = mat2gray(H);
    filterMask = uint8(filterMask * 255);
    
    % Generate 3D visualization
    filterImage = captureFilterVisualization(H, cutOff, type, order);
end

function H = createHighPassFilter(P, Q, D0, n, type)
    % Membuat High-Pass Filter di frequency domain
    
    % Set up range of variables
    u = 0:(P-1);
    v = 0:(Q-1);
    
    % Compute the indices for use in meshgrid
    idx = find(u > P/2);
    u(idx) = u(idx) - P;
    idy = find(v > Q/2);
    v(idy) = v(idy) - Q;
    
    % Compute the meshgrid arrays
    [V, U] = meshgrid(v, u);
    
    % Distance from center
    D = sqrt(U.^2 + V.^2);
    
    % Generate filter based on type
    switch type
        case 'IHPF'  % Ideal High-Pass Filter
            H = double(D > D0);
            
        case 'GHPF'  % Gaussian High-Pass Filter
            H = 1 - exp(-(D.^2) / (2 * D0^2));
            
        case 'BHPF'  % Butterworth High-Pass Filter
            D(D == 0) = eps;  % Avoid division by zero
            H = 1 ./ (1 + (D0 ./ D).^(2 * n));
            
        otherwise
            error('Type harus "IHPF", "GHPF", atau "BHPF"');
    end
    
    % Shift H untuk matching dengan fftshift
    H = fftshift(H);
end

function output = normalizeEdgeImage(img)
    % Normalisasi hasil edge detection
    
    % Ambil absolute value
    img = abs(img);
    
    % Normalisasi ke [0, 1]
    minVal = min(img(:));
    maxVal = max(img(:));
    
    if maxVal > minVal
        img = (img - minVal) / (maxVal - minVal);
    end
    
    % Tingkatkan kontras dengan histogram stretching
    % Gunakan percentile untuk robustness
    low_in = prctile(img(:), 1);
    high_in = prctile(img(:), 99);
    
    if high_in > low_in
        img = (img - low_in) / (high_in - low_in);
        img = max(0, min(1, img));
    end
    
    % Gamma correction untuk highlight edges
    img = img .^ 0.5;
    
    % Convert to uint8
    output = uint8(img * 255);
end

function filterImage = captureFilterVisualization(H, D0, type, order)
    % Visualisasi 3D mesh dari filter
    
    figHandle = figure('Visible', 'off', 'Position', [0 0 800 600]);
    
    [P, Q] = size(H);
    
    % Ambil subset untuk visualisasi (center region)
    visualSize = min(200, min(P, Q));
    center_p = floor(P/2) + 1;
    center_q = floor(Q/2) + 1;
    
    half_size = floor(visualSize/2);
    row_start = max(1, center_p - half_size);
    row_end = min(P, center_p + half_size);
    col_start = max(1, center_q - half_size);
    col_end = min(Q, center_q + half_size);
    
    H_subset = H(row_start:row_end, col_start:col_end);
    
    % Create mesh
    [rows, cols] = size(H_subset);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    mesh(X, Y, H_subset);
    colormap(jet);
    colorbar;
    
    xlabel('u (Horizontal Frequency)');
    ylabel('v (Vertical Frequency)');
    zlabel('H(u,v) - Filter Response');
    
    % Title
    if strcmp(type, 'BHPF')
        titleStr = sprintf('%s HPF (D_0=%d, n=%d)', type, round(D0), order);
    else
        titleStr = sprintf('%s HPF (D_0=%d)', type, round(D0));
    end
    title(titleStr);
    
    view(-37.5, 30);
    grid on;
    zlim([0 1]);
    lighting gouraud;
    shading interp;
    
    % Capture figure
    frame = getframe(figHandle);
    filterImage = frame.cdata;
    
    close(figHandle);
end