function [output, filterMask, filterImage] = periodicNoise(imagePath, filterType, lowCutoff, highCutoff, distance, width)
    % Fungsi untuk menghilangkan periodic noise menggunakan frequency domain filtering
    % imagePath  : path ke file gambar
    % filterType : 'bandreject', 'bandpass', 'notch'
    % 
    % PARAMETER UNTUK SETIAP FILTER:
    % - BANDREJECT: distance, width (menggunakan D0=distance, W=width)
    % - BANDPASS  : lowCutoff, highCutoff
    % - NOTCH     : lowCutoff (sebagai jarak notch dari center)
    % 
    % OUTPUT:
    % output       : hasil filtering (cleaned image)
    % filterMask   : matrix H(u,v) dari filter
    % filterImage  : visualisasi 3D mesh graph
    
    % Order fixed = 2 untuk semua filter
    n = 1;
    
    % Baca gambar
    img = imread(imagePath);
    
    % Simpan info untuk rekonstruksi
    isColor = (size(img, 3) == 3);
    
    % Dapatkan ukuran untuk membuat filter
    if isColor
        [M, N, ~] = size(img);
    else
        [M, N] = size(img);
    end
    
    % Buat filter mask berdasarkan tipe
    switch filterType
        case 'bandreject'
            % Bandreject: menggunakan distance (D0) dan width (W)
            filterMask = createBandrejectFilter(M, N, distance, width, n);
            
        case 'bandpass'
            % Bandpass: menggunakan lowCutoff dan highCutoff
            filterMask = createBandpassFilter(M, N, lowCutoff, highCutoff, n);
            
        case 'notch'
            % Notch: menggunakan lowCutoff sebagai notch position
            filterMask = createNotchFilter(M, N, lowCutoff, n);
            
        otherwise
            error('filterType harus "bandreject", "bandpass", atau "notch"');
    end
    
    % Jika RGB, process setiap channel
    if isColor
        R = processChannelPeriodic(double(img(:,:,1)), filterMask);
        G = processChannelPeriodic(double(img(:,:,2)), filterMask);
        B = processChannelPeriodic(double(img(:,:,3)), filterMask);
        output = cat(3, R, G, B);
    else
        output = processChannelPeriodic(double(img), filterMask);
    end
    
    % Konversi ke uint8
    output = uint8(output);
    
    % Buat visualisasi 3D mesh graph
    filterImage = capturePeriodicNoiseVisualization(filterMask, filterType, ...
        lowCutoff, highCutoff, distance, width, n);
end

function filtered = processChannelPeriodic(channel, H)
    F = fft2(channel);
    F_shifted = fftshift(F);
    G_shifted = F_shifted .* H;
    G = ifftshift(G_shifted);
    filtered = ifft2(G);
    filtered = real(filtered);
    
    minVal = min(filtered(:));
    maxVal = max(filtered(:));
    
    if maxVal > minVal
        filtered = (filtered - minVal) / (maxVal - minVal);
        filtered = filtered .^ 0.7;
        filtered = filtered * 255;
    else
        filtered = filtered - minVal;
    end
    
    filtered = max(0, min(255, filtered));
end

%% ==================== BANDREJECT FILTER ====================

function H = createBandrejectFilter(M, N, D0, W, n)
    % Membuat Butterworth Bandreject Filter
    % M, N  : ukuran image
    % D0    : center frequency (distance from origin)
    % W     : bandwidth (width of rejected band)
    % n     : order (fixed = 2)
    % 
    % Rejects frequencies in band around D0 with width W
    % H(u,v) = 1 / (1 + [D(u,v)*W / (D²(u,v) - D0²)]^(2n))
    
    % Buat koordinat mesh
    [u, v] = meshgrid(1:N, 1:M);
    
    % Hitung center
    center_u = floor(N/2) + 1;
    center_v = floor(M/2) + 1;
    
    % Hitung jarak D(u,v) dari center
    D = sqrt((u - center_u).^2 + (v - center_v).^2);
    
    % Avoid division by zero
    D(D == 0) = 0.0001;
    
    % Butterworth Bandreject formula
    numerator = D * W;
    denominator = D.^2 - D0^2;
    
    % Handle denominator = 0 (at exactly D0)
    denominator(abs(denominator) < 0.0001) = 0.0001;
    
    H = 1 ./ (1 + (numerator ./ denominator).^(2*n));
end

%% ==================== BANDPASS FILTER ====================

function H = createBandpassFilter(M, N, D0, D1, n)
    if D1 <= D0
        error('highCutoff (D1) harus > lowCutoff (D0)');
    end

    [u, v] = meshgrid(1:N, 1:M);
    center_u = floor(N/2) + 1;
    center_v = floor(M/2) + 1;
    D = sqrt((u - center_u).^2 + (v - center_v).^2);

    H_low = exp(-(D.^2) ./ (2 * D1^2));
    H_high = exp(-(D.^2) ./ (2 * D0^2));
    
    H = H_low .* (1 - H_high);
    
    H = max(0, min(1, H));
end


%% ==================== NOTCH FILTER ====================

function H = createNotchFilter(M, N, D_notch, n)
    [u, v] = meshgrid(1:N, 1:M);
    center_u = floor(N/2) + 1;
    center_v = floor(M/2) + 1;
    H = ones(M, N);
    
    W_notch = max(10, D_notch * 0.3);
    
    notch_u1 = center_u + D_notch;
    notch_v1 = center_v + D_notch;
    
    notch_u2 = center_u - D_notch;
    notch_v2 = center_v - D_notch;
    
    D1 = sqrt((u - notch_u1).^2 + (v - notch_v1).^2);
    D2 = sqrt((u - notch_u2).^2 + (v - notch_v2).^2);
    
    D1(D1 < 0.01) = 0.01;
    D2(D2 < 0.01) = 0.01;
    
    H_notch1 = 1 ./ (1 + (W_notch ./ D1).^(2*n));
    H_notch2 = 1 ./ (1 + (W_notch ./ D2).^(2*n));
    
    H = H .* H_notch1 .* H_notch2;
end


%% ==================== VISUALIZATION ====================

function filterImage = capturePeriodicNoiseVisualization(H, filterType, D0, D1, distance, W, n)
    % Membuat visualisasi 3D mesh graph dari filter
    
    % Buat figure invisible
    figHandle = figure('Visible', 'off', 'Position', [0 0 800 600]);
    
    % Ambil ukuran filter
    [M, N] = size(H);
    
    % Ambil subset untuk visualisasi
    visualSize = min(200, min(M, N));
    center_v = floor(M/2) + 1;
    center_u = floor(N/2) + 1;
    
    half_size = floor(visualSize/2);
    row_start = max(1, center_v - half_size);
    row_end = min(M, center_v + half_size);
    col_start = max(1, center_u - half_size);
    col_end = min(N, center_u + half_size);
    
    H_subset = H(row_start:row_end, col_start:col_end);
    
    % Buat koordinat untuk plotting
    [rows, cols] = size(H_subset);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Plot 3D mesh
    mesh(X, Y, H_subset);
    
    % Atur properti grafik
    colormap(jet);
    colorbar;
    
    % Labels dan title
    xlabel('u (Horizontal Frequency)');
    ylabel('v (Vertical Frequency)');
    zlabel('H(u,v) - Filter Response');
    
    % Title berdasarkan tipe
    switch filterType
        case 'bandreject'
            titleStr = sprintf('Bandreject Filter\n(D_0=%d, W=%d, n=%d)', ...
                round(distance), round(W), n);
        case 'bandpass'
            titleStr = sprintf('Bandpass Filter\n(D_0=%d, D_1=%d, n=%d)', ...
                round(D0), round(D1), n);
        case 'notch'
            titleStr = sprintf('Notch Filter\n(D_{notch}=%d, n=%d)', ...
                round(D0), n);
    end
    title(titleStr);
    
    % Viewing angle
    view(-37.5, 30);
    
    % Grid
    grid on;
    
    % Axis properties
    zlim([0 1]);
    
    % Lighting
    lighting gouraud;
    shading interp;
    
    % Capture
    frame = getframe(figHandle);
    filterImage = frame.cdata;
    
    % Close
    close(figHandle);
end