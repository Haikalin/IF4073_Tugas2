function [output, filterMask, filterImage] = imageSmoothingFrequency(imagePath, cutOff, order, type)
    % Fungsi untuk melakukan image smoothing di ranah frekuensi
    % imagePath : path ke file gambar
    % cutOff    : cutoff frequency D0 (radius)
    % order     : order untuk Butterworth (n), diabaikan untuk ILPF dan GLPF
    % type      : jenis filter ("ILPF", "GLPF", atau "BLPF")
    % 
    % OUTPUT:
    % output       : hasil filtering (image)
    % filterMask   : matrix H(u,v) dari filter
    % filterImage  : image dari 3D mesh graph (untuk ditampilkan di uiimage)
    
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
    
    % Buat filter mask (ini akan menjadi output ke-2)
    filterMask = createFilter(M, N, cutOff, order, type);
    
    % Jika RGB, process setiap channel
    if isColor
        % Proses setiap channel secara terpisah
        R = processChannel(double(img(:,:,1)), filterMask);
        G = processChannel(double(img(:,:,2)), filterMask);
        B = processChannel(double(img(:,:,3)), filterMask);
        
        % Gabungkan kembali
        output = cat(3, R, G, B);
    else
        % Grayscale
        output = processChannel(double(img), filterMask);
    end
    
    % Konversi ke uint8
    output = uint8(output);
    
    filterImage = captureFilterVisualization(filterMask, cutOff, type, order);
end

function filtered = processChannel(channel, H)
    % Process satu channel image dengan filter H yang sudah dibuat
    % channel : matrix 2D (M x N)
    % H       : filter mask (M x N)
    
    F = fft2(channel);
    
    F_shifted = fftshift(F);
    
    G_shifted = F_shifted .* H;
    
    G = ifftshift(G_shifted);
    
    filtered = ifft2(G);
    
    filtered = real(filtered);
    
    filtered = max(0, min(255, filtered));
end

function H = createFilter(M, N, D0, n, type)
    % Membuat filter mask H(u,v)
    % M, N  : ukuran image
    % D0    : cutoff frequency
    % n     : order (untuk Butterworth)
    % type  : "ILPF", "GLPF", "BLPF"
    
    % Buat koordinat mesh untuk u dan v
    [u, v] = meshgrid(1:N, 1:M);
    
    % Hitung center dari frequency domain
    center_u = floor(N/2) + 1;
    center_v = floor(M/2) + 1;
    
    % Hitung jarak D(u,v) dari setiap point ke center
    D = sqrt((u - center_u).^2 + (v - center_v).^2);
    
    switch type
        case 'ILPF'
            % Ideal Low-Pass Filter
            % H(u,v) = 1 if D(u,v) <= D0, else 0
            H = double(D <= D0);
            
        case 'GLPF'
            % Gaussian Low-Pass Filter
            % H(u,v) = exp(-(D^2(u,v)) / (2*D0^2))
            H = exp(-(D.^2) / (2 * D0^2));
            
        case 'BLPF'
            % Butterworth Low-Pass Filter
            % H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
            
            % Avoid division by zero
            D(D == 0) = 0.0001;
            
            % Butterworth formula
            H = 1 ./ (1 + (D / D0).^(2*n));
            
        otherwise
            error('Type harus "ILPF", "GLPF", atau "BLPF"');
    end
end

function filterImage = captureFilterVisualization(H, D0, type, order)
    % Membuat visualisasi 3D mesh graph dari filter dan capture sebagai image
    % H     : filter mask (M x N)
    % D0    : cutoff frequency
    % type  : tipe filter
    % order : order (untuk display title)
    % 
    % OUTPUT:
    % filterImage : uint8 RGB image dari visualization
    
    figHandle = figure('Visible', 'off', 'Position', [0 0 800 600]);
    
    [M, N] = size(H);
    
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
    
    mesh(X, Y, H_subset);
    
    colormap(jet);
    colorbar;
    
    xlabel('u (Horizontal Frequency)');
    ylabel('v (Vertical Frequency)');
    zlabel('H(u,v) - Filter Response');
    
    if strcmp(type, 'BLPF')
        titleStr = sprintf('%s Filter (D_0=%d, n=%d)', type, round(D0), order);
    else
        titleStr = sprintf('%s Filter (D_0=%d)', type, round(D0));
    end
    title(titleStr);
    
    view(-37.5, 30);
    
    grid on;
    
    zlim([0 1]);
    
    lighting gouraud;
    shading interp;
    
    frame = getframe(figHandle);
    filterImage = frame.cdata;
    
    % Tutup figure
    close(figHandle);
end