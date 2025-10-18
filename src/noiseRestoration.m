function [output, pictWithNoise] = noiseRestoration(imagePath, noiseType, filterType, nWindow, mean, variance, density)
    % Fungsi untuk noise restoration dengan berbagai filter
    % imagePath  : path ke file gambar
    % noiseType  : 'salt_pepper' atau 'gaussian'
    % filterType : 'min', 'max', 'median', 'arithmetic', 'geometric', 
    %              'harmonic', 'contraharmonic', 'midpoint', 'alpha_trimmed'
    % nWindow    : ukuran window filter (3, 5, 7, dll - harus ganjil)
    % mean       : mean untuk gaussian noise (biasanya 0)
    % variance   : variance untuk gaussian noise (misal 0.01)
    % density    : density untuk salt & pepper (misal 0.05 = 5%)
    % 
    % OUTPUT:
    % output         : hasil filtering (restored image)
    % pictWithNoise  : gambar dengan noise
    
    % Baca gambar
    img = imread(imagePath);
    isColor = (size(img, 3) == 3);
    
    % Tambahkan noise
    switch noiseType
        case 'salt_pepper'
            noisyImg = addSaltPepperNoise(img, density);
        case 'gaussian'
            noisyImg = addGaussianNoise(img, mean, variance);
        otherwise
            error('noiseType harus "salt_pepper" atau "gaussian"');
    end
    
    pictWithNoise = noisyImg;
    
    % Apply filter untuk menghilangkan noise
    if isColor
        % Process setiap channel
        R = applyFilter(double(noisyImg(:,:,1)), filterType, nWindow);
        G = applyFilter(double(noisyImg(:,:,2)), filterType, nWindow);
        B = applyFilter(double(noisyImg(:,:,3)), filterType, nWindow);
        output = cat(3, R, G, B);
    else
        output = applyFilter(double(noisyImg), filterType, nWindow);
    end
    
    output = uint8(output);
end

%% ==================== NOISE ADDITION ====================

function noisyImg = addSaltPepperNoise(img, density)
    % Menambahkan salt & pepper noise
    % density: probabilitas total noise (misal 0.05 = 5%)
    % Salt (white) = density/2, Pepper (black) = density/2
    
    [M, N, C] = size(img);
    noisyImg = img;
    
    % Salt density dan Pepper density
    saltDensity = density / 2;
    pepperDensity = density / 2;
    
    % Generate random matrix
    randMatrix = rand(M, N);
    
    for c = 1:C
        channel = noisyImg(:,:,c);
        
        % Add pepper (black pixels = 0)
        channel(randMatrix < pepperDensity) = 0;
        
        % Add salt (white pixels = 255)
        channel(randMatrix > (1 - saltDensity)) = 255;
        
        noisyImg(:,:,c) = channel;
    end
end

function noisyImg = addGaussianNoise(img, mean, variance)
    % Menambahkan gaussian noise
    % mean: rata-rata noise (biasanya 0)
    % variance: variance noise (misal 0.01)
    
    noisyImg = double(img);
    [M, N, C] = size(img);
    
    for c = 1:C
        % Generate gaussian noise
        noise = mean + sqrt(variance) * randn(M, N) * 255;
        
        % Add noise to channel
        channel = noisyImg(:,:,c) + noise;
        
        % Clip to [0, 255]
        channel = max(0, min(255, channel));
        
        noisyImg(:,:,c) = channel;
    end
    
    noisyImg = uint8(noisyImg);
end

%% ==================== FILTER APPLICATION ====================

function filtered = applyFilter(img, filterType, nWindow)
    % Apply filter ke image
    % img: channel tunggal (grayscale atau satu channel RGB)
    % filterType: jenis filter
    % nWindow: ukuran window (harus ganjil)
    
    if mod(nWindow, 2) == 0
        error('nWindow harus ganjil (3, 5, 7, ...)');
    end
    
    [M, N] = size(img);
    filtered = zeros(M, N);
    pad = floor(nWindow / 2);
    
    % Padding image (replicate)
    imgPadded = padarray(img, [pad pad], 'replicate');
    
    % Apply filter ke setiap pixel
    for i = 1:M
        for j = 1:N
            % Extract window
            window = imgPadded(i:i+nWindow-1, j:j+nWindow-1);
            
            % Apply filter
            switch filterType
                case 'min'
                    filtered(i,j) = minFilter(window);
                case 'max'
                    filtered(i,j) = maxFilter(window);
                case 'median'
                    filtered(i,j) = medianFilter(window);
                case 'arithmetic'
                    filtered(i,j) = arithmeticMeanFilter(window);
                case 'geomethric'
                    filtered(i,j) = geometricMeanFilter(window);
                case 'harmonic'
                    filtered(i,j) = harmonicMeanFilter(window);
                case 'contraharmonic'
                    % Q parameter untuk contraharmonic (default = 1.5)
                    Q = 1.5;
                    filtered(i,j) = contraharmonicMeanFilter(window, Q);
                case 'midpoint'
                    filtered(i,j) = midpointFilter(window);
                case 'alpha_trimmed'
                    % d parameter untuk alpha-trimmed (delete d/2 from each end)
                    d = 4; % For 5x5, delete 2 lowest and 2 highest
                    filtered(i,j) = alphaTrimmedMeanFilter(window, d);
                otherwise
                    error('filterType tidak dikenal');
            end
        end
    end
    
    % Clip to [0, 255]
    filtered = max(0, min(255, filtered));
end

%% ==================== FILTER FUNCTIONS ====================

function val = minFilter(window)
    % Min Filter - ambil nilai minimum
    val = min(window(:));
end

function val = maxFilter(window)
    % Max Filter - ambil nilai maximum
    val = max(window(:));
end

function val = medianFilter(window)
    values = sort(window(:));
    n = length(values);
    
    if mod(n, 2) == 1
        % Odd number of elements
        val = values((n+1)/2);
    else
        % Even number of elements
        val = (values(n/2) + values(n/2 + 1)) / 2;
    end
end

function val = arithmeticMeanFilter(window)
    % Arithmetic Mean Filter
    val = mean(window(:));
end

function val = geometricMeanFilter(window)
    % Geometric Mean Filter
    % G = (prod(pixels))^(1/n)
    pixels = window(:);
    
    % Avoid log(0)
    pixels(pixels == 0) = 1;
    
    % Geometric mean using log
    val = exp(mean(log(pixels)));
end

function val = harmonicMeanFilter(window)
    % Harmonic Mean Filter
    % H = n / sum(1/pixels)
    pixels = window(:);
    
    % Avoid division by zero
    pixels(pixels == 0) = 1;
    
    n = length(pixels);
    val = n / sum(1 ./ pixels);
end

function val = contraharmonicMeanFilter(window, Q)
    % Contraharmonic Mean Filter
    % C = sum(pixels^(Q+1)) / sum(pixels^Q)
    % Q > 0: eliminates pepper noise
    % Q < 0: eliminates salt noise
    pixels = window(:);
    
    % Avoid issues with zero
    pixels(pixels == 0) = 1;
    
    numerator = sum(pixels .^ (Q + 1));
    denominator = sum(pixels .^ Q);
    
    if denominator == 0
        val = 0;
    else
        val = numerator / denominator;
    end
end

function val = midpointFilter(window)
    % Midpoint Filter
    % M = (min + max) / 2
    val = (min(window(:)) + max(window(:))) / 2;
end

function val = alphaTrimmedMeanFilter(window, d)
    % Alpha-Trimmed Mean Filter
    % Buang d/2 nilai terkecil dan d/2 nilai terbesar, lalu hitung mean
    values = sort(window(:));
    n = length(values);
    
    % Ensure d is valid
    if d >= n
        d = n - 1;
    end
    
    % Trim d/2 from each end
    trimAmount = floor(d / 2);
    
    if trimAmount > 0
        trimmedValues = values(trimAmount+1 : end-trimAmount);
    else
        trimmedValues = values;
    end
    
    val = mean(trimmedValues);
end