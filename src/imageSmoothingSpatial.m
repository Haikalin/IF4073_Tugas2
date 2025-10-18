function output = imageSmoothingSpatial(imagePath, nMask, sigma, type)
    % Fungsi untuk melakukan image smoothing di ranah spasial
    % imagePath: path ke file gambar
    % nMask: ukuran kernel (n x n)
    % sigma: standar deviasi untuk Gaussian filter
    % type: jenis filter ("Mean" atau "Gaussian")
    
    if strcmp(type, "Mean")
        mask = ones(nMask, nMask) / (nMask * nMask);
        output = convolusion(imagePath, mask);
        
    elseif strcmp(type, "Gaussian")
        mask = generateGaussianKernel(nMask, sigma);
        output = convolusion(imagePath, mask);
        
    else
        error('Type harus "Mean" atau "Gaussian"');
    end
end

function kernel = generateGaussianKernel(n, sigma)
    center = floor(n/2);
    
    kernel = zeros(n, n);
    
    k = 1 / (2 * pi * sigma^2);
    
    % Generate Gaussian kernel
    for i = 1:n
        for j = 1:n
            % Hitung jarak dari center
            x = i - center - 1;
            y = j - center - 1;
            
            % Formula Gaussian 2D
            exponent = -(x^2 + y^2) / (2 * sigma^2);
            kernel(i, j) = k * exp(exponent);
        end
    end
end